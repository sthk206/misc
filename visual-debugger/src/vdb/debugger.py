"""The stepping engine: a bdb.Bdb that only ever stops in the project's own code."""

import bdb
import builtins
import os
import runpy
import sys
import traceback

from .graph import GraphBuilder

LIBRARY_DIRS = ("site-packages", "dist-packages", ".venv", "venv", ".tox", ".nox", "node_modules")
VDB_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep


def flush_output():
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        try:
            stream.flush()
        except Exception:
            pass


class VisualDebugger(bdb.Bdb):
    def __init__(self, session, roots):
        super().__init__()
        self.session = session
        self.roots = [os.path.join(os.path.realpath(r), "") for r in roots]
        self.excluded = {VDB_DIR} | {
            os.path.join(os.path.realpath(p), "")
            for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)
        }
        self.step = 0
        self._user_files = {}
        self._leaving = None

    # Which code is "yours"

    def is_user_file(self, filename):
        known = self._user_files.get(filename)
        if known is None:
            known = self._user_files[filename] = self._classify(filename)
        return known

    def _classify(self, filename):
        if not filename or filename.startswith("<"):
            return False
        path = os.path.realpath(filename)
        if not any(path.startswith(root) for root in self.roots):
            return False
        if any(path.startswith(prefix) for prefix in self.excluded):
            return False
        return not set(path.split(os.sep)) & set(LIBRARY_DIRS)

    def stop_here(self, frame):
        # Library frames are never stepped through, so they are never traced either.
        if not self.is_user_file(frame.f_code.co_filename):
            return False
        return super().stop_here(frame)

    # Pausing

    def user_line(self, frame):
        if not frame.f_lineno:
            return
        if frame is self._leaving and not self.get_break(self.canonic(frame.f_code.co_filename), frame.f_lineno):
            return
        self._leaving = None
        stack, hidden_outer = self._user_stack((f, f.f_lineno) for f in self._walk_out(frame))
        for f, _, _ in stack:
            f.f_trace = self.trace_dispatch  # so "step out" can land in any caller
        self._interact(self._paused_state("paused", stack, hidden_outer), frame)

    def post_mortem(self, exc):
        tb, pairs = exc.__traceback__, []
        while tb is not None:
            pairs.append((tb.tb_frame, tb.tb_lineno))
            tb = tb.tb_next
        stack, hidden_outer = self._user_stack(reversed(pairs))
        state = self._paused_state("exception", stack, hidden_outer)
        state["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        self._interact(state, None)

    def _walk_out(self, frame):
        while frame is not None and frame is not self.botframe:
            yield frame
            frame = frame.f_back

    def _user_stack(self, pairs):
        """Keep only user frames (innermost first), counting the library frames between them."""
        stack, hidden = [], 0
        for frame, lineno in pairs:
            if self.is_user_file(frame.f_code.co_filename):
                stack.append((frame, lineno, hidden))
                hidden = 0
            else:
                hidden += 1
        return stack, hidden

    def _paused_state(self, status, stack, hidden_outer):
        self.step += 1
        snapshot = GraphBuilder(self.is_user_file).build(stack, hidden_outer)
        for frame in snapshot["frames"]:
            frame["file"] = self.canonic(frame["file"])
            self.session.readable_files.add(frame["file"])
        return {
            "status": status,
            "step": self.step,
            "root": self.roots[0],
            "snapshot": snapshot,
            "breakpoints": self._breakpoints(),
        }

    def _breakpoints(self):
        return {f: sorted(set(lines)) for f, lines in self.get_all_breaks().items()}

    def _interact(self, state, frame):
        self.session.publish(state)
        while True:
            command = self.session.next_command()
            action = command.get("action")
            if action in ("toggle_breakpoint", "set_breakpoints"):
                if action == "toggle_breakpoint":
                    self._toggle_breakpoint(command["file"], int(command["line"]))
                else:
                    self._set_breakpoints(command["breakpoints"])
                state = dict(state, breakpoints=self._breakpoints())
                self.session.publish(state)
                continue
            if action == "quit":
                if frame is not None:
                    self._stop_process()
                self.set_quit()
                break
            if frame is None:  # post-mortem: nothing left to run
                continue
            if action == "step_into":
                self.set_step()
            elif action == "step_over":
                self.set_next(frame)
            elif action == "step_out":
                self._step_out(frame)
            elif action == "continue":
                self.set_continue()
            else:
                continue
            break
        self.session.publish({"status": "running", "breakpoints": self._breakpoints()})

    def _stop_process(self):
        """Stop means stop: pytest and web frameworks would catch BdbQuit and keep going."""
        print("vdb: stopped", file=sys.stderr)
        flush_output()
        self.session.publish({"status": "finished", "exit_code": 1, "stopped": True})
        self.session.wait_until_delivered(timeout=2)
        os._exit(1)

    def _toggle_breakpoint(self, filename, line):
        filename = self.canonic(filename)
        if self.get_break(filename, line):
            self.clear_break(filename, line)
        else:
            self.set_break(filename, line)

    def _set_breakpoints(self, breakpoints):
        """Replace every breakpoint: the page restores its saved ones this way after a restart."""
        self.clear_all_breaks()
        for filename, lines in breakpoints.items():
            for line in lines:
                self.set_break(self.canonic(filename), int(line))  # a line that no longer exists is skipped

    def _step_out(self, frame):
        caller = frame.f_back
        if caller is not None and caller is not self.botframe and self.is_user_file(caller.f_code.co_filename):
            self.set_return(frame)
        else:
            # The caller is library code (pytest, a framework): stop at the next
            # line of your code that is not in this frame.
            self.set_step()
            self._leaving = frame

    def breakpoint_hook(self, *args, **kwargs):
        """Installed as sys.breakpointhook: breakpoint() pauses on the next line."""
        frame = sys._getframe(1)
        for f in self._walk_out(frame):
            if self.is_user_file(f.f_code.co_filename):
                f.f_trace = self.trace_dispatch
        self.set_step()
        sys.settrace(self.trace_dispatch)

    # Running the target

    def run_script(self, path, args):
        path = os.path.abspath(path)
        main = self._fresh_main({"__file__": path})
        sys.argv = [path, *args]
        sys.path[0] = os.path.dirname(path)
        with open(path, "rb") as f:
            code = compile(f.read(), path, "exec")
        self.run(code, main)

    def run_module(self, name, args):
        _, spec, code = runpy._get_module_details(name)
        main = self._fresh_main({
            "__file__": spec.origin, "__package__": spec.parent,
            "__loader__": spec.loader, "__spec__": spec,
        })
        sys.argv = [spec.origin, *args]
        self.run(code, main)

    def run_console_script(self, entry_point, args):
        """Run an installed command such as `skill-eval` the way its generated wrapper would."""
        self._fresh_main({})
        sys.argv = [entry_point.name, *args]
        source = (
            "import importlib, sys\n"
            f"target = importlib.import_module({entry_point.module!r})\n"
            f"for part in {entry_point.attr.split('.')!r}:\n"
            "    target = getattr(target, part)\n"
            "sys.exit(target())\n"
        )
        self.run(compile(source, f"<console script {entry_point.name}>", "exec"), sys.modules["__main__"].__dict__)

    @staticmethod
    def _fresh_main(extra):
        import __main__
        __main__.__dict__.clear()
        __main__.__dict__.update({"__name__": "__main__", "__builtins__": builtins, **extra})
        return __main__.__dict__
