"""Command line entry: vdb [options] [python] script.py args... | -m module args..."""

import bdb
import importlib.metadata
import os
import sys
import traceback
import webbrowser

from .debugger import VisualDebugger
from .server import DebugServer, Session

USAGE = """\
usage: vdb [options] [python] script.py [args...]
       vdb [options] [python] -m module [args...]
       vdb [options] [python] command [args...]     (an installed console script, e.g. skill-eval)

Runs the command and pauses at the first line of your own code
(files under the current folder, not the venv or the standard library).

options:
  --port=N       port for the browser page (default 5679, falls back to any free port)
  --root=DIR     treat DIR as your code instead of the current folder
  --no-browser   do not open a browser tab
"""


class UsageError(Exception):
    pass


def parse(argv):
    options = {"port": 5679, "browser": True, "root": os.getcwd()}
    rest = list(argv)
    while rest and rest[0].startswith("--"):
        arg = rest.pop(0)
        if arg == "--no-browser":
            options["browser"] = False
        elif arg.startswith("--port="):
            options["port"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--root="):
            options["root"] = os.path.abspath(arg.split("=", 1)[1])
        else:
            raise UsageError(f"unknown option {arg}")
    if not rest:
        raise UsageError("nothing to run")
    if rest[0] == "-m":
        if len(rest) < 2:
            raise UsageError("-m needs a module name")
        return options, ("module", rest[1], rest[2:])
    return options, ("script", rest[0], rest[1:])


def find_console_script(name):
    for entry_point in importlib.metadata.entry_points(group="console_scripts"):
        if entry_point.name == name:
            return entry_point
    return None


def exit_code_of(exc):
    if exc.code is None:
        return 0
    if isinstance(exc.code, int):
        return exc.code
    print(exc.code, file=sys.stderr)
    return 1


def main(argv):
    try:
        options, (mode, target, args) = parse(argv)
    except UsageError as exc:
        print(f"vdb: {exc}\n\n{USAGE}", file=sys.stderr)
        return 2

    roots = [options["root"]]
    if mode == "script" and os.path.isfile(target):
        roots.append(os.path.dirname(os.path.abspath(target)))
    elif mode == "script":
        entry_point = find_console_script(target) if os.sep not in target else None
        if entry_point is None:
            print(f"vdb: {target} is not a file or an installed command", file=sys.stderr)
            return 2
        mode, target = "console_script", entry_point

    session = Session()
    server = DebugServer(session, options["port"])
    server.start()
    print(f"vdb: debugging at {server.url}", file=sys.stderr)
    # A tab left open from an earlier run reconnects by itself within a second.
    if options["browser"] and not session.client_seen.wait(1.5):
        webbrowser.open(server.url)

    debugger = VisualDebugger(session, roots)
    sys.breakpointhook = debugger.breakpoint_hook
    code = 0
    try:
        if mode == "module":
            debugger.run_module(target, args)
        elif mode == "console_script":
            debugger.run_console_script(target, args)
        else:
            debugger.run_script(target, args)
    except SystemExit as exc:
        code = exit_code_of(exc)
    except KeyboardInterrupt:
        code = 130
    except BaseException as exc:
        traceback.print_exc()
        print("vdb: stopped on an uncaught exception; inspect it in the browser, then press Stop.", file=sys.stderr)
        code = 1
        try:
            debugger.post_mortem(exc)
        except (bdb.BdbQuit, KeyboardInterrupt):
            pass
    finally:
        sys.settrace(None)

    session.publish({"status": "finished", "exit_code": code})
    session.wait_until_delivered(timeout=2)
    return code
