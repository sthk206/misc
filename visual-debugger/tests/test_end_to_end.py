"""Runs bin/vdb on real programs and drives it through the same HTTP API the browser uses."""

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VDB = os.path.join(ROOT, "bin", "vdb")
SHAPES = os.path.join(ROOT, "examples", "shapes.py")


def line_of(path, text):
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if text in line:
                return n
    raise AssertionError(f"{text!r} not in {path}")


class Run:
    """One vdb process plus a tiny client for its API."""

    def __init__(self, *args, cwd=ROOT):
        self.proc = subprocess.Popen(
            [VDB, "--no-browser", "--port=0", sys.executable, *args],
            cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        banner = self.proc.stderr.readline()
        self.url = re.search(r"http://\S+", banner).group(0)
        self.version = 0

    def _request(self, path, body=None):
        req = urllib.request.Request(self.url + path.lstrip("/"), headers={"X-VDB": "1"})
        if body is not None:
            req.data = json.dumps(body).encode()
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as res:
            return json.loads(res.read())

    def wait(self, *statuses):
        while True:
            msg = self._request(f"/api/state?since={self.version}")
            self.version = msg["version"]
            if msg["state"]["status"] in statuses:
                return msg["state"]

    def command(self, action, until=("paused", "finished", "exception"), **extra):
        self._request("/api/command", {"action": action, **extra})
        return self.wait(*until)

    def finish(self):
        out, err = self.proc.communicate(timeout=30)
        return self.proc.returncode, out, err


def top(state):
    return state["snapshot"]["frames"][0]


def local(state, name, frame=0):
    return dict(state["snapshot"]["frames"][frame]["locals"])[name]


class EndToEndTest(unittest.TestCase):
    def test_stepping_through_shapes(self):
        run = Run(SHAPES)
        state = run.wait("paused")
        self.assertEqual(top(state)["file"], SHAPES)
        self.assertEqual(top(state)["line"], 1)

        append_line = line_of(SHAPES, "alias.append")
        state = run.command("toggle_breakpoint", until=("paused",), file=SHAPES, line=append_line)
        self.assertEqual(state["breakpoints"], {SHAPES: [append_line]})

        state = run.command("continue")
        self.assertEqual((top(state)["name"], top(state)["line"]), ("main", append_line))
        items = local(state, "items")
        self.assertEqual(local(state, "alias"), items, "alias and items must point at one object")
        self.assertEqual(state["snapshot"]["objects"][items["r"]]["len"], 2)
        cart = state["snapshot"]["objects"][local(state, "cart")["r"]]
        self.assertEqual(cart["kind"], "dict")

        state = run.command("step_over")
        self.assertEqual(state["snapshot"]["objects"][items["r"]]["len"], 3)
        milk = state["snapshot"]["objects"][items["r"]]["items"][2]
        self.assertEqual(state["snapshot"]["objects"][milk["r"]]["badge"], "dataclass")

        state = run.command("step_into")
        self.assertEqual(top(state)["name"], "build_chain")
        self.assertEqual(state["snapshot"]["frames"][1]["name"], "main")

        state = run.command("step_out")
        self.assertEqual(top(state)["name"], "main")
        self.assertEqual(top(state)["line"], append_line + 2)  # resumes after the call line

        state = run.command("continue")
        self.assertEqual(state, {"status": "finished", "exit_code": 0})
        code, out, _ = run.finish()
        self.assertEqual(code, 0)
        self.assertIn("total 3.95", out)

    def test_library_code_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "uses_json.py")
            with open(script, "w") as f:
                f.write(textwrap.dedent("""\
                    import json

                    def encode(obj):
                        return sorted(obj)

                    print(json.dumps({"s": {3, 1}}, default=encode))
                    done = True
                """))
            run = Run(script, cwd=tmp)
            run.wait("paused")
            run.command("step_over")  # import json
            run.command("step_over")  # def encode
            state = run.command("step_into")  # into json.dumps -> straight to encode()
            self.assertEqual(top(state)["name"], "encode")
            self.assertEqual(top(state)["hidden_below"], 0)
            self.assertGreater(state["snapshot"]["frames"][1]["hidden_below"], 0, "json's frames are counted, not shown")
            state = run.command("step_out")  # caller is library code: next user line is back at module level
            self.assertEqual((top(state)["name"], top(state)["line"]), ("module __main__", 7))
            run.command("continue")
            code, out, _ = run.finish()
            self.assertEqual((code, out.strip()), (0, '{"s": [1, 3]}'))

    def test_module_mode_breakpoint_call_and_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "crashy.py"), "w") as f:
                f.write(textwrap.dedent("""\
                    import sys

                    def explode(data):
                        breakpoint()
                        return data["missing"]

                    explode({"present": sys.argv[1]})
                """))
            run = Run("-m", "crashy", "hello", cwd=tmp)
            run.wait("paused")
            state = run.command("continue")  # no breakpoints: runs freely until breakpoint()
            self.assertEqual((top(state)["name"], top(state)["line"]), ("explode", 5))
            data = state["snapshot"]["objects"][local(state, "data")["r"]]
            self.assertEqual(data["entries"][0][1], {"p": "'hello'", "t": "str"})
            state = run.command("continue")
            self.assertEqual(state["status"], "exception")
            self.assertEqual(state["exception"]["type"], "KeyError")
            self.assertEqual(top(state)["line"], 5)
            state = run.command("quit", until=("finished",))
            self.assertEqual(state["exit_code"], 1)
            code, _, err = run.finish()
            self.assertEqual(code, 1)
            self.assertIn("KeyError", err)

    def test_restart_runs_the_program_again_in_the_same_process(self):
        run = Run(SHAPES)
        run.wait("paused")
        session = run._request("/api/hello")["session"]
        chain_line = line_of(SHAPES, 'print("chain starts at"')
        run.command("toggle_breakpoint", until=("paused",), file=SHAPES, line=chain_line)
        run.command("continue")  # prints the total, then pauses
        run._request("/api/command", {"action": "restart"})
        while True:  # same port, new session, once the fresh process is listening
            try:
                if run._request("/api/hello")["session"] != session:
                    break
            except (OSError, ValueError):
                pass
        run.version = 0
        state = run.wait("paused")
        self.assertEqual((top(state)["line"], state["breakpoints"]), (1, {}), "a fresh run; the page restores breakpoints")

        state = run.command("set_breakpoints", until=("paused",), breakpoints={SHAPES: [chain_line, 99999]})
        self.assertEqual(state["breakpoints"], {SHAPES: [chain_line]}, "lines that no longer exist are dropped")
        state = run.command("continue")
        self.assertEqual(top(state)["line"], chain_line)
        run.command("continue")
        code, out, err = run.finish()
        self.assertEqual(code, 0)
        self.assertEqual(out.count("total 3.95"), 2, "output from before the restart is kept")
        self.assertIn("vdb: restarting", err)

    def test_stop_ends_the_process_even_if_the_program_catches_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "stubborn.py")
            with open(script, "w") as f:
                f.write(textwrap.dedent("""\
                    try:
                        x = 1
                    except BaseException:
                        pass
                    print("kept running")
                """))
            run = Run(script, cwd=tmp)
            run.wait("paused")
            run.command("step_over")
            state = run.command("quit", until=("finished",))
            self.assertTrue(state["stopped"])
            code, out, _ = run.finish()
            self.assertEqual((code, out), (1, ""))


if __name__ == "__main__":
    unittest.main()
