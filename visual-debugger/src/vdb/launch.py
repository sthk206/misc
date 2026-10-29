"""The `vdb` command (run by bin/vdb): hands off to your project's Python with the debugger loaded.

Your code has to run in your project's interpreter to see its packages. So this
picks that interpreter (an explicit `python...` argument, else $VIRTUAL_ENV,
else ./.venv, else python3) and starts it with only the vdb package loaded
from this clone.

Imports nothing from vdb: it runs as a plain file under whatever python3 is on PATH.
"""

import os
import re
import shutil
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

BOOTSTRAP = """\
import importlib.util, sys
if sys.version_info < (3, 10):
    sys.exit("vdb needs Python 3.10 or newer; this project uses " + sys.version.split()[0])
spec = importlib.util.spec_from_file_location("vdb", {init!r}, submodule_search_locations=[{package!r}])
module = importlib.util.module_from_spec(spec)
sys.modules["vdb"] = module
spec.loader.exec_module(module)
import vdb.cli
sys.exit(vdb.cli.main(sys.argv[1:]))
"""


def is_interpreter(arg):
    return re.fullmatch(r"python[0-9.]*", os.path.basename(arg)) is not None


def project_python(args):
    """The interpreter to debug with, and the arguments left after removing it."""
    if args and is_interpreter(args[0]):
        return args[0], args[1:]
    for env in (os.environ.get("VIRTUAL_ENV"), ".venv"):
        if env and os.access(os.path.join(env, "bin", "python"), os.X_OK):
            return os.path.join(env, "bin", "python"), args
    return shutil.which("python3") or shutil.which("python") or "python3", args


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    split = next((i for i, a in enumerate(argv) if not a.startswith("--")), len(argv))
    options, rest = argv[:split], argv[split:]
    python, rest = project_python(rest)
    code = BOOTSTRAP.format(init=str(PACKAGE_DIR / "__init__.py"), package=str(PACKAGE_DIR))
    try:
        os.execvp(python, [python, "-c", code, *options, *rest])
    except FileNotFoundError:
        print(f"vdb: cannot find the Python interpreter {python!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
