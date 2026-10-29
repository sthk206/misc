# vdb: a visual Python debugger

Step through any Python command line by line, like the VS Code debugger, but see
your data drawn as boxes and arrows: lists as rows of cells, dicts as key/value
tables, dataclasses and pydantic models as cards, DataFrames as tables, and an
arrow wherever two names point at the same object.

```bash
vdb examples/shapes.py                 # a script
vdb -m pytest tests/test_parser.py     # a module (pytest, your package, ...)
vdb python3.11 main.py --input x.csv   # a specific interpreter
uv run vdb skill-eval validate         # an installed command (a [project.scripts] entry)
```

It pauses on the first line of **your** code: files under the current folder,
never the venv, site-packages or the standard library. Library frames are
skipped when stepping and folded into "… 12 library frames" in the call stack.

## Install

No dependencies. Clone once and put `bin/vdb` on your PATH:

```bash
git clone https://github.com/<you>/visual-debugger.git ~/tools/visual-debugger
mkdir -p ~/.local/bin
ln -s ~/tools/visual-debugger/bin/vdb ~/.local/bin/vdb
```

To update: `git pull` in the clone. Nothing else.

`vdb` runs your code with your project's interpreter, so it sees your packages:
the active venv (`$VIRTUAL_ENV`), else `./.venv`, else `python3`. Name an
interpreter as the first argument to override. Python 3.10+.

## Controls

| Action | Button | Keys |
|---|---|---|
| Continue to the next breakpoint | ▶ Continue | `F5` or `c` |
| Step over | ↷ Over | `F10` or `n` |
| Step into | ↓ Into | `F11` or `s` |
| Step out | ↑ Out | `Shift+F11` or `o` |
| Stop the program | ■ Stop | `q` |
| Look at the previous / next step | ◀ ▶ | `[` and `]` |

- Click a line number to toggle a breakpoint.
- `breakpoint()` in your code pauses there too.
- Click a frame (in the call stack or the graph) to see its source line.
- Hover an object or a dot to highlight every arrow pointing at it.
- Variables and objects that changed since the previous step are highlighted.
- On an uncaught exception, vdb stays paused on the failing line so you can inspect it; press Stop to exit.
- The browser tab stays open between runs and reloads itself on the next `vdb` run.

## Options

```text
--port=N       port for the page (default 5679, falls back to any free port)
--root=DIR     treat DIR as your code instead of the current folder
--no-browser   do not open a browser tab
```

## Limits (for now)

- Only the main thread is debugged. For Flask use `flask run --no-reload --without-threads`.
- Objects of library classes (other than dataclasses, pydantic models, pandas and numpy) are shown as their `repr`, not expanded.
- Large containers show their first 40 items; a snapshot holds at most 300 objects.
- Tests that fail inside pytest are caught by pytest, so vdb does not stop on them. Set a breakpoint in the test instead.

## Layout

```text
bin/vdb                 the command you symlink: runs src/vdb/launch.py
src/vdb/launch.py       picks your project's interpreter and loads vdb into it
src/vdb/cli.py          argument parsing, runs the target, exit codes
src/vdb/debugger.py     bdb-based stepping that only stops in your code
src/vdb/graph.py        live objects -> JSON graph (nodes by id, inline scalars)
src/vdb/server.py       local HTTP server: long-poll state, post commands
src/vdb/highlight.py    Python syntax highlighting for the source pane
src/vdb/static/         the browser page (one HTML file, no build step)
```

## Tests

```bash
python3 -m unittest discover -s tests
```

The end-to-end tests run `bin/vdb` on real programs and drive it through the
same HTTP API the page uses.
