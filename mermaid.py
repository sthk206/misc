#!/usr/bin/env python3
"""Generate a Mermaid diagram from a project directory."""
import os, sys
 
IGNORE = {'.git','node_modules','__pycache__','.venv','venv','.idea','.vscode','dist','build'}
 
def scan(path, parent_id="root", lines=[], counter=[0]):
    if not lines:
        lines.append("graph LR")
        lines.append(f'  root["{os.path.basename(path) or path}/"]')
    try: entries = sorted(os.listdir(path))
    except PermissionError: return lines
    for e in entries:
        if e.startswith('.') or e in IGNORE: continue
        counter[0] += 1
        full = os.path.join(path, e)
        if os.path.isdir(full):
            nid = f"d{counter[0]}"
            lines.append(f'  {parent_id} --> {nid}["{e}/"]')
            scan(full, nid, lines, counter)
        elif os.path.isfile(full):
            lines.append(f'  {parent_id} --> f{counter[0]}("{e}")')
    return lines
 
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "project.mermaid"
    with open(out, "w") as f:
        f.write("\n".join(scan(os.path.abspath(target))))
    print(f"Done -> {out}")
 
