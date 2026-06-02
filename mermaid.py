#!/usr/bin/env python3
"""Generate a Mermaid diagram of .py files in a project."""
import os, sys
 
IGNORE = {'.git','node_modules','__pycache__','.venv','venv','.idea','.vscode','dist','build'}
 
def scan(path, pid="root", lines=[], c=[0]):
    try: entries = sorted(os.listdir(path))
    except PermissionError: return lines
    for e in entries:
        if e.startswith('.') or e in IGNORE: continue
        full = os.path.join(path, e)
        if os.path.isdir(full):
            prev = len(lines)
            c[0] += 1; nid = f"d{c[0]}"
            lines.append(f'  {pid} --> {nid}["{e}/"]')
            scan(full, nid, lines, c)
            if len(lines) == prev + 1: lines.pop()  # remove empty dirs
        elif e.endswith('.py'):
            c[0] += 1
            lines.append(f'  {pid} --> f{c[0]}("{e}")')
    return lines
 
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "project.mermaid"
    lines = ["graph LR", f'  root["{os.path.basename(os.path.abspath(target))}/"]']
    scan(os.path.abspath(target), "root", lines)
    with open(out, "w") as f: f.write("\n".join(lines))
    print(f"Done -> {out}")
 
