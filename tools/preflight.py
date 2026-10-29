"""Print what an install would change in this environment. Reads pip's --report JSON."""
import json
import sys
from importlib import metadata

report = json.load(open(sys.argv[1]))
installed = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in metadata.distributions()}

changes = []
for item in report.get('install', []):
    meta = item.get('metadata', {})
    name = meta.get('name', '').lower().replace('_', '-')
    new = meta.get('version', '')
    old = installed.get(name)
    if old is not None and old != new:
        changes.append((name, old, new))

if not changes:
    print('  no installed package would change')
else:
    print(f'  {len(changes)} installed package(s) would change:')
    for name, old, new in sorted(changes):
        print(f'    {name:28s} {old:14s} -> {new}')
sys.exit(1 if changes else 0)
