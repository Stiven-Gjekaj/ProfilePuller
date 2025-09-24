PYTHON ?= python3

.PHONY: setup build-index query

setup:
	$(PYTHON) - <<'PYCODE'
import os
import pathlib
import subprocess
import sys
import venv

venv_dir = pathlib.Path('.venv')
if not venv_dir.exists():
    print('Creating virtual environment in .venv')
    venv.EnvBuilder(with_pip=True).create(venv_dir)
else:
    print('Reusing existing .venv')
python = venv_dir / ('Scripts' if os.name == 'nt' else 'bin') / 'python'
subprocess.check_call([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'])
subprocess.check_call([str(python), '-m', 'pip', 'install', '-r', 'requirements.txt'])
activate_hint = '.venv\\\Scripts\\activate' if os.name == 'nt' else 'source .venv/bin/activate'
print(f'Environment ready. Activate with: {activate_hint}')
PYCODE

build-index:
	$(PYTHON) scripts/build_index.py --known faces/known --out-index faiss_index.bin --out-meta faiss_meta.json

query:
	$(PYTHON) - <<'PYCODE'
import subprocess
import sys
from pathlib import Path

query_dir = Path('faces/queries')
if not query_dir.exists():
    print('faces/queries/ does not exist. Add images and rerun.')
    sys.exit(0)

extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
files = [p for p in query_dir.rglob('*') if p.suffix.lower() in extensions]
if not files:
    print('No query images found in faces/queries/. Add an image and rerun.')
    sys.exit(0)

cmd = [sys.executable, 'scripts/query.py', str(files[0])]
print(f'Running: {" ".join(cmd)}')
sys.exit(subprocess.call(cmd))
PYCODE
