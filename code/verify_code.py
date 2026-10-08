"""Lightweight syntax, manifest and ID-selection checks; does not load model weights."""
from pathlib import Path
import ast
import hashlib
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def main():
    files = sorted(ROOT.rglob('*.py'))
    for path in files:
        text = path.read_text(encoding='utf-8-sig')
        ast.parse(text, filename=str(path))
        compile(text, str(path), 'exec')
    manifest = json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        path = ROOT / record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['output_sha256'], record['path']
    import numpy as np
    from portable_eval import select_positions
    np.testing.assert_array_equal(select_positions(np.array([3,6,9,12]),np.array([12,6])),[1,3])
    for invalid in ([6,6],[6,7],[],[6.5]):
        try:
            select_positions(np.array([3,6,9,12]),np.asarray(invalid))
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid IDs were accepted: {invalid}')
    print(json.dumps(dict(status='passed',python_files=len(files),manifest_files=len(manifest['files']),
                          id_selection_cases=5,model_or_gpu_execution=False),indent=2))


if __name__ == '__main__':
    main()
