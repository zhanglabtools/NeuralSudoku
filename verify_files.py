"""Verify every file listed in this distribution with Python's standard library.

Run from the supplement directory: python verify_files.py
The manifest uses paths relative to this directory.
"""
from pathlib import Path, PurePosixPath
import hashlib
import sys


def main():
    supplement = Path(__file__).resolve().parent
    root = supplement
    manifest = supplement / 'MANIFEST.sha256'
    failures = []
    count = 0
    for line in manifest.read_text('utf-8').splitlines():
        if not line.strip():
            continue
        expected, relative = line.split('  ', 1)
        parts = PurePosixPath(relative)
        if parts.is_absolute() or '..' in parts.parts:
            raise ValueError('Invalid manifest path: ' + relative)
        path = root.joinpath(*parts.parts)
        if not path.is_file():
            failures.append('Missing: ' + relative)
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            failures.append('Hash mismatch: ' + relative)
        count += 1
    if failures:
        print('\n'.join(failures))
        print(f'FAILED: {len(failures)} issue(s).')
        return 1
    print(f'PASS: {count} files match this distribution manifest.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
