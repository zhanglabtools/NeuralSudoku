"""Run a supplied source entry with package-relative defaults and optional C5 base override."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from portable_common import CODE, PACKAGE, SOURCE, REVIEW, add_source_paths, file_path, load_symbolic_explicit, load_iterative_explicit


def available_scripts():
    paths = list(SOURCE.glob('*.py')) + list(REVIEW.glob('*.py'))
    return {path.stem: path for path in paths}


def main():
    p = argparse.ArgumentParser(description=__doc__, epilog='Place source arguments after --. Relative source arguments are relative to the attachment root.')
    p.add_argument('--backbone', help='Override serialized C5 base path; weights must match the reflection checkpoint')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('script', choices=sorted(available_scripts()))
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = p.parse_args()
    script = available_scripts()[args.script]
    rest = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if args.dry_run:
        print(json.dumps(dict(script=script.relative_to(CODE).as_posix(), working_directory='attachment root',
                              arguments=rest, explicit_backbone=args.backbone), indent=2))
        return
    backbone = file_path(args.backbone) if args.backbone else None
    add_source_paths()
    if backbone:
        import eval_symbolic_active_reflection
        import eval_iterative_hyper_reflection
        eval_symbolic_active_reflection.load_symbolic = lambda path, device: load_symbolic_explicit(path, backbone, device)
        eval_iterative_hyper_reflection.load_reflector = lambda path, device: load_iterative_explicit(path, backbone, device)
        eval_symbolic_active_reflection.load_reflector = eval_iterative_hyper_reflection.load_reflector
    os.chdir(PACKAGE)
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
