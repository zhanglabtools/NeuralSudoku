"""Construct the published A07/A08 training commands using explicit portable paths."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from portable_common import CODE, PACKAGE, REVIEW


def command(args):
    cache, out = str(Path(args.cache).resolve()), str(Path(args.output).resolve())
    if args.workflow == 'e2':
        dims = {'graph': (208,384), 'hypergraph': (128,432), 'joint': (128,256)}
        d, hidden = dims[args.architecture]
        result = [sys.executable, str(REVIEW/'e2_train_production.py'), '--architecture', args.architecture,
                  '--cache_path', cache, '--output_dir', out, '--D',str(d),'--msg_hidden',str(hidden),
                  '--seed',str(args.seed),'--data_seed',str(args.seed),'--device',args.device,
                  '--validation-global-ids',str(Path(args.validation_ids).resolve()),
                  '--steps','50000','--batch_size','64','--train_T','32','--eval_T','64',
                  '--microbatch_size','16','--no-checkpoint-steps']
        if args.resume:
            result.append('--resume-if-exists')
        if args.max_new_updates:
            result.extend(['--max_new_updates',str(args.max_new_updates)])
        return result
    return [sys.executable,str(REVIEW/'a08_seed_pipeline.py'),'--seed',str(args.seed),'--stage',args.stage,
            '--backbone',str(Path(args.backbone).resolve()),'--cache-path',cache,'--output-dir',out,
            '--clean-test-ids',str(Path(args.clean_test_ids).resolve()),'--device',args.device]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='workflow', required=True)
    for name in ('e2','a08'):
        s = sub.add_parser(name)
        s.add_argument('--cache',required=True)
        s.add_argument('--output',required=True,help='Use a new reproduction directory, not supplied result records')
        s.add_argument('--seed',type=int,choices=(0,1,2),required=True)
        s.add_argument('--device',default='cuda')
        s.add_argument('--dry-run',action='store_true')
        if name == 'e2':
            s.add_argument('--architecture',choices=('graph','hypergraph','joint'),required=True)
            s.add_argument('--validation-ids',required=True)
            s.add_argument('--resume',action='store_true')
            s.add_argument('--max-new-updates',type=int,default=0,help='0=full run; bounded run keeps 50000-step protocol and pauses')
        else:
            s.add_argument('--backbone',required=True,help='Matching A07 joint best.pt, with finished.json beside it')
            s.add_argument('--clean-test-ids',required=True)
            s.add_argument('--stage',choices=('mine','train','evaluate','all'),default='all')
    args = p.parse_args()
    output = Path(args.output).resolve()
    for name in ('results','historical_results','checkpoints','data','code'):
        if output == PACKAGE/name or PACKAGE/name in output.parents:
            p.error('Choose a reproduction output directory outside supplied data/code/results.')
    cmd = command(args)
    if args.dry_run:
        print(json.dumps(dict(command=cmd,output_is_new_reproduction=True),indent=2))
        return
    if args.workflow == 'e2' and args.max_new_updates < 0:
        p.error('--max-new-updates must be nonnegative')
    raise SystemExit(subprocess.call(cmd,cwd=PACKAGE))


if __name__ == '__main__':
    main()
