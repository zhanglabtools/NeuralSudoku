"""Exact standard Sudoku symmetry classes using a colored incidence graph.

Cells (81), rows+columns (18), boxes (9), symbols (9) are separate colors.
Each cell connects to its row, column and box and, if given, its symbol.
The row/column incidence is K(9,9) subdivided by cells, forcing a global
orientation preservation or transpose. Box incidence restricts permutations
to bands/stacks. Thus isomorphism encodes the usual Sudoku symmetry group.
No answer/solution is used to canonicalize the puzzle.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, multiprocessing as mp, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if (HERE/'vendor').exists(): sys.path.insert(0, str(HERE/'vendor'))
import pynauty

COLORS = [set(range(81)), set(range(81,99)), set(range(99,108)), set(range(108,117))]
BASE_ADJ = {i: [] for i in range(117)}
for i in range(81):
    r,c = divmod(i,9)
    for j in (81+r,90+c,99+(r//3)*3+c//3):
        BASE_ADJ[i].append(j); BASE_ADJ[j].append(i)

def graph(puzzle):
    p = np.asarray(puzzle).reshape(81)
    if not np.all((p >= 0) & (p <= 9)) or not np.all(p == p.astype(int)):
        raise ValueError('Expected integer symbols 0..9')
    adj = {i: x.copy() for i,x in BASE_ADJ.items()}
    for i in np.flatnonzero(p):
        j = 107+int(p[i]); adj[int(i)].append(j); adj[j].append(int(i))
    return pynauty.Graph(117, directed=False, adjacency_dict=adj, vertex_coloring=COLORS)

def certificate(puzzle): return pynauty.certificate(graph(puzzle))
def key(puzzle): return hashlib.sha256(certificate(puzzle)).digest()
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def save_json(path, obj):
    temp=Path(str(path)+'.tmp'); temp.write_text(json.dumps(obj, indent=2),encoding='utf-8'); temp.replace(path)

def transform(puzzle, rng):
    bands=rng.permutation(3); stacks=rng.permutation(3)
    rows=np.concatenate([3*b+rng.permutation(3) for b in bands])
    cols=np.concatenate([3*b+rng.permutation(3) for b in stacks])
    p=np.asarray(puzzle).reshape(9,9)[rows][:,cols]
    if rng.integers(2): p=p.T
    symbols=np.r_[0,rng.permutation(np.arange(1,10))]
    return symbols[p].reshape(81)

def self_test():
    started=time.perf_counter()
    rng=np.random.default_rng(20260929)
    solution=np.array([((r*3+r//3+c)%9)+1 for r in range(9) for c in range(9)])
    p=solution.copy(); p[rng.choice(81,53,replace=False)]=0
    original=certificate(p)
    for _ in range(150): assert certificate(transform(p,rng)) == original
    q=p.copy(); q[np.flatnonzero(q)[0]]=0
    assert certificate(q) != original, 'clue removal must not be equivalent'
    wrong=p.reshape(9,9).copy(); wrong[[0,3]]=wrong[[3,0]]
    assert certificate(wrong.reshape(81)) != original, 'cross-band row swap was not detected'
    group=pynauty.autgrp(graph(np.zeros(81,dtype=int)))
    observed=int(round(group[1]*(10**group[2])))
    expected=2*(math.factorial(3)**8)*math.factorial(9)
    assert observed == expected, (observed,expected)
    return {'positive_transform_tests':150,'negative_tests':2,'empty_grid_group_order':observed,'expected_group_order':expected,'elapsed_seconds':time.perf_counter()-started,'pynauty_version':pynauty.__version__}

_PUZZLES=None
def worker(task):
    start,stop=task
    out=np.empty(stop-start,dtype='V32')
    for j in range(start,stop): out[j-start]=np.void(key(_PUZZLES[j]))
    return start,stop,out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache-path'); ap.add_argument('--output-dir')
    ap.add_argument('--workers',type=int,default=8); ap.add_argument('--chunk-size',type=int,default=1000)
    ap.add_argument('--limit',type=int,default=0); ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.workers < 1 or args.chunk_size < 1 or args.limit < 0: ap.error('workers/chunk-size must be positive; limit must be nonnegative')
    checks=self_test(); print(json.dumps({'self_test':checks}),flush=True)
    if args.self_test: return
    if not args.cache_path or not args.output_dir: ap.error('cache-path and output-dir are required')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    global _PUZZLES
    with np.load(args.cache_path,allow_pickle=False) as data:
        _PUZZLES=data['puzzles']; splits=data['splits']
    if _PUZZLES.ndim != 2 or _PUZZLES.shape[1] != 81 or not np.issubdtype(_PUZZLES.dtype,np.integer): raise ValueError('Expected integer puzzles[N,81]')
    if splits.shape != (len(_PUZZLES),) or not np.issubdtype(splits.dtype,np.integer) or not np.all(np.isin(splits,[0,1,2])): raise ValueError('Invalid split array')
    if not np.all((_PUZZLES>=0)&(_PUZZLES<=9)): raise ValueError('Puzzle symbols out of range')
    if args.limit: _PUZZLES=_PUZZLES[:args.limit]; splits=splits[:args.limit]
    n=len(_PUZZLES)
    manifest={'cache_path':str(Path(args.cache_path).resolve()),'cache_sha256':sha256_file(args.cache_path),'script_sha256':sha256_file(__file__),'count':n,'limit':args.limit,'group':'digit permutations; row permutations within bands; band permutations; column permutations within stacks; stack permutations; transpose','pynauty_version':pynauty.__version__,'self_test':checks}
    mf=out/'manifest.json'
    if mf.exists():
        old=json.loads(mf.read_text())
        for f in ('cache_sha256','script_sha256','count','pynauty_version'):
            if old[f]!=manifest[f]: raise RuntimeError('Resume manifest mismatch: '+f)
    else: save_json(mf,manifest)
    progress=out/'progress.json'; done=json.loads(progress.read_text())['processed'] if progress.exists() else 0
    if not isinstance(done,int) or not 0 <= done <= n: raise ValueError('Invalid resume progress')
    datafile=out/'canonical_sha256.dat'
    if done and not datafile.exists(): raise RuntimeError('Missing resumed key array')
    keys=np.memmap(datafile,dtype='V32',mode='r+' if datafile.exists() else 'w+',shape=(n,))
    started=time.perf_counter(); completed_start=done
    if done<n:
        context=mp.get_context('fork')
        tasks=((i,min(n,i+args.chunk_size)) for i in range(done,n,args.chunk_size))
        with context.Pool(args.workers) as pool:
            for start,stop,values in pool.imap(worker,tasks,chunksize=1):
                keys[start:stop]=values; keys.flush(); done=stop
                elapsed=time.perf_counter()-started
                state={'processed':done,'total':n,'elapsed_this_session_seconds':elapsed,'rate_per_second':(done-completed_start)/max(elapsed,1e-9),'updated_utc':datetime.now(timezone.utc).isoformat()}
                save_json(progress,state)
                if done%10000==0 or done==n: print(json.dumps(state),flush=True)
    keys.flush()
    unique,first,inverse,counts=np.unique(keys,return_index=True,return_inverse=True,return_counts=True)
    masks=np.zeros(len(unique),dtype=np.uint8)
    np.bitwise_or.at(masks,inverse,np.left_shift(np.uint8(1),splits.astype(np.uint8)))
    cross=(masks & (masks-1)) != 0
    repeated_group_ids=np.flatnonzero(counts>1)
    duplicate_rows=np.flatnonzero(counts[inverse]>1)
    # Confirm all hash-equal groups with full canonical certificate bytes.
    certs={}
    for idx in duplicate_rows:
        gid=int(inverse[idx])
        if gid not in certs: certs[gid]=certificate(_PUZZLES[first[gid]])
        if certificate(_PUZZLES[idx])!=certs[gid]: raise RuntimeError('Hash collision detected')
    np.savez_compressed(out/'class_membership.npz',class_id=inverse.astype(np.int32),representative=first.astype(np.int32),count=counts.astype(np.int32),split_mask=masks)
    with (out/'duplicate_members.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['global_index','split','class_id','representative_index','class_size','split_mask','cross_split','canonical_sha256'])
        for idx in duplicate_rows:
            gid=int(inverse[idx]); writer.writerow([int(idx),int(splits[idx]),gid,int(first[gid]),int(counts[gid]),int(masks[gid]),int(cross[gid]),bytes(keys[idx]).hex()])
    summary={'status':'complete','count':n,'scope':'full_dataset' if not args.limit else 'prefix_pilot','unique_symmetry_classes':len(unique),'repeated_classes':len(repeated_group_ids),'rows_in_repeated_classes':len(duplicate_rows),'cross_split_classes':int(cross.sum()),'rows_in_cross_split_classes':int(cross[inverse].sum()),'test_rows_equivalent_to_train':int(((splits==2)&((masks[inverse]&1)!=0)).sum()),'test_rows_equivalent_to_validation':int(((splits==2)&((masks[inverse]&2)!=0)).sum()),'full_certificate_collision_check':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),'group':manifest['group']}
    save_json(out/'summary.json',summary); print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
