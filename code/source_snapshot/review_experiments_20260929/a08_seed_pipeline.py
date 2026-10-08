"""Independent C5 system per seed: A07 joint -> split0 bank -> C5 -> test.

Invoked by the scheduler; this file does not allocate GPUs or launch other
seeds. Mining and training resume atomically. No test data is loaded before
the evaluate stage. A07 backbone training is shared once per matching seed.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT))
import eval_d4_reflection_ablation as util

COEFFICIENTS=dict(best_ce_coef=1.,mean_ce_coef=.10,selected_ce_coef=.20,critic_coef=.15,
                 cycle_ce_coef=.35,progress_coef=0.,progress_margin=.002,diversity_coef=.02,
                 constraint_coef=.03,correction_reg_coef=.001,critic_temperature=.25,
                 critic_target_temperature=.05)


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    util.save_json(tmp,value); tmp.replace(path)


def atomic_npz(path,**values):
    tmp=path.with_suffix('.pending.npz')
    np.savez_compressed(tmp,**values); tmp.replace(path)


def atomic_torch(path,value):
    import torch
    tmp=path.with_suffix('.pending.pt')
    torch.save(value,tmp); tmp.replace(path)


def atomic_copy(source,target):
    temporary=target.with_suffix(target.suffix+'.copying')
    shutil.copyfile(source,temporary);temporary.replace(target)


def versioned_best(out,payload,step):
    """Save an immutable candidate; only a later resume.pt commits its reference."""
    temporary=out/'best_candidate.pending.pt';atomic_torch(temporary,payload)
    digest=util.sha_file(temporary)
    version=out/f'best_step_{step:07d}_{digest[:16]}.pt';temporary.replace(version)
    atomic_copy(version,out/'best.pt')
    return dict(path=version.name,sha256=digest,step=step)


def restore_committed_selection(out,selection,best_step,history):
    """Roll back public selection artifacts to the last atomic resume commit."""
    if best_step is None:
        if selection is not None:raise ValueError('Unselected resume has a best reference')
        if (out/'best.pt').exists():
            (out/'best.pt').replace(out/f'uncommitted_best_{time.time_ns()}.pt')
    else:
        if selection is None or selection['step']!=best_step:raise ValueError('Missing committed best reference')
        version=(out/selection['path']).resolve()
        if version.parent!=out.resolve() or util.sha_file(version)!=selection['sha256']:
            raise ValueError('Committed best version is missing or changed')
        atomic_copy(version,out/'best.pt')
    atomic_json(out/'validation_history.json',history)


def backbone_gate(args):
    import torch
    checkpoint=Path(args.backbone).resolve()
    finished=json.loads((checkpoint.parent/'finished.json').read_text())
    if finished['status']!='complete': raise ValueError('A07 backbone is not complete')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=payload['cfg']; train_args=payload['args']
    expected=dict(model_type='hybrid',D=128,msg_hidden=256,train_T=32,eval_T=64,force_clues=True,dropout=0.)
    if any(cfg.get(k)!=v for k,v in expected.items()): raise ValueError('Backbone configuration differs from A07 joint protocol')
    if payload.get('architecture')!='joint' or train_args['seed']!=args.seed or train_args['data_seed']!=args.seed:
        raise ValueError('A07 architecture/independent seed provenance mismatch')
    if train_args['steps']!=50000 or train_args['batch_size']!=64:
        raise ValueError('A07 backbone must have fixed 50k updates and batch64')
    if int(finished.get('best_step',payload['step']))!=int(payload['step']):
        raise ValueError('Use the validation-selected best checkpoint, not an arbitrary step')
    checkpoint_sha=util.sha_file(checkpoint)
    if finished.get('best_sha256')!=checkpoint_sha:
        raise ValueError('A07 best checkpoint changed after training completed')
    return dict(path=str(checkpoint),sha256=checkpoint_sha,
                finished_sha256=util.sha_file(checkpoint.parent/'finished.json'),seed=args.seed,
                training_updates=50000,training_exposure=50000*64,best_step=payload['step'],
                shared_with_A07=True,count_backbone_training_once=True)


def setup(args):
    import torch
    from sudoku_exchange_experiment import set_seed,set_torch_threads
    set_torch_threads(); set_seed(10000+args.seed)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.set_float32_matmul_precision('highest')
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    source=util.source_manifest()
    source[str(Path(__file__).resolve())]=util.sha_file(__file__)
    import train_c5_random_init_comparison as old
    source[str(Path(old.__file__).resolve())]=util.sha_file(old.__file__)
    cache_path=Path(args.cache_path).resolve()
    cache_files=sorted(cache_path.glob('*.npy')) if cache_path.is_dir() else [cache_path]
    cache_sha256={p.name:util.sha_file(p) for p in cache_files}
    protocol=dict(seed=args.seed,backbone=backbone_gate(args),cache_path=str(cache_path),cache_sha256=cache_sha256,
                  reflection_steps=args.train_steps,reflection_batch=args.batch_size,eval_every=args.eval_every,
                  evaluation_batch=args.eval_batch_size,mining_batch=args.mine_batch_size,save_every=args.save_every,
                  hidden=256,lr=1.5e-4,weight_decay=1e-4,grad_clip=1.,coefficients=COEFFICIENTS,
                  reflection_init_seed=10000+args.seed,train_sampler_seed=20000+args.seed,
                  validation_seed=1729,validation_all_n=args.val_all_n,validation_hard_n=args.val_hard_n,
                  source_sha256=source,smoke=args.smoke,bank_limit=args.bank_limit,
                  precision='FP32; AMP off; TF32 off',test_for_selection=False,
                  selection='max sum of first-valid-selected Exact on fixed validation hard/all pools; strict improvement only; no test')
    saved=out/'protocol.json'
    if saved.exists() and json.loads(saved.read_text())!=protocol:
        raise ValueError('Resume protocol changed; refuse to mix models, code, data or hyperparameters')
    if not saved.exists(): atomic_json(saved,protocol)
    return out,protocol,torch.device(args.device)


def mine(args,out,protocol,device):
    import torch
    from eval_hybrid_hyper_rrn_restarts import load_hybrid
    from sudoku_cache_utils import load_sudoku_dataset,load_sudoku_cache
    final=out/'failure_bank.npz'
    if final.exists() and (out/'failure_bank.json').exists():
        meta=json.loads((out/'failure_bank.json').read_text())
        if meta['status']!='complete' or meta['bank_sha256']!=util.sha_file(final) or meta['backbone_sha256']!=protocol['backbone']['sha256']:
            raise ValueError('Completed failure bank fingerprint mismatch')
        return
    model,_,_=load_hybrid(args.backbone,device)
    ds=load_sudoku_dataset(args.cache_path,split=0,limit=0)
    cache=load_sudoku_cache(args.cache_path)
    raw=np.flatnonzero(np.asarray(cache['splits'])==0)
    if hasattr(cache,'close'):cache.close()
    hard=np.flatnonzero(np.nan_to_num(ds.ratings,nan=-np.inf)>4).astype(np.int64)
    if args.bank_limit: hard=hard[:args.bank_limit]
    partial=out/'failure_bank.partial.npz'
    failures=[]; offset=0; valid_not_exact=0
    data_key=util.sha_array(hard,'<i8')+util.sha_array(raw,'<i8')
    if partial.exists():
        with np.load(partial,allow_pickle=False) as p:
            if str(p['data_key'])!=data_key: raise ValueError('Train selection changed while resuming mining')
            failures=p['indices'].tolist(); offset=int(p['scanned']); valid_not_exact=int(p['valid_not_exact'])
    started=time.perf_counter()
    with torch.no_grad():
        for start in range(offset,len(hard),args.mine_batch_size):
            take=hard[start:start+args.mine_batch_size]
            puzzle=torch.as_tensor(ds.puzzles[take],dtype=torch.long,device=device)
            target=torch.as_tensor(ds.solutions[take],dtype=torch.long,device=device)
            logits,_=model(puzzle,steps=64); pred=logits.argmax(-1)+1
            valid,_=util.torch_candidate_flags(pred,puzzle)
            exact=(pred==target).reshape(len(take),-1).all(1)
            valid_not_exact+=int((valid&~exact).sum())
            # Preserve the historical failure-bank criterion: target Exact failure.
            failures.extend(take[(~exact).cpu().numpy()].tolist())
            offset=start+len(take)
            atomic_npz(partial,indices=np.asarray(failures,dtype=np.int64),scanned=offset,
                        data_key=data_key,valid_not_exact=valid_not_exact)
            if offset%4096<len(take) or offset==len(hard):
                print(json.dumps(dict(stage='mine',seed=args.seed,scanned=offset,total=len(hard),failures=len(failures))),flush=True)
    if not failures: raise ValueError('Empty failure bank; no silent fallback to test or a different seed')
    atomic_npz(final,indices=np.asarray(failures,dtype=np.int64),raw_cache_ids=raw[failures],
                scanned_indices=hard,split=np.array(0),steps=np.array(64),strict_min_rating=np.array(4.0))
    atomic_json(out/'failure_bank.json',dict(status='complete',split=0,criterion='rating>4 AND target Exact failure at T64',
                test_used=False,scanned=len(hard),n=len(failures),valid_not_exact=valid_not_exact,
                backbone_sha256=protocol['backbone']['sha256'],bank_sha256=util.sha_file(final),data_key=data_key,
                elapsed_seconds=time.perf_counter()-started,smoke=args.smoke))


def evaluate_indices(model,ds,indices,device,batch_size,save_grids=False):
    import torch
    model.eval(); arrays={}
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            take=indices[start:start+batch_size]
            puzzle=torch.as_tensor(ds.puzzles[take],dtype=torch.long,device=device)
            target=torch.as_tensor(ds.solutions[take],dtype=torch.long,device=device)
            parent=model.encode_parent(puzzle)
            result=util.active_ablation_batch(model,puzzle,target,parent,'full')
            for key,value in result.items():
                if key=='selected_grid' and not save_grids: continue
                arrays.setdefault(key,[]).append(value.cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    exact=arrays['selected_exact'];valid=arrays['first_valid']
    return dict(n=len(indices),selected_exact_count=int(exact.sum()),selected_exact=float(exact.mean()),
                valid_count=int(valid.sum()),valid=float(valid.mean()),valid_not_exact=int((valid&~exact).sum())),arrays


def train(args,out,protocol,device):
    import torch
    from sudoku_cache_utils import load_sudoku_dataset
    from train_c5_random_init_comparison import build_model,trainable_fingerprint
    from train_symbolic_primal_dual_reflection import total_loss
    if (out/'finished.json').exists():
        done=json.loads((out/'finished.json').read_text())
        if done['status']!='complete' or done['best_checkpoint_sha256']!=util.sha_file(out/'best.pt'): raise ValueError('Finished checkpoint changed')
        return
    if not (out/'failure_bank.npz').exists() or not (out/'failure_bank.json').exists(): raise ValueError('Complete mine first')
    bank_meta=json.loads((out/'failure_bank.json').read_text())
    if (bank_meta['status']!='complete' or bank_meta['backbone_sha256']!=protocol['backbone']['sha256']
            or bank_meta['bank_sha256']!=util.sha_file(out/'failure_bank.npz')):
        raise ValueError('Failure bank is not locked to this independent backbone')
    with np.load(out/'failure_bank.npz',allow_pickle=False) as bank:
        if int(bank['split'])!=0: raise ValueError('Failure bank is not split0')
        pool=bank['indices'].copy()
    train_ds=load_sudoku_dataset(args.cache_path,split=0,limit=0)
    val_ds=load_sudoku_dataset(args.cache_path,split=1,limit=0)
    if (pool.min()<0 or pool.max()>=len(train_ds) or not(np.asarray(train_ds.ratings[pool])>4).all()):
        raise ValueError('Invalid failure pool membership')
    val_hard=np.flatnonzero(np.nan_to_num(val_ds.ratings,nan=-np.inf)>4)[:args.val_hard_n]
    val_rng=np.random.default_rng(1729)
    val_all=np.sort(val_rng.choice(len(val_ds),min(args.val_all_n,len(val_ds)),replace=False))
    selection_key=util.sha_array(pool,'<i8')+util.sha_array(val_hard,'<i8')+util.sha_array(val_all,'<i8')
    np.savez_compressed(out/'training_validation_ids.npz',train_failure_positions=pool,val_hard_positions=val_hard,
                        val_all_positions=val_all,train_split=0,validation_split=1)
    model,cfg,_=build_model(SimpleNamespace(checkpoint=args.backbone,hidden=256));model=model.to(device)
    initial_fingerprint=trainable_fingerprint(model)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=1.5e-4,weight_decay=1e-4)
    rng=np.random.default_rng(20000+args.seed)
    loss_args=SimpleNamespace(**COEFFICIENTS)
    step0=0;best_score=-float('inf');best_step=None;best_selection=None;history=[];metrics=None
    resume=out/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['selection_key']!=selection_key or state['protocol']!=protocol: raise ValueError('Resume provenance mismatch')
        model.load_state_dict(state['model_state']); optimizer.load_state_dict(state['optimizer'])
        step0=state['step'];best_score=state['best_score'];best_step=state['best_step'];history=state['history'];metrics=state['metrics']
        best_selection=state['best_selection']
        restore_committed_selection(out,best_selection,best_step,history)
        initial_fingerprint=state['initial_fingerprint'];rng.bit_generator.state=state['sampler_state']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda' and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    started=time.perf_counter()

    def payload(step):
        return dict(model_type='symbolic_primal_dual_reflection',variant=cfg.variant,base_checkpoint=str(Path(args.backbone).resolve()),
                    reflection_cfg=asdict(cfg),model_state=model.state_dict(),step=step,metrics=metrics,
                    args=vars(args),metadata=dict(protocol=protocol,initial_trainable_fingerprint=initial_fingerprint,
                    train_pool_fingerprint=util.sha_array(pool,'<i8'),test_used_for_training_or_selection=False))

    for step in range(step0+1,args.train_steps+1):
        take=rng.choice(pool,args.batch_size,replace=True)
        puzzle=torch.as_tensor(train_ds.puzzles[take],dtype=torch.long,device=device)
        target=torch.as_tensor(train_ds.solutions[take],dtype=torch.long,device=device)
        model.train();model.backbone.eval()
        result=model(puzzle,include_continuation=False)
        loss,pieces=total_loss(model,result,target,loss_args)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite reflection loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%25==0: print(json.dumps(dict(stage='train',seed=args.seed,step=step,loss=float(loss.detach()),
                                                       grad_norm=float(norm),elapsed=time.perf_counter()-started)),flush=True)
        if step%args.eval_every==0 or step==args.train_steps:
            hard,_=evaluate_indices(model,val_ds,val_hard,device,args.eval_batch_size)
            all_metrics,_=evaluate_indices(model,val_ds,val_all,device,args.eval_batch_size)
            metrics=dict(hard=hard,all=all_metrics,score=hard['selected_exact']+all_metrics['selected_exact'])
            history.append(dict(step=step,**metrics));atomic_json(out/'validation_history.json',history)
            if metrics['score']>best_score:
                best_score=metrics['score'];best_step=step;best_selection=versioned_best(out,payload(step),step)
            print(json.dumps(dict(stage='validation',seed=args.seed,step=step,**metrics)),flush=True)
        if step%args.save_every==0 or step==args.train_steps:
            # Save AFTER validation/best selection; sampler and RNG resume at the next update.
            atomic_torch(resume,dict(**payload(step),optimizer=optimizer.state_dict(),protocol=protocol,
                selection_key=selection_key,sampler_state=rng.bit_generator.state,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                best_score=best_score,best_step=best_step,best_selection=best_selection,history=history,initial_fingerprint=initial_fingerprint))
    atomic_torch(out/'final.pt',payload(args.train_steps))
    atomic_json(out/'finished.json',dict(status='complete',seed=args.seed,training_steps=args.train_steps,best_step=best_step,
                best_validation_score=best_score,best_checkpoint_sha256=util.sha_file(out/'best.pt'),
                final_checkpoint_sha256=util.sha_file(out/'final.pt'),reflection_exposure=args.train_steps*args.batch_size,
                backbone=protocol['backbone'],test_used_for_selection=False,smoke=args.smoke,
                latest_invocation_seconds=time.perf_counter()-started))


def evaluate(args,out,protocol,device):
    if args.smoke:raise ValueError('Smoke training cannot run final test')
    import torch
    from eval_symbolic_active_reflection import load_symbolic
    from sudoku_cache_utils import load_sudoku_dataset,load_sudoku_cache
    if not (out/'finished.json').exists():raise ValueError('Complete training before reading test data')
    done=json.loads((out/'finished.json').read_text())
    if done['best_checkpoint_sha256']!=util.sha_file(out/'best.pt'):raise ValueError('Checkpoint changed after validation lock')
    test_out=out/'test';test_out.mkdir(exist_ok=True)
    if (test_out/'summary.json').exists():
        previous=json.loads((test_out/'summary.json').read_text())
        if (previous['status']!='complete' or previous['checkpoint_sha256']!=done['best_checkpoint_sha256']
                or previous['clean_manifest_sha256']!=util.sha_file(args.clean_test_ids)):
            raise ValueError('Existing test result has different provenance')
        return
    cache=load_sudoku_cache(args.cache_path);raw_test=np.flatnonzero(np.asarray(cache['splits'])==2)
    if hasattr(cache,'close'):cache.close()
    allowed=np.load(args.clean_test_ids,allow_pickle=False)
    if allowed.ndim!=1 or len(np.unique(allowed))!=len(allowed) or not np.isin(allowed,raw_test).all():
        raise ValueError('Invalid clean test row manifest')
    indices=np.flatnonzero(np.isin(raw_test,allowed))
    ds=load_sudoku_dataset(args.cache_path,split=2,limit=0)
    model,_,_=load_symbolic(out/'best.pt',device)
    summary,values=evaluate_indices(model,ds,indices,device,args.eval_batch_size,True)
    grids=values['selected_grid'];puzzles=np.asarray(ds.puzzles[indices]);targets=np.asarray(ds.solutions[indices])
    np.testing.assert_array_equal(util.numpy_valid(grids,puzzles),values['first_valid'])
    np.testing.assert_array_equal((grids==targets).all((1,2))&values['first_valid'],values['selected_exact'])
    ratings=np.asarray(ds.ratings[indices]);rows=[]
    for name,mask in [('all',np.ones(len(indices),bool)),('D4',ratings>4)]:
        rows.append(dict(bucket=name,n=int(mask.sum()),exact_count=int(values['selected_exact'][mask].sum()),
                         exact_rate=float(values['selected_exact'][mask].mean()),valid_count=int(values['first_valid'][mask].sum())))
    atomic_npz(test_out/'details.npz',test_position=indices,raw_cache_id=raw_test[indices],rating=ratings,**values)
    atomic_json(test_out/'summary.json',dict(status='complete',seed=args.seed,rows=rows,
                 checkpoint_sha256=done['best_checkpoint_sha256'],clean_manifest_sha256=util.sha_file(args.clean_test_ids),
                 test_id_sha256=util.sha_array(raw_test[indices],'<i8'),smoke=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--stage',choices=['mine','train','evaluate','all'],required=True)
    p.add_argument('--backbone',required=True);p.add_argument('--cache-path',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--clean-test-ids',default=str(HERE/'results/clean_evaluation/test_global_ids.npy'))
    p.add_argument('--device',default='cuda');p.add_argument('--train-steps',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=12);p.add_argument('--eval-batch-size',type=int,default=32)
    p.add_argument('--mine-batch-size',type=int,default=128);p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--save-every',type=int,default=100);p.add_argument('--val-hard-n',type=int,default=4096)
    p.add_argument('--val-all-n',type=int,default=4096);p.add_argument('--bank-limit',type=int,default=0)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if not args.smoke and (args.train_steps!=10000 or args.batch_size!=12 or args.eval_every!=500 or args.bank_limit or args.val_hard_n!=4096 or args.val_all_n!=4096):
        p.error('Formal protocol fixed:10k/batch12/val every500/4096+4096/full train-hard mining; partial runs need --smoke')
    if min(args.train_steps,args.batch_size,args.eval_batch_size,args.mine_batch_size,args.eval_every,args.save_every,args.val_hard_n,args.val_all_n)<1:p.error('Positive sizes required')
    out,protocol,device=setup(args)
    for stage in (['mine','train','evaluate'] if args.stage=='all' else [args.stage]):
        print(json.dumps(dict(stage=stage,seed=args.seed,status='starting')),flush=True)
        {'mine':mine,'train':train,'evaluate':evaluate}[stage](args,out,protocol,device)


if __name__=='__main__':main()
