"""A04 scope extension: frozen D4-validation settings applied to clean all-test.

The original validation lock is read and verified without alteration. No new
sigma/schedule selection is performed. D4 is also reported as a subset of this
all-test batching, so it need not exactly match the separate D4-only run.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
import eval_c5_random_budget as core
import eval_d4_reflection_ablation as util


def locked_configurations(lock):
    if lock.get('selection')!='d4' or lock.get('smoke_only'):
        raise ValueError('Clean all-test requires a complete formal D4-validation lock')
    plan=lock['plan'];core.validate_plan(plan)
    if len(plan['schedules'])!=5 or plan['random_repeats']!=3:
        raise ValueError('Require the preregistered five schedules and three random repeats')
    result=[]
    for schedule in plan['schedules']:
        result.append(dict(schedule=schedule,method='c5',sigma=0.,repeat=0,label=schedule['name']+'_c5'))
        sigma=lock['selected_sigma_by_schedule'][schedule['name']]
        if sigma not in plan['sigmas']:raise ValueError('Selected sigma is outside the validation grid')
        for repeat in range(plan['random_repeats']):
            result.append(dict(schedule=schedule,method='random',sigma=float(sigma),repeat=repeat,
                               label=f"{schedule['name']}_random_s{sigma}_r{repeat}"))
    return result


def clean_positions(global_ids,raw_test_ids,expected_count):
    values=np.asarray(global_ids)
    if (values.ndim!=1 or values.dtype.kind not in 'iu' or len(values)!=expected_count
            or len(np.unique(values))!=len(values) or not np.isin(values,raw_test_ids).all()):
        raise ValueError('Clean manifest must contain the expected number of unique split2 global IDs')
    # Preserve the clean manifest's order; no outcome or rating-based selection.
    return np.searchsorted(raw_test_ids,values).astype(np.int64)


def discrete_paired_ci(numerators,denominator,repeats,seed):
    """Exact empirical paired-mean bootstrap via multinomial category counts.

Each puzzle's difference is integer/3 for three random repeats. Resampling the
at-most-seven categories is distributionally identical to resampling N paired
puzzles, avoiding N*20,000 arrays for the complete test set.
"""
    values=np.asarray(numerators)
    if values.ndim!=1 or not len(values) or denominator<1 or repeats<1:raise ValueError('Invalid paired sample')
    if not np.equal(values,np.rint(values)).all():raise ValueError('Expected integer paired numerators')
    support,counts=np.unique(values.astype(np.int64),return_counts=True)
    if len(support)==1:return np.array([support[0]/denominator]*2,dtype=float)
    rng=np.random.default_rng(seed)
    draws=rng.multinomial(len(values),counts/counts.sum(),size=repeats)
    means=(draws@support)/(len(values)*denominator)
    return np.quantile(means,[.025,.975])


def paired_rows(results,configurations,ratings,bootstrap_repeats,seed):
    rows=[];schedules={c['schedule']['name']:c['schedule'] for c in configurations}
    for group,mask in [('all',np.ones(len(ratings),bool)),('D4',np.asarray(ratings)>4)]:
        if not mask.any():continue
        for index,name in enumerate(schedules):
            c5=results[name+'_c5']
            randoms=[c for c in configurations if c['method']=='random' and c['schedule']['name']==name]
            for metric_index,metric in enumerate(('selected_exact','first_valid')):
                matrix=np.stack([results[c['label']][metric][mask] for c in randoms]).astype(np.int64)
                success=np.asarray(c5[metric][mask],dtype=np.int64)
                numerators=len(matrix)*success-matrix.sum(axis=0)
                ci_seed=seed+1009*index+metric_index+(100000 if group=='D4' else 0)
                low,high=discrete_paired_ci(numerators,len(matrix),bootstrap_repeats,ci_seed)
                rows.append(dict(bucket=group,schedule=name,metric=metric,n=len(success),sigma=randoms[0]['sigma'],
                    c5_rate=float(success.mean()),random_repeat_mean=float(matrix.mean()),
                    random_repeat_rate_std=float(matrix.mean(axis=1).std(ddof=1)),
                    c5_minus_random_pp=float(numerators.mean()/len(matrix)*100),ci_low_pp=float(low*100),ci_high_pp=float(high*100),
                    bootstrap_repeats=bootstrap_repeats,bootstrap_seed=ci_seed,
                    ci_scope='paired puzzle percentile bootstrap, conditional on these three inference-noise repeats; not training-seed uncertainty',
                    bootstrap_algorithm='multinomial frequencies of paired difference categories; exact empirical-bootstrap distribution'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--cache-path',required=True)
    p.add_argument('--clean-test-ids',required=True);p.add_argument('--lock-config',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--expected-count',type=int,default=300702)
    p.add_argument('--device',default='cuda');p.add_argument('--bootstrap-repeats',type=int,default=20000)
    args=p.parse_args()
    if args.expected_count<1 or args.bootstrap_repeats<1:p.error('Positive count and bootstrap repeats required')
    output=Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Preserve prior output; use a new directory for a retry')
    lock,lock_sha=core.verify_validation_lock(args.lock_config)
    configurations=locked_configurations(lock)
    checkpoint_sha=util.sha_file(args.checkpoint)
    if checkpoint_sha!=lock['checkpoint_sha256']:raise ValueError('Checkpoint differs from formal D4 validation')
    sources=util.source_manifest()
    sources[str(Path(core.__file__).resolve())]=util.sha_file(core.__file__)
    sources[str(Path(util.__file__).resolve())]=util.sha_file(util.__file__)
    if sources!=lock['source_sha256']:raise ValueError('Frozen core code differs from D4 validation')
    validation_meta=json.loads((Path(args.lock_config).parent/'metadata.json').read_text())
    if str(Path(args.cache_path).resolve())!=str(Path(validation_meta['args']['cache_path']).resolve()):
        raise ValueError('Evaluation cache path must match validation cache')
    import torch
    from eval_symbolic_active_reflection import load_symbolic
    from sudoku_cache_utils import load_sudoku_dataset,load_sudoku_cache
    from sudoku_exchange_experiment import set_seed,set_torch_threads
    set_torch_threads();set_seed(lock['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('highest')
    device=torch.device(args.device);model,cfg,checkpoint=load_symbolic(args.checkpoint,device)
    model=model.float().eval();plan=lock['plan'];batch_size=lock['batch_size']
    if (cfg.parent_steps!=plan['parent_steps'] or cfg.mode!='global8' or not model.backbone.cfg.force_clues
            or any(s['slots']>cfg.slots or s['cycles']>cfg.cycles for s in plan['schedules'])):
        raise ValueError('Checkpoint does not support locked inference settings')
    base_sha=util.sha_file(checkpoint['base_checkpoint'])
    if base_sha!=validation_meta['base_checkpoint_sha256']:raise ValueError('Base checkpoint changed after validation')
    # No test IDs or targets are opened until all validation/code/model gates pass.
    raw_ids=np.load(args.clean_test_ids,allow_pickle=False)
    cache=load_sudoku_cache(args.cache_path);raw_test=np.flatnonzero(np.asarray(cache['splits'])==2)
    if hasattr(cache,'close'):cache.close()
    positions=clean_positions(raw_ids,raw_test,args.expected_count)
    dataset=load_sudoku_dataset(args.cache_path,split=2,limit=0)
    puzzles=np.asarray(dataset.puzzles[positions],dtype=np.uint8);targets=np.asarray(dataset.solutions[positions],dtype=np.uint8)
    ratings=np.asarray(dataset.ratings[positions],dtype=np.float32)
    if not util.numpy_valid(targets,puzzles).all():raise ValueError('Test reference violates Sudoku or clues')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'evaluation_ids.npz',raw_cache_id=raw_ids,split_position=positions,rating=ratings)
    metadata=dict(status='running',phase='test',evaluation_scope='clean all test; D4 is a subset of the same full-test batches',
        sigma_selection_scope='D4 validation only; no all-test retuning',validation_selection_unchanged=lock['selection'],
        validation_lock_sha256=lock_sha,validation_metadata_sha256=util.sha_file(Path(args.lock_config).parent/'metadata.json'),
        checkpoint_sha256=checkpoint_sha,base_checkpoint_sha256=base_sha,core_source_sha256=sources,
        wrapper_source_sha256={str(Path(__file__).resolve()):util.sha_file(__file__)},args=vars(args),cfg=asdict(cfg),plan=plan,
        selected_sigma_by_schedule=lock['selected_sigma_by_schedule'],seed=lock['seed'],batch_size=batch_size,
        clean_manifest_sha256=util.sha_file(args.clean_test_ids),data=dict(split=2,n=len(raw_ids),D4_n=int((ratings>4).sum()),
        raw_ids_sha256=util.sha_array(raw_ids,'<i8'),puzzle_sha256=util.sha_array(puzzles,'u1'),target_sha256=util.sha_array(targets,'u1')),
        precision='FP32; AMP off; TF32 off',device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,
        outcome_policy='anchor then increasing cycle then first valid slot; valid includes original clues; unresolved output zero and Exact false',
        grid_storage='Predicted grids are independently verified per batch; per-puzzle outcomes and actual work are persisted without full grids.',
        interpretation='Matched propagation slot-step and candidate caps are not equal FLOPs or realized active work.',
        D4_comparison_note='D4 subset here uses all-test batch membership; compare to separate D4-only run with this numerical batching distinction disclosed.')
    util.save_json(output/'metadata.json',metadata)
    collected={c['label']:{} for c in configurations};elapsed={c['label']:0. for c in configurations}
    calls={c['label']:dict(reflection_function_calls=0,recovery_step_function_calls=0) for c in configurations}
    parent_seconds=0.;started=time.perf_counter()
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    try:
        for start in range(0,len(raw_ids),batch_size):
            end=min(start+batch_size,len(raw_ids))
            puzzle=torch.as_tensor(puzzles[start:end],device=device,dtype=torch.long)
            target=torch.as_tensor(targets[start:end],device=device,dtype=torch.long)
            with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
                sync();begin=time.perf_counter();parent=model.encode_parent(puzzle);sync();parent_seconds+=time.perf_counter()-begin
                for condition in configurations:
                    label=condition['label'];begin=time.perf_counter()
                    values=core.evaluate_batch(model,puzzle,target,parent,raw_ids[start:end],condition['schedule'],condition['method'],
                        condition['sigma'],condition['repeat'],lock['seed'],2)
                    sync();elapsed[label]+=time.perf_counter()-begin
                    calls[label]['reflection_function_calls']+=int(values['reflection_events'].max())
                    active_cycles=int(torch.maximum(values['reflection_events'],values['perturbation_events']).max())
                    calls[label]['recovery_step_function_calls']+=active_cycles*condition['schedule']['recovery_steps']
                    arrays={key:value.cpu().numpy() for key,value in values.items()}
                    grid=arrays.pop('selected_grid');valid=util.numpy_valid(grid,puzzles[start:end])
                    np.testing.assert_array_equal(valid,arrays['first_valid'])
                    np.testing.assert_array_equal((grid==targets[start:end]).all((1,2))&valid,arrays['selected_exact'])
                    if grid[~valid].any():raise AssertionError('Unsolved outputs are not zero')
                    for key,value in arrays.items():collected[label].setdefault(key,[]).append(value)
            if end%4096<end-start or end==len(raw_ids):
                print(json.dumps(dict(stage='A04_clean_all_test',seen=end,total=len(raw_ids),configurations=len(configurations),elapsed_seconds=time.perf_counter()-started)),flush=True)
        results={label:{key:np.concatenate(parts) for key,parts in values.items()} for label,values in collected.items()}
        rows=[]
        for condition in configurations:
            label=condition['label'];values=results[label]
            np.savez_compressed(output/(label+'_outcomes.npz'),raw_cache_id=raw_ids,rating=ratings,**values)
            for group,mask in [('all',np.ones(len(raw_ids),bool)),('D4',ratings>4)]:
                if not mask.any():continue
                valid=values['first_valid'][mask];exact=values['selected_exact'][mask]
                row=dict(bucket=group,label=label,schedule=condition['schedule']['name'],method=condition['method'],sigma=condition['sigma'],repeat=condition['repeat'],
                    n=int(mask.sum()),selected_exact_count=int(exact.sum()),selected_exact_rate=float(exact.mean()),first_valid_count=int(valid.sum()),
                    first_valid_rate=float(valid.mean()),valid_not_exact_count=int((valid&~exact).sum()),unsolved_count=int((~valid).sum()))
                for key in ('propagation_slot_steps','candidates_generated','reflection_events','perturbation_events'):row['actual_'+key]=int(values[key][mask].sum())
                row['actual_reflection_slot_events']=row['actual_reflection_events']*condition['schedule']['slots'];rows.append(row)
        paired=paired_rows(results,configurations,ratings,args.bootstrap_repeats,lock['seed'])
        util.save_csv(output/'condition_summary.csv',rows);util.save_csv(output/'paired_budget_comparison.csv',paired)
        metadata.update(status='complete',parent_seconds=parent_seconds,condition_seconds=elapsed,function_calls=calls,
            shared_parent_forward_calls=(len(raw_ids)+batch_size-1)//batch_size,elapsed_seconds=time.perf_counter()-started)
        util.save_json(output/'metadata.json',metadata)
        util.save_json(output/'summary.json',dict(status='complete',evaluation_scope=metadata['evaluation_scope'],sigma_selection_scope=metadata['sigma_selection_scope'],
            n=len(raw_ids),D4_n=int((ratings>4).sum()),rows=rows,paired_comparisons=paired,metadata_sha256=util.sha_file(output/'metadata.json')))
    except Exception as exc:
        metadata.update(status='failed',error=f'{type(exc).__name__}: {exc}');util.save_json(output/'metadata.json',metadata);raise


if __name__=='__main__':main()
