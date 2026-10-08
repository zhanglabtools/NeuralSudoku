"""A05 shared timing boundary for target-free exact and neural Sudoku solvers.

Input puzzles are resident in host RAM; model/CNF construction and warmup are
outside steady-state latency. Per-query input transfer, solve, prediction transfer
and clue-aware independent verification are timed. Reference Exact is scored later.
CPU measurements use explicit affinity/threads on a monitored shared server.
GPU-exclusive measurements are rejected if another process uses the selected GPU.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,random,subprocess,sys,threading,time
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,default=str),encoding='utf-8'); tmp.replace(path)


def gpu_processes(physical):
    uuid=subprocess.run(['nvidia-smi',f'--id={physical}','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    if uuid.returncode: raise RuntimeError('Cannot identify timing GPU')
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    if result.returncode: raise RuntimeError('Cannot inspect timing GPU processes')
    return [int(fields[0].strip()) for line in result.stdout.splitlines() if len(fields:=line.split(','))==2 and fields[1].strip()==uuid.stdout.strip()]


class Monitor:
    def __init__(self,mode,physical):
        self.mode=mode; self.physical=physical; self.stop=threading.Event(); self.samples=[]; self.errors=[]
    def sample(self):
        import psutil
        proc=psutil.Process()
        value={'monotonic':time.monotonic(),'rss':proc.memory_info().rss,'loadavg':os.getloadavg(),'cpu_times':tuple(proc.cpu_times()[:2]),'context_switches':tuple(proc.num_ctx_switches())}
        if self.mode=='gpu-exclusive': value['other_gpu_pids']=[p for p in gpu_processes(self.physical) if p!=os.getpid()]
        self.samples.append(value)
    def loop(self):
        while not self.stop.wait(.5):
            try:self.sample()
            except Exception as exc:self.errors.append(repr(exc))
    def __enter__(self):
        self.sample(); self.thread=threading.Thread(target=self.loop,daemon=True); self.thread.start(); return self
    def __exit__(self,*_):
        self.stop.set(); self.thread.join(); self.sample()
    def summary(self):
        return {'sampling_seconds':.5,'peak_process_rss_bytes':max(s['rss'] for s in self.samples),'other_gpu_process_observed':any(s.get('other_gpu_pids') for s in self.samples),'monitor_errors':self.errors,'samples':self.samples,'limitation':'Sampling may miss intervals shorter than 0.5s; CPU affinity is not an exclusive core reservation; RSS includes model, inputs, runtime and sampling overhead.'}


def c5_predict(model,puzzle):
    """Original active solver ordering, returning grids without access to targets."""
    import torch
    from eval_symbolic_active_reflection import _keep_slots
    from eval_d4_reflection_ablation import torch_candidate_flags
    batch=len(puzzle); slots=int(model.cfg.slots)
    x0,u0,h,u,logits=model.encode_parent(puzzle)
    pred=logits.argmax(-1)+1
    valid,_=torch_candidate_flags(pred,puzzle)
    chosen=torch.zeros_like(puzzle); chosen[valid]=pred[valid]
    cycles=torch.full((batch,),-1,dtype=torch.int16,device=puzzle.device); cycles[valid]=0
    active=torch.nonzero(~valid,as_tuple=False).squeeze(-1)
    if not len(active):return chosen,cycles
    state={k:model._expand_slots(v.index_select(0,active),slots) for k,v in [('h',h),('u',u),('x0',x0),('u0',u0),('puzzle',puzzle)]}
    state['dual']=h.new_zeros(len(active)*slots,27,9)
    for cycle in range(int(model.cfg.cycles)):
        n=len(active)
        state['h'],state['u'],state['dual'],_=model._symbolic_reflect_once(state['h'],state['u'],state['x0'],state['u0'],state['puzzle'],state['dual'],source_batch=n,slots=slots,cycle_index=cycle)
        state['h'],state['u']=model._rollout(state['h'],state['u'],state['x0'],state['u0'],model.cfg.recovery_steps)
        pred=model.backbone.logits_from_state(state['h'],state['puzzle']).argmax(-1)+1
        valid,_=torch_candidate_flags(pred,state['puzzle']); valid=valid.reshape(n,slots)
        solved=valid.any(1)
        first=valid.to(torch.int64).argmax(1)
        picked=pred.reshape(n,slots,9,9)[torch.arange(n,device=puzzle.device),first]
        chosen[active[solved]]=picked[solved]; cycles[active[solved]]=cycle+1
        keep=~solved; active=active[keep]
        if not len(active):break
        state={k:_keep_slots(v,keep,n,slots) for k,v in state.items()}
    return chosen,cycles


def quantiles(values):
    values=np.asarray(values,dtype=float)
    return {'n':len(values),'mean_seconds':float(values.mean()),'p50_seconds':float(np.quantile(values,.5)),'p95_seconds':float(np.quantile(values,.95))}


def run(args):
    from e2_metrics import board_metrics
    from eval_d4_reflection_ablation import numpy_valid
    import psutil
    import torch
    affinity=[int(x) for x in args.cpu_affinity.split(',') if x]
    if args.mode=='cpu-controlled' and not affinity:raise ValueError('CPU timing requires explicit affinity')
    if affinity:os.sched_setaffinity(0,set(affinity))
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if args.mode=='gpu-exclusive':
        if args.device!='cuda' or args.solver in ('dlx','sat'):raise ValueError('GPU-exclusive mode is for CUDA neural models')
        if any(pid!=os.getpid() for pid in gpu_processes(args.physical_gpu)):raise RuntimeError('Selected GPU already has another compute process')
    if args.mode=='cpu-controlled' and args.device!='cpu':raise ValueError('CPU timing cannot run CUDA')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    with np.load(args.sample,allow_pickle=False) as data:
        puzzles=data['puzzles']; targets=data['solutions']; ids=data['source_row']; ratings=data['ratings']
    if args.limit:
        if args.mode!='pilot':raise ValueError('Only pilot permits limit')
        puzzles,targets,ids,ratings=(x[:args.limit] for x in (puzzles,targets,ids,ratings))
    n=len(ids); device=torch.device(args.device); model=None; modelmeta={}; external_predictor=None
    if args.solver in ('rrn','hybrid'):
        from e2_eval_checkpoints import load_model
        model,modelmeta=load_model(args.checkpoint,device)
    elif args.solver=='c5':
        from eval_symbolic_active_reflection import load_symbolic
        from dataclasses import asdict
        model,cfg,checkpoint=load_symbolic(args.checkpoint,device)
        if not model.backbone.cfg.force_clues:raise ValueError('C5 must preserve clues')
        modelmeta={'cfg':asdict(cfg),'checkpoint_sha256':sha(args.checkpoint),'base_checkpoint_sha256':sha(checkpoint['base_checkpoint']),'parameters':sum(p.numel() for p in model.parameters())}
    elif args.solver in ('satnet','abl_raw','abl_repair','hrm'):
        from a08_train_baseline import load_predictor
        external_predictor=load_predictor(args.solver,args.checkpoint,device)
        modelmeta={'checkpoint_sha256':sha(args.checkpoint),'adapter':'a08_train_baseline.load_predictor','method':args.solver}
    else:
        import benchmark_exact_solvers as exact
        if args.solver=='sat':modelmeta=exact.load_sat(args.pysat_path)
        modelmeta['dlx_source_sha256']=sha(exact.dlx.__file__)
    if model is not None:model=model.float().eval()
    def sync():
        if device.type=='cuda':torch.cuda.synchronize()
    def solve(batch):
        if external_predictor is not None:return np.asarray(external_predictor(batch),dtype=np.uint8)
        if model is None:
            predictions=[]
            for puzzle in batch:
                result=exact.solve_task((0,puzzle.reshape(-1).tolist(),args.solver,args.timeout_seconds,0,0))
                if result['status'] in ('error','invalid_solver_output'):raise RuntimeError(result)
                predictions.append(np.zeros((9,9),dtype=np.uint8) if not result['valid'] else np.array(list(result['prediction']),dtype=np.uint8).reshape(9,9))
            return np.array(predictions)
        inp=torch.as_tensor(batch,dtype=torch.long,device=device)
        if args.solver=='c5':pred,_=c5_predict(model,inp)
        else:
            from e2_eval_checkpoints import forward_logits
            logits=forward_logits(model,inp,args.steps,readout_protocol='common_final'); pred=logits.argmax(-1)+1
        return pred.cpu().numpy().astype(np.uint8)
    protocol={'sample_sha256':sha(args.sample),'source_rows_sha256':hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),'n':n,'solver':args.solver,'model':modelmeta,'args':vars(args),'source_sha256':{str(Path(__file__).resolve()):sha(__file__)},'timing_boundary':'host-resident input -> solver computation including candidate checks and data transfers -> host prediction -> independent clue/rule validation; Exact scoring, disk IO, model load and warmup excluded','exact_verification_note':'Exact solvers also verify internally in solve_task; shared external checker is included for all methods','cpu_affinity':sorted(os.sched_getaffinity(0)),'cpu_model':platform.processor(),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'tf32':False,'host':platform.node(),'loadavg':os.getloadavg(),'status':'running'}
    # Include relevant project modules loaded by the solver, not only this wrapper.
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).is_absolute() and Path(p).is_file() and Path(p).suffix=='.py' and Path(p).resolve().is_relative_to(Path(__file__).resolve().parents[2]):
            protocol['source_sha256'][str(Path(p).resolve())]=sha(p)
    fingerprint=hashlib.sha256(json.dumps({k:v for k,v in protocol.items() if k not in ('loadavg','status')},sort_keys=True,default=str).encode()).hexdigest()
    protocol['fingerprint']=fingerprint
    protocol['repeat_rng_rule']='Python/NumPy/PyTorch seed = base seed + repeat index, set outside timing after warmup; repetitions resume independently.'
    protocol['rrn_hybrid_readout']='common_final: recurrent updates unchanged, omit unused intermediate classifiers, read out once at the final state'
    if (out/'metadata.json').exists():
        old=json.loads((out/'metadata.json').read_text())
        if old['fingerprint']!=fingerprint:raise ValueError('Timing resume fingerprint changed')
    save(out/'metadata.json',protocol)
    with torch.inference_mode():
        for _ in range(args.warmup):solve(puzzles[:args.batch_size])
        sync()
        summaries=[]
        for repeat in range(args.repeats):
            dest=out/f'repeat{repeat}.json'
            if dest.exists():
                previous=json.loads(dest.read_text())
                if sha(out/f'repeat{repeat}.npz')!=previous['result_sha256']:raise ValueError('Saved repeat data hash mismatch')
                summaries.append(previous); continue
            random.seed(args.seed+repeat);np.random.seed(args.seed+repeat);torch.manual_seed(args.seed+repeat)
            if device.type=='cuda':torch.cuda.manual_seed_all(args.seed+repeat)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            predictions=np.zeros_like(puzzles,dtype=np.uint8); durations=[]; batch_sizes=[]
            with Monitor(args.mode,args.physical_gpu) as monitor:
                started=time.perf_counter()
                for begin in range(0,n,args.batch_size):
                    end=min(n,begin+args.batch_size); sync(); before=time.perf_counter()
                    pred=solve(puzzles[begin:end]); valid=numpy_valid(pred,puzzles[begin:end]); sync()
                    elapsed=time.perf_counter()-before
                    predictions[begin:end]=pred; durations.append(elapsed); batch_sizes.append(end-begin)
                    if end%256==0 or end==n:print(json.dumps({'solver':args.solver,'device':str(device),'repeat':repeat,'n':end,'total':n}),flush=True)
                wall=time.perf_counter()-started
            telemetry=monitor.summary()
            eligible=not telemetry['monitor_errors'] and not telemetry['other_gpu_process_observed'] and args.mode!='pilot'
            metrics=board_metrics(predictions,puzzles,targets)
            np.savez_compressed(out/f'repeat{repeat}.npz',source_row=ids,ratings=ratings,predictions=predictions,**metrics,batch_pipeline_seconds=np.array(durations),batch_sizes=np.array(batch_sizes))
            result={'solver':args.solver,'repeat':repeat,'n':n,'device':str(device),'batch_size':args.batch_size,'batch_latency':quantiles(durations),'single_query_latency':quantiles(durations) if args.batch_size==1 else None,'pipeline_seconds':sum(durations),'loop_wall_seconds':wall,'throughput_puzzles_per_second':n/sum(durations),'valid':int(metrics['valid'].sum()),'exact':int(metrics['exact'].sum()),'valid_not_exact':int((metrics['valid']&~metrics['exact']).sum()),'D4_n':int((ratings>4).sum()),'D4_valid':int(metrics['valid'][ratings>4].sum()),'eligible':eligible,'timing_context':args.mode,'memory_scope':'RSS includes runtime/model/host inputs; CUDA peak_allocated includes live model and activations','peak_process_rss_bytes':telemetry['peak_process_rss_bytes'],'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else None,'result_sha256':sha(out/f'repeat{repeat}.npz')}
            save(out/f'repeat{repeat}_telemetry.json',telemetry); save(dest,result); summaries.append(result)
            save(out/'summary.json',{'status':'running','runs':summaries,'fingerprint':fingerprint})
            if args.mode=='gpu-exclusive' and not eligible:raise RuntimeError('Selected GPU timing was contaminated; artifacts retained, cannot report as exclusive')
    save(out/'summary.json',{'status':'complete','runs':summaries,'fingerprint':fingerprint})
    protocol['status']='complete'; save(out/'metadata.json',protocol)
    print(json.dumps({'output':str(out),'status':'complete','n':n,'solver':args.solver,'repeats':len(summaries)}),flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--solver',choices=['dlx','sat','rrn','hybrid','c5','satnet','abl_raw','abl_repair','hrm'],required=True)
    p.add_argument('--checkpoint',default='');p.add_argument('--steps',type=int,default=64)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--physical-gpu',type=int,default=0)
    p.add_argument('--mode',choices=['pilot','cpu-controlled','gpu-exclusive'],default='cpu-controlled')
    p.add_argument('--cpu-affinity',default='');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--seed',type=int,default=20260930);p.add_argument('--timeout-seconds',type=float,default=5)
    p.add_argument('--pysat-path',default='code/optional_dependencies/python_sat')
    args=p.parse_args()
    if min(args.threads,args.batch_size,args.repeats,args.steps)<1:raise ValueError('Positive counts required')
    return args


if __name__=='__main__':run(parse_args())
