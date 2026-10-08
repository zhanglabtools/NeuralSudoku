"""Reproduce Figure 4's paired, outcome-stratified bootstrap from original CSVs.

Example:
  python ablation_192/recompute_stratified_bootstrap.py --data-dir ablation_192 --output-dir recomputed/ablation_192

Requires Python 3 and NumPy. Reads but never modifies the original data.
The 64 validation and 128 test records are pooled within each of four outcome
strata, giving 48 records in each stratum. Each bootstrap sample preserves
these four stratum sizes and the pairing between full and ablated conditions.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20260801)
    parser.add_argument('--replicates',type=int,default=20000)
    args=parser.parse_args()
    records=[]
    for split in ('val','test'):
        p=args.data_dir/split/'c5_ablation_summary.csv'
        with p.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                r['split']=split; records.append(r)
    keys=sorted({(r['split'],r['split_position']) for r in records})
    assert len(keys)==192
    positions={key:i for i,key in enumerate(keys)}
    successes={}; group_by_position={}; seen=set()
    for r in records:
        name=r['ablation']; i=positions[(r['split'],r['split_position'])]
        assert (name,i) not in seen
        seen.add((name,i))
        successes.setdefault(name,np.zeros(192,dtype=np.int8))[i]=float(r['first_valid_cycle'])>=0
        group_by_position[i]=r['group']
    group_names=sorted(set(group_by_position.values()))
    assert len(group_names)==4
    groups=[np.array([i for i in range(192) if group_by_position[i]==name]) for name in group_names]
    assert all(len(indices)==48 for indices in groups)
    rng=np.random.default_rng(args.seed)
    sampled_indices=np.concatenate([rng.choice(indices,size=(args.replicates,len(indices)),replace=True) for indices in groups],axis=1)
    rows=[]
    for name in ('full','no_recovery','recovery_only','no_cell'):
        a=successes[name]; full=successes['full']; differences=a-full
        lower,upper=np.quantile(100*differences[sampled_indices].mean(axis=1),[.025,.975],method='linear')
        counts={split:int(a[[i for i,key in enumerate(keys) if key[0]==split]].sum()) for split in ('val','test')}
        delta=100*float(differences.mean())
        rows.append({'condition':name,'val_success':counts['val'],'test_success':counts['test'],'total_success':int(a.sum()),'n':192,'success_rate_percent':100*float(a.mean()),'delta_pp':delta,'delta_pp_2dp':f'{delta:.2f}','lost':int(((full==1)&(a==0)).sum()),'gained':int(((full==0)&(a==1)).sum()),'ci95_lower_pp':float(lower),'ci95_upper_pp':float(upper),'ci95_2dp':[f'{lower:.2f}',f'{upper:.2f}']})
    result={'success_definition':'first_valid_cycle >= 0: at least one legal candidate generated within the budget','difference_direction':'ablated minus full, in percentage points','grouping':'Four outcome strata; validation and test pooled within each outcome stratum','group_order':group_names,'group_sizes':[48]*4,'sample_identity_order':'lexicographic (split, split_position as CSV text)','seed':args.seed,'replicates':args.replicates,'sampling':'Within-stratum uniform paired sampling with replacement; identical sampled indices for all conditions','quantiles':[.025,.975],'quantile_method':'numpy.quantile linear','numpy_version':np.__version__,'display_rounding':'Python fixed-point formatting to 2 decimals; exact -40.625 prints -40.62','rows':rows}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'stratified_bootstrap_192.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.output_dir/'stratified_bootstrap_192.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'rows':rows,'seed':args.seed,'replicates':args.replicates},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
