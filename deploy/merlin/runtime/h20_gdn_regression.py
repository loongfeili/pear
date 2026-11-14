#!/usr/bin/env python3
"""Draft strict GDN regression. --describe is CPU-only; GPU execution requires --run.

The original tests remain unchanged and their failures remain failures. This
harness reports new strict checks separately; it cannot certify full preflight.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types

FIXTURES = {
    1: ('seed1_original_inputs.pt', '311c0d5f55954e5b6b00f6cf1d7b9291dc7dcd0315b7ead32008b562d1a902cc'),
    7: ('seed7_original_inputs.pt', '9455bb18106f2cb0dd67c668da91191abeb3602c2f647d1b338dc674386d5464'),
}

@dataclass(frozen=True)
class Case:
    name: str
    seed: int
    lengths: tuple
    decodes: int
    steps: int
    reset_steps: tuple

# Fixed before any candidate GPU run; no result-dependent case selection.
CASES = tuple(Case(f'boundary_{n}', 1000+n, (1,1,n,n,37), 2, 3, (2,))
              for n in (63,64,65,127,128,129)) + (
    Case('mixed_graph_ceiling_1024', 21024, (1,1,1022), 2, 3, (2,)),
    Case('prefill_1024_carry_16k_then_reset', 31024, (1024,), 0, 17, (16,)),
    Case('mixed_prefill_4096', 24096, (1,1,4096,65,37), 2, 3, (2,)),
    Case('prefill_8192_carry_16k_then_reset', 28192, (8192,), 0, 3, (2,)),
)
BUCKETS = (32,48,64,80,96,112,128,160,176,208,224,240,256,272,288,336,368,384,432,480,
           496,512,592,624,640,656,672,688,720,768,784,800,816,848,864,880,912,928,944,960,976,1008,1024)
PLAN = {
    'version':1, 'per_tp4_rank_dims':{'H':4,'HV':12,'K':128,'V':128,'TP':4},
    'chunk_size':64, 'full_mixed_graph_max_tokens':1024, 'configured_batched_token_budget':8192,
    'cases':[asdict(c) for c in CASES], 'pinned_original_fixtures':FIXTURES,
    'expectations':['g_cum/w/u exact FLA equality','prefill output exact FLA equality',
                    'prefill final state exact FLA equality','graph/eager output and all state bitwise equality',
                    'graph padding zero','empty capture does not change state','state reset and carry'],
    'original_tests':'Execute the unmodified seed1 mixed and seed7 chunk tests separately; retain every failure.',
    'known_baseline':'Seed1 fp64 tolerance failure was shared by FLA chunks + the same gated norm; it remains a failed original assertion.',
    'coverage_limit':'Single-GPU per-TP4-rank GDN numerics; not TP communication or complete model/GRPO certification.',
}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def baseline_evidence(records):
    """Normalize only measurements from a fresh pinned seed1 comparison."""
    result={'baseline_verified':False,'fixture_seed':1,'fixture_sha256':FIXTURES[1][1]}
    if len(records)!=1 or records[0].get('fixture_seed')!=1:return result
    row=records[0];comparisons={r['path']:r for r in row.get('comparisons',[])}
    if set(comparisons)!={'custom','FLA'}:return result
    custom,fla=comparisons['custom'],comparisons['FLA']
    cf,ff=custom.get('failures',[]),fla.get('failures',[])
    coords=[p['index'] for p in cf]
    result.update(shared_gated_bitexact=row.get('same_gated_output_bitexact',False),
                  custom_fp64_bad_count=custom.get('failure_count'),fla_fp64_bad_count=fla.get('failure_count'),
                  coords=coords,custom_actual_values=[p['actual'] for p in cf],
                  fla_actual_values=[p['actual'] for p in ff],reference_values=[p['expected'] for p in cf],
                  rtol=custom.get('rtol'),atol=custom.get('atol'),
                  custom_assertion_passed=custom.get('assertion_passed'),fla_assertion_passed=fla.get('assertion_passed'),
                  all_finite=bool(custom.get('all_finite') and fla.get('all_finite')))
    result['baseline_verified']=bool(
        result['shared_gated_bitexact'] and result['all_finite']
        and all(r.get('rtol')==.05 and r.get('atol')==.05 and r.get('failure_count')==1
                and r.get('assertion_passed') is False for r in (custom,fla))
        and cf==ff and coords==[[77,0,30]]
        and result['custom_actual_values']==[0.9140625]
        and result['reference_values']==[1.1064301385699877])
    return result

class Recorder:
    def __init__(self,path):
        self.stream=path.open('x');self.records=[];self.strict_failures=[];self.baseline_records=[]
    def emit(self,kind,**data):
        item={'kind':kind,**data};self.records.append(item)
        line=json.dumps(item,sort_keys=True);self.stream.write(line+'\n');self.stream.flush()
        print(line,flush=True)
    def exact(self,label,a,b):
        import torch
        equal=bool(torch.equal(a,b))
        finite=bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        passed=equal and finite
        item={'label':label,'shape':list(a.shape),'passed':passed,'bitwise_equal':equal,'all_finite':finite}
        if not passed:
            assert a.shape==b.shape
            af,bf=a.detach().double(),b.detach().double();delta=(af-bf).abs();finite=torch.isfinite(delta)
            indices=torch.nonzero(af!=bf,as_tuple=False)[:3].cpu().tolist()
            item.update(mismatches=int((af!=bf).sum()),actual_nonfinite=int((~torch.isfinite(af)).sum()),
                        expected_nonfinite=int((~torch.isfinite(bf)).sum()),
                        max_abs_finite=float(torch.where(finite,delta,0).max()),
                        mean_abs_finite=float(delta[finite].mean()) if bool(finite.any()) else None,
                        first=[{'index':p,'actual':float(af[tuple(p)]),'expected':float(bf[tuple(p)])} for p in indices])
            self.strict_failures.append(label)
        self.emit('strict_check',**item)
        return passed
    def require(self,label,value):
        passed=bool(value)
        if not passed:self.strict_failures.append(label)
        self.emit('strict_check',label=label,passed=passed)

class GraphRunner:
    def __init__(self,t,layer,T,n,rec,label):
        import numpy as np
        import torch
        self.t,self.layer,self.rec,self.label=t,layer,rec,label
        self.rows=next(x for x in BUCKETS if x>=T)
        self.layout=t.GDNLayout(1024,t.SLOTS,t.SLOTS,t.H,t.HV,t.K,t.V,torch.device('cuda',0))
        self.layout.fill(np.zeros(1,dtype=np.int64),0,np.zeros(0,dtype=np.int32))
        self.slots=torch.zeros(t.SLOTS,dtype=torch.int32,device='cuda')
        self.proj=torch.zeros(self.rows,t.QKVZ+2*t.HV,dtype=torch.bfloat16,device='cuda')
        self.out=torch.zeros(self.rows,t.HV,t.V,dtype=torch.bfloat16,device='cuda')
        before=[x.clone() for x in layer.kv_cache]
        def step():
            t.MixedStepGDNCore.mixed_step(layer,self.layout,self.slots,self.layout.seq_init,
                                         self.proj[:,:t.QKVZ],self.proj[:,t.QKVZ:],self.out)
        step();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):step()
        torch.cuda.synchronize()
        for i,(a,b) in enumerate(zip(layer.kv_cache,before)):
            rec.exact(f'{label}/empty_capture_state{i}',a,b)
    def compare(self,proj,qsl,D,slots,init,start,eager_out,eager_states,step):
        import numpy as np
        import torch
        T=int(qsl[-1]);n=len(slots)
        for target,value in zip(self.layer.kv_cache,start):target.copy_(value)
        self.proj.zero_();self.proj[:T].copy_(proj[:T]);self.out.zero_();self.slots.zero_()
        self.slots[:n].copy_(torch.tensor(slots,dtype=torch.int32,device='cuda'))
        self.layout.fill(qsl,D,np.array(init,dtype=np.int32))
        self.graph.replay();torch.cuda.synchronize()
        prefix=f'{self.label}/step{step}/graph'
        self.rec.exact(prefix+'/output',self.out[:T],eager_out)
        self.rec.require(prefix+'/padding_zero',not bool(self.out[T:].any()))
        for i,(a,b) in enumerate(zip(self.layer.kv_cache,eager_states)):
            self.rec.exact(prefix+f'/state{i}',a,b)
        # Continue the subsequent eager step from its own state even if parity fails.
        for target,value in zip(self.layer.kv_cache,eager_states):target.copy_(value)


def layer_from_fixture(t,data):
    import torch
    return types.SimpleNamespace(prefix='language_model.model.layers.0.linear_attn',
        num_k_heads=t.TP*t.H,num_v_heads=t.TP*t.HV,head_k_dim=t.K,head_v_dim=t.V,tp_size=t.TP,
        kv_cache=(data['conv_state'].to('cuda'),data['ssm_state'].to('cuda')),
        conv1d=types.SimpleNamespace(weight=data['conv_weight'].to('cuda')),
        A_log=data['A_log'].to('cuda'),dt_bias=data['dt_bias'].to('cuda'),
        norm=types.SimpleNamespace(weight=data['norm_weight'].to('cuda'),eps=data['norm_eps']))


def run_case(t,fla,rec,case,fixture=None):
    import numpy as np
    import torch
    gen=torch.Generator(device='cuda').manual_seed(case.seed)
    layer=layer_from_fixture(t,fixture) if fixture else t.gdn_layer(gen)
    lens=list(case.lengths);D=case.decodes;n=len(lens);T=sum(lens)
    assert T<=8192 and n<=t.SLOTS-1 and all(x>1 for x in lens[D:])
    slots=list(fixture['slots']) if fixture else [3,1,2,6,4,5,7][:n]
    qsl=np.concatenate([[0],np.cumsum(lens)]).astype(np.int64)
    _,_,P,NT=t.GDNLayout.needs(qsl,D)
    layout=t.GDNLayout(T,n,P,t.H,t.HV,t.K,t.V,torch.device('cuda',0))
    slot_tensor=torch.tensor(slots,dtype=torch.int32,device='cuda')
    pf=torch.tensor(slots[D:],device='cuda')
    fla_pool=layer.kv_cache[1].clone()
    graph=GraphRunner(t,layer,T,n,rec,case.name) if T<=1024 else None
    rec.emit('case_begin',case=asdict(case),tokens_per_step=T,graph_rows=graph.rows if graph else None)
    for step in range(case.steps):
        if fixture:
            assert step==0
            init=list(fixture['init']);proj=fixture['proj'].to('cuda')[:T]
        else:
            init=[1]*n
            if step==0:
                for i in range(D,n):init[i]=int((i-D)%2==0)
                if n==1:init[0]=0
            if step in case.reset_steps:init[-1]=0
            proj=torch.randn(T,t.QKVZ+2*t.HV,generator=gen,device='cuda').bfloat16()
        before=[x.clone() for x in layer.kv_cache]
        initial=torch.where(torch.tensor(init[D:],device='cuda')[:,None,None,None].bool(),fla_pool[pf],0)
        layout.fill(qsl,D,np.array(init,dtype=np.int32))
        out=torch.zeros(T,t.HV,t.V,dtype=torch.bfloat16,device='cuda')
        t.MixedStepGDNCore.mixed_step(layer,layout,slot_tensor,layout.seq_init,proj[:,:t.QKVZ],proj[:,t.QKVZ:],out)
        torch.cuda.synchronize();eager_states=[x.clone() for x in layer.kv_cache]
        ref=fla(layout.scratch,layout,NT,initial,t.K**-.5);torch.cuda.synchronize()
        prefix=f'{case.name}/step{step}'
        for key in ('g_cum','w','u','o'):
            rec.exact(prefix+'/'+key,getattr(layout.scratch,key)[D:T],ref[key][D:T])
        rec.exact(prefix+'/final',layer.kv_cache[1][pf],ref['final'])
        rec.require(prefix+'/all_output_finite',bool(torch.isfinite(out).all()))
        if fixture and case.seed==1:
            from pear.models.qwen3_5.rollout.hopper.gdn import _gated_norm_kernel
            raw=layout.scratch.o.clone();raw[D:T].copy_(ref['o'][D:T]);fla_out=torch.zeros_like(out)
            _gated_norm_kernel[((T*t.HV+t.MixedStepGDNCore.NORM_ROWS-1)//t.MixedStepGDNCore.NORM_ROWS,)](
                raw,proj[:,t.C:t.QKVZ],layer.norm.weight,fla_out,layout.counts,proj.stride(0),layer.norm.eps,
                HV=t.HV,D=t.V,ROWS=t.MixedStepGDNCore.NORM_ROWS,num_warps=t.MixedStepGDNCore.NORM_WARPS)
            want,_,_=t.reference_core(layer,before,proj,qsl,slots,init)
            torch.cuda.synchronize()
            same=rec.exact(prefix+'/measured_custom_vs_FLA_gated',out,fla_out)
            observed=[]
            for name,value in (('custom',out),('FLA',fla_out)):
                bad=~torch.isclose(value.double(),want,rtol=.05,atol=.05)
                indices=torch.nonzero(bad,as_tuple=False).cpu().tolist()
                record={'path':name,'rtol':.05,'atol':.05,'failure_count':int(bad.sum()),
                        'all_finite':bool(torch.isfinite(value).all() and torch.isfinite(want).all()),
                        'failures':[{'index':idx,'actual':float(value[tuple(idx)]),'expected':float(want[tuple(idx)])} for idx in indices[:8]]}
                try:torch.testing.assert_close(value.double(),want,rtol=.05,atol=.05)
                except AssertionError as exc:record.update(assertion_passed=False,error=str(exc)[:1600])
                else:record['assertion_passed']=True
                observed.append(record)
            shared=(same and all(not r['assertion_passed'] and r['failure_count']==1 and r['all_finite'] for r in observed)
                    and observed[0]['failures']==observed[1]['failures']
                    and observed[0]['failures'][0]['index']==[77,0,30]
                    and observed[0]['failures'][0]['actual']==0.9140625
                    and observed[0]['failures'][0]['expected']==1.1064301385699877)
            baseline={'fixture_seed':1,'measured_shared_failure':shared,'same_gated_output_bitexact':same,'comparisons':observed}
            rec.baseline_records.append(baseline);rec.emit('measured_fp64_shared_baseline',**baseline)
            rec.require(prefix+'/known_shared_fp64_failure_preserved',shared)
        fla_pool[pf]=ref['final']
        if D:fla_pool[slot_tensor[:D].long()]=eager_states[1][slot_tensor[:D].long()]
        if graph:graph.compare(proj,qsl,D,slots,init,before,out,eager_states,step)
        rec.emit('case_step_complete',case=case.name,step=step,init=init,reset=step in case.reset_steps)
    rec.emit('case_complete',case=case.name,steps=case.steps)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--describe',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--repo',type=Path);p.add_argument('--fixtures-dir',type=Path);p.add_argument('--out-dir',type=Path)
    p.add_argument('--backend',choices=('original','candidate'),default='original')
    p.add_argument('--candidate-manifest',type=Path);p.add_argument('--candidate-loader',type=Path)
    args=p.parse_args()
    if not args.run:
        print(json.dumps(PLAN,indent=2));return 0
    assert not args.describe
    assert args.repo and args.fixtures_dir and args.out_dir
    assert os.environ.get('PEAR_GDN_REGRESSION_BOUNDED')=='1','Use the timeout wrapper'
    # A single explicit UUID and an empty compute-process list are required.
    uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
    assert uuid.startswith('GPU-') and ',' not in uuid and len(uuid)>12
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(line.split(',')[0].strip()==uuid for line in apps.splitlines()),'Selected GPU is occupied'
    args.out_dir.mkdir(parents=True,exist_ok=False)
    rec=Recorder(args.out_dir/'checks.jsonl');rec.emit('plan',plan=PLAN,plan_sha256=hashlib.sha256(json.dumps(PLAN,sort_keys=True).encode()).hexdigest())
    report={'strict_regression_pass':False,'original_tests_all_pass':False,'full_preflight_pass':False}
    started=time.monotonic();original=[]
    provenance=None
    try:
        args.repo=args.repo.resolve()
        csrc=args.repo/'pear/models/qwen3_5/rollout/hopper/csrc'
        test_source=args.repo/'tests/models/test_qwen3_5_rollout_kernels_hopper_on_gpu.py'
        repository_sources={p.name:digest(p) for p in sorted(csrc.iterdir()) if p.is_file()}
        assert len(repository_sources)==7
        provenance={'repo':str(args.repo),'backend':args.backend,'source_sha256':repository_sources,
                    'test_source_sha256':digest(test_source),'fixtures_sha256':{},
                    'native_source_dir':str(csrc),'repository_source_sha256':repository_sources}
        os.chdir(args.repo);sys.path.insert(0,str(args.repo))
        import torch
        assert torch.cuda.device_count()==1
        assert torch.cuda.get_device_capability()==(9,0)
        from pear.models.qwen3_5.rollout.hopper.extension import HopperExtension
        if args.backend=='candidate':
            assert args.candidate_manifest and args.candidate_loader
            manifest=json.loads(args.candidate_manifest.read_text())
            assert digest(manifest['library'])==manifest['library_sha256']
            provenance['native_source_dir']=manifest['source_dir']
            provenance['source_sha256']=manifest['source_sha256']
            loader=import_file('pear_h20_candidate_loader',args.candidate_loader)
            assert tuple(HopperExtension.CUDA_FLAGS)==loader.CUDA_FLAGS
            def load(cls):
                if cls.is_loaded:return
                result=loader.load_candidate(manifest['source_dir'],manifest['build_dir'],manifest['name'])
                assert result['library_sha256']==manifest['library_sha256'],'Candidate rebuilt to a different binary'
                assert result['source_sha256']==manifest['source_sha256'],'Candidate sources changed'
                cls.register_fakes();cls.is_loaded=True
                rec.emit('candidate_loaded',provenance=result)
            HopperExtension.load=classmethod(load)
        t=import_file('pear_original_hopper_tests',args.repo/'tests/models/test_qwen3_5_rollout_kernels_hopper_on_gpu.py')
        assert (t.TP,t.H,t.HV,t.K,t.V,t.CHUNK)==(4,4,12,128,128,64)
        assert t.HopperMixedStepGraphs.MAX_TOKENS==1024
        assert tuple(t.HopperMixedStepGraphs.BUCKETS)==BUCKETS
        before_libraries=set(torch.ops.loaded_libraries)
        t.HopperExtension.load()
        native_libraries=[Path(p).resolve() for p in set(torch.ops.loaded_libraries)-before_libraries
                          if Path(p).name.startswith('pear_rollout_hopper') and Path(p).suffix=='.so']
        assert len(native_libraries)==1,native_libraries
        provenance.update(library=str(native_libraries[0]),library_sha256=digest(native_libraries[0]))
        loaded_sources={p.name:digest(p) for p in sorted(Path(provenance['native_source_dir']).iterdir()) if p.is_file()}
        provenance['source_unchanged_after_load']=(loaded_sources==provenance['source_sha256'])
        assert provenance['source_unchanged_after_load'],'Native sources changed while loading the extension'
        source=inspect.getsource(t.fla_chunks)
        assert source.count('    return o, final')==1
        source=source.replace('    return o, final','    return dict(g_cum=g_cum,w=w,u=u,o=o,final=final)')
        ns=dict(t.__dict__);exec(compile(source,'<unmodified_fla_math_with_stage_exports>','exec'),ns);fla=ns['fla_chunks']
        for name in ('test_the_chunk_kernels_equal_flas','test_a_mixed_step_matches_the_reference_eagerly_and_from_a_graph'):
            item={'name':name,'passed':True}
            try:getattr(t,name)()
            except AssertionError as exc:item.update(passed=False,error=str(exc)[:2000])
            original.append(item);rec.emit('unmodified_original_test',**item)
        for seed,(filename,expected) in FIXTURES.items():
            path=args.fixtures_dir/filename;actual=digest(path);assert actual==expected
            provenance['fixtures_sha256'][filename]=actual
            # These files were produced by this task and verified against pinned SHA256.
            data=torch.load(path,map_location='cpu',weights_only=False)
            case=Case(f'pinned_fixture_seed{seed}',seed,tuple(data['lens']),data['decodes'],1,())
            run_case(t,fla,rec,case,data)
        for case in CASES:run_case(t,fla,rec,case)
        report.update(strict_regression_pass=not rec.strict_failures,original_tests_all_pass=all(x['passed'] for x in original))
    except Exception:
        report['error']=traceback.format_exc();rec.emit('execution_error',error=report['error'])
    finally:
        if provenance is not None:
            native_dir=Path(provenance['native_source_dir'])
            native_after={p.name:digest(p) for p in sorted(native_dir.iterdir()) if p.is_file()}
            repository_after={p.name:digest(p) for p in sorted(csrc.iterdir()) if p.is_file()}
            provenance['source_unchanged_after_run']=(native_after==provenance['source_sha256']
                                                     and repository_after==repository_sources)
            provenance['test_source_unchanged_after_run']=(digest(test_source)==provenance['test_source_sha256'])
            provenance['library_unchanged_after_run']=('library' in provenance
                and digest(provenance['library'])==provenance['library_sha256'])
            if not all(provenance[k] for k in ('source_unchanged_after_run','test_source_unchanged_after_run',
                                               'library_unchanged_after_run')):
                rec.strict_failures.append('runtime_provenance/identity_changed_or_incomplete')
                report['strict_regression_pass']=False
            report['runtime_provenance']=provenance
            rec.emit('runtime_provenance',**provenance)
        report.update(backend=args.backend,seconds=time.monotonic()-started,strict_failures=rec.strict_failures,
                      original_tests=original,measured_fp64_baselines=rec.baseline_records,
                      known_fp64_shared_baseline_retained=any(x['measured_shared_failure'] for x in rec.baseline_records),
                      harness_sha256=digest(__file__),plan=PLAN)
        report['baseline_evidence']=baseline_evidence(rec.baseline_records)
        report['baseline_verified']=report['baseline_evidence']['baseline_verified']
        report['known_fp64_shared_baseline_retained']=report['baseline_verified']
        output=args.out_dir/'report.json';temporary=output.with_suffix('.json.partial');temporary.write_text(json.dumps(report,indent=2));temporary.replace(output)
        rec.emit('report',report=report);rec.stream.close()
    return 0 if report['strict_regression_pass'] and report['original_tests_all_pass'] else 1
if __name__=='__main__':raise SystemExit(main())
