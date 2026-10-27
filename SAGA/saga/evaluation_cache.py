"""Atomic per-candidate results let interrupted generations replay without new relaxations."""
from pathlib import Path
import hashlib,json,os
import numpy as np

def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)

class EvaluationCache:
    def __init__(self,inner,config,path):
        self.inner=inner;self._config=config;self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.captured={};self.hits=0;self.misses=0
        if hasattr(inner,'gas_energies'):
            refs=self.path.parent/'gas_references.json'
            current=dict(gas_energies=inner.gas_energies,references=config.references)
            if refs.exists():
                stored=json.loads(refs.read_text())
                if set(stored['gas_energies'])!=set(inner.gas_energies) or any(not np.isclose(inner.gas_energies[g],value,rtol=0,atol=1e-7) for g,value in stored['gas_energies'].items()):
                    raise ValueError('recomputed gas references differ from the frozen run')
                inner.gas_energies=stored['gas_energies']
                inner._refs={a:sum(c*inner.gas_energies[g] for g,c in terms) for a,terms in config.references.items()}
            else:atomic_json(refs,current)
    @property
    def config(self):return self._config
    @config.setter
    def config(self,cfg):
        self._config=cfg
        if hasattr(self.inner,'config'):self.inner.config=cfg
        if hasattr(self.inner,'cfg'):self.inner.cfg=cfg
    @property
    def stats(self):return dict(getattr(self.inner,'stats',{}),cache_hits=self.hits,cache_misses=self.misses)
    def _capture_write(self,path,cap):
        from ase.io import write
        frames=[]
        for i,atoms in enumerate(cap.get('slabs',[])):
            a=atoms.copy();a.calc=None;a.info.update(cache_kind='slab',cache_index=i);frames.append(a)
        for species,atoms,energy,index in cap.get('probes',[]):
            a=atoms.copy();a.calc=None;a.info.update(cache_kind='probe',cache_species=species,cache_energy=float(energy),cache_index=int(index));frames.append(a)
        for atoms in cap.get('rejected',[]):
            a=atoms.copy();a.calc=None;a.info.update(cache_kind='rejected');frames.append(a)
        if frames:
            temp=path.with_suffix('.tmp');write(temp,frames,format='extxyz');temp.replace(path)
        return bool(frames)
    def _capture_read(self,path,data):
        from ase.io import iread
        cap=dict(data,slabs=[],probes=[],rejected=[])
        for a in iread(path,format='extxyz'):
            kind=a.info.pop('cache_kind')
            if kind=='slab':cap['slabs'].append(a)
            elif kind=='probe':cap['probes'].append((a.info['cache_species'],a,float(a.info['cache_energy']),int(a.info['cache_index'])))
            else:cap['rejected'].append(a)
        return cap
    def evaluate_many(self,cands,rng,label='',force=None):
        from .genome import candidate_dict
        self.captured={};out=[]
        seeds=rng.integers(0,2**62,size=len(cands))
        for cand,seed in zip(cands,seeds):
            identity=dict(candidate=candidate_dict(cand),seed=int(seed),label=label,n_configs=self.config.n_configs,n_probes=self.config.n_probes)
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();path=self.path/(key+'.json');geometry=self.path/(key+'.extxyz')
            if path.exists():
                record=json.loads(path.read_text())
                if record['identity']!=identity:raise ValueError('evaluation cache identity mismatch')
                if record['has_geometry']:
                    if not geometry.exists():raise ValueError('committed evaluation cache lost its geometry')
                    self.captured[cand.id]=self._capture_read(geometry,record['capture'])
                out.append(record['observations']);self.hits+=1;continue
            result=self.inner.evaluate_many([cand],np.random.default_rng(int(seed)),label=label,force=force)[0]
            cap=getattr(self.inner,'captured',{}).get(cand.id);has_geometry=False;metadata={}
            if cap is not None:
                self.captured[cand.id]=cap
                has_geometry=self._capture_write(geometry,cap)
                metadata={k:v for k,v in cap.items() if k not in {'slabs','probes','rejected'}}
            atomic_json(path,dict(identity=identity,observations=result,has_geometry=has_geometry,capture=metadata))
            out.append(result);self.misses+=1
        return out
