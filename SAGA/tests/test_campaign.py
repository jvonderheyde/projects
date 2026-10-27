import copy,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ase import Atoms
from saga.config import Config
from saga.demo import DemoEvaluator
from saga.optimizer import ActiveLoop
from saga.surrogate import select_batch
from saga.slotting import window_quality,pick_window
from saga.validation import probe_reason
from saga.campaign import launch,parse_args

class CampaignTests(unittest.TestCase):
    def config(self,out,**kw):
        values=dict(demo=True,seed=7,out_dir=str(out),n_generations=2,bootstrap_size=8,batch_size=3,
                    offspring_size=24,population_size=8,elite_size=2,n_configs=2,n_probes=2,
                    max_components=10,component_mode='cap',component_scope='global',facets=(100,111),
                    window_slot=True,window_draws=32,finalist_count=2,finalist_repeats=1,validation_n_probes=3,
                    progress=False,collect=False)
        values.update(kw);return Config.for_reaction('nh3_full',**values)
    def test_recipe(self):
        r=json.loads(Path('campaign.json').read_text());reaction=r.pop('reaction');c=Config.for_reaction(reaction,**r)
        self.assertEqual((c.n_generations,c.bootstrap_size,c.batch_size),(60,48,8))
        self.assertEqual(c.objectives['NH2'],(0.,1.));self.assertEqual(c.references['NH2'],[('N2',.5),('H2',1.)])
        self.assertEqual(c.relaxation_steps('ads'),500);self.assertEqual(c.relaxation_steps('slab'),500)
        self.assertEqual(c.adsorbate_weights['N'],2);self.assertEqual(c.adsorbate_weights['NH2'],2)
        self.assertEqual(c.max_components,10);self.assertTrue(c.require_adsorbate_contact);self.assertEqual(c.clean_starts,3)
    def test_reserved_slots_and_window(self):
        with tempfile.TemporaryDirectory() as d:c=self.config(d)
        x=np.random.default_rng(1).normal(size=(8,c.feature_dim));q=np.arange(8,dtype=float)
        picks=select_batch(x,q,5,c,initial=[1,2]);self.assertEqual(picks[:2],[1,2]);self.assertEqual(len(set(picks)),5)
        centers=np.array([sum(c.objectives[a])/2 for a in c.adsorbates]);mu=np.vstack([centers,centers+3,centers+.1]);sd=np.zeros_like(mu)
        score=window_quality(mu,sd,np.ones(3),c,1)
        self.assertEqual(score[0],1.);self.assertLess(score[1],score[2]);self.assertEqual(pick_window(mu,sd,np.ones(3),c,1,exclude=[0]),2)
        self.assertTrue(np.array_equal(score,window_quality(mu,sd,np.ones(3),c,1)))
    def test_contact_uses_every_adsorbate_atom(self):
        c=Config.for_reaction('nh3',slab_size=(1,1,3),n_configs=1,min_valid_configs=1,require_adsorbate_contact=True,molecular_integrity=False)
        clean=Atoms('Ni3',positions=[[0,0,0],[0,0,2],[0,0,4]],cell=[10,10,20],pbc=[1,1,0])
        base=Atoms('NH3',positions=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
        detached=clean+Atoms('NH3',positions=[[0,0,7.2],[0,0,8.2],[1,0,7.2],[0,1,7.2]])
        self.assertEqual(probe_reason(clean,detached,base,True,False,c),'adsorbate_no_contact')
        # Nitrogen is far away, but an original molecule hydrogen contacts Ni.
        contact=detached.copy();contact.positions[4]=[0,0,5.8]
        self.assertIsNone(probe_reason(clean,contact,base,True,False,c))
        translated=contact.copy();translated.positions[4,0]+=10
        self.assertIsNone(probe_reason(clean,translated,base,True,False,c))
    def test_clean_multistart_selects_lowest_valid_energy(self):
        from saga.chemistry import Evaluator
        from saga.genome import random_candidate
        cfg=Config.for_reaction('nh3',n_configs=1,min_valid_configs=1,n_probes=1,clean_starts=3,collect=True,relax_cell=False)
        class Fake(Evaluator):
            def __init__(self,c):self.config=c;self._refs={a:0. for a in c.adsorbates};self.captured={}
            def _relax_many(self,atoms,stage):
                self.starts=atoms if stage=='slab' else self.starts
                result=[]
                for i,a in enumerate(atoms):
                    a=a.copy();e=-float(i) if stage=='slab' else -2.;a.info['nn_energy']=e;result.append((a,e,True,False))
                return result
        e=Fake(cfg);cand=random_candidate(cfg,np.random.default_rng(9));cand.id='test'
        e.evaluate_many([cand],np.random.default_rng(2));chosen=e.captured['test']['slabs'][0]
        self.assertEqual(chosen.info['nn_energy'],-2.)
        self.assertTrue(np.array_equal(e.starts[0].positions[:9],e.starts[1].positions[:9]))
        self.assertFalse(np.array_equal(e.starts[0].positions[9:],e.starts[1].positions[9:]))
    def test_interrupted_generation_reuses_cache_and_replays_exactly(self):
        class Interrupted(Exception):pass
        class Counter(DemoEvaluator):
            def __init__(self,cfg,limit=None):super().__init__(cfg);self.calls=0;self.limit=limit
            def evaluate_many(self,*args,**kw):
                if self.limit is not None and self.calls==self.limit:raise Interrupted()
                self.calls+=1;return super().evaluate_many(*args,**kw)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);a=self.config(d/'full');full_eval=Counter(a);full=ActiveLoop(a,full_eval);full.run()
            b=self.config(d/'resumed');first=Counter(b,limit=10)
            with self.assertRaises(Interrupted):ActiveLoop(b,first).run()
            state=json.loads((d/'resumed/checkpoint.json').read_text());self.assertEqual(state['gen'],0)
            with (d/'resumed/run_log.jsonl').open('ab') as f:f.write(b'{"partial":')
            b.resume=True;b.max_batch_atoms=256;second=Counter(b);resumed=ActiveLoop(b,second);resumed.run()
            self.assertGreaterEqual(resumed.evaluator.hits,2)
            self.assertEqual(first.calls+second.calls,full_eval.calls)
            for key in full.archive:
                self.assertEqual(full.archive[key]['energies'],resumed.archive[key]['energies'])
            self.assertEqual(len(resumed.archive),14)
            events=[json.loads(x) for x in (d/'resumed/run_log.jsonl').read_text().splitlines()]
            ids=[x['event_id'] for x in events if 'event_id' in x];self.assertEqual(len(ids),len(set(ids)))
            slots=[x['slots'] for x in events if x.get('event')=='selection']
            self.assertTrue(all([v['role'] for v in s]==['coverage','window','acquisition'] for s in slots))
            self.assertTrue((d/'resumed/run_log.jsonl.interrupted_tail').exists())
            for row in resumed.validation.values():self.assertEqual(row['stats']['H']['attempted'],6)
    def test_interrupted_final_validation_reuses_completed_candidate(self):
        class Stop(Exception):pass
        class Counter(DemoEvaluator):
            def __init__(self,cfg,limit=None):super().__init__(cfg);self.calls=0;self.limit=limit
            def evaluate_many(self,*args,**kw):
                if self.limit is not None and self.calls==self.limit:raise Stop()
                self.calls+=1;return super().evaluate_many(*args,**kw)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);a=self.config(d/'full');e=Counter(a);full=ActiveLoop(a,e);full.run()
            b=self.config(d/'resume');first=Counter(b,limit=e.calls-1)
            with self.assertRaises(Stop):ActiveLoop(b,first).run()
            state=json.loads((d/'resume/checkpoint.json').read_text());self.assertEqual(state['gen'],2);self.assertEqual(state['validation_round'],0)
            b.resume=True;second=Counter(b);again=ActiveLoop(b,second);again.run()
            self.assertTrue(again.validation_complete);self.assertEqual(first.calls+second.calls,e.calls)
            self.assertEqual(second.calls,1);self.assertEqual(again.evaluator.hits,1)
            for key in full.validation:self.assertEqual(full.validation[key]['energies'],again.validation[key]['energies'])
    def test_launcher_skip_complete_no_evaluator_and_reject_recipe_change(self):
        with tempfile.TemporaryDirectory() as d:
            args=parse_args(['--seed','123','--smoke-test','--root',d]);launch(args)
            with patch('saga.optimizer.ActiveLoop',side_effect=AssertionError('must skip completed run')):launch(args)
            recipe=json.loads(Path('campaign.json').read_text());recipe['adsorbate_weights']['N']=3
            source=Path(d)/'changed.json';source.write_text(json.dumps(recipe));args.recipe=source
            with self.assertRaisesRegex(ValueError,'frozen'):launch(args)
    def test_model_guard_and_shell_explicit_path(self):
        import subprocess,sys,os
        from saga.campaign import source_identity,semantic_config,digest
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);model=d/'actual.model';model.write_bytes(b'model identity fixture')
            recipe=json.loads(Path('campaign.json').read_text());reaction=recipe.pop('reaction')
            out=d/'runs/seed_123';out.mkdir(parents=True)
            cfg=Config.for_reaction(reaction,**recipe,seed=123,out_dir=str(out),model_path=str(model.resolve()))
            identity=dict(schema='big-nh3-campaign-v1',source=source_identity(),model=dict(sha256=digest(model),size_bytes=model.stat().st_size),config=semantic_config(cfg))
            (out/'campaign_manifest.json').write_text(json.dumps(identity))
            env=dict(os.environ,DRY_RUN='1',MAX_BATCH_ATOMS='256',PYTHON=sys.executable,MODEL='wrong-default.model')
            result=subprocess.run(['bash','scripts/submit_big_saga.sh','123',str(model),str(d/'runs')],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn(str(model.resolve()),result.stdout);self.assertNotIn('wrong-default.model',result.stdout)
            self.assertFalse((out/'checkpoint.json').exists())
            model.write_bytes(b'changed model')
            with self.assertRaisesRegex(ValueError,'frozen'):launch(parse_args(['--seed','123','--model',str(model),'--root',str(d/'runs'),'--dry-run']))
    def test_cache_geometry_and_rejected_frames_round_trip(self):
        from saga.evaluation_cache import EvaluationCache
        from saga.genome import random_candidate
        class Inner:
            def __init__(self,cfg):self.cfg=cfg;self.captured={};self.stats={};self.calls=0
            def evaluate_many(self,cands,rng,**kw):
                c=cands[0];self.calls+=1
                clean=Atoms('Ni',positions=[[0,0,0]],cell=[10,10,10]);probe=clean+Atoms('H',positions=[[0,0,1.8]])
                bad=probe.copy();bad.info['rejection']='adsorbate_no_contact'
                self.captured={c.id:dict(id=c.id,layers=c.layers,facet=c.facet,slabs=[clean],probes=[('H',probe,-.4,0)],rejected=[bad])}
                return DemoEvaluator(self.cfg).evaluate_many(cands,rng)
        with tempfile.TemporaryDirectory() as d:
            cfg=self.config(d,collect=True);inner=Inner(cfg);cache=EvaluationCache(inner,cfg,Path(d)/'cache')
            c=random_candidate(cfg,np.random.default_rng(0));c.id='test'
            one=cache.evaluate_many([c],np.random.default_rng(4));two=cache.evaluate_many([c],np.random.default_rng(4))
            self.assertEqual(one,two);self.assertEqual(inner.calls,1);self.assertEqual(len(cache.captured['test']['rejected']),1)
            self.assertEqual(cache.captured['test']['probes'][0][2],-.4)
if __name__=='__main__':unittest.main()
