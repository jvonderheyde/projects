import sys,types,unittest
from unittest.mock import patch
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from saga.config import Config
from saga import chemistry
from saga.geometry import build_adsorbate
class LiteralStepTests(unittest.TestCase):
    def test_caps_for_both_engines_and_algorithms(self):
        cfg=Config.for_reaction('nh3_full',relax_cell=False)
        self.assertEqual(cfg.relax_algo,'lbfgs')
        seen=[]
        class Optimizer:
            def __init__(self,*args,**kwargs):pass
            def run(self,*,fmax,steps):seen.append(steps);return True
        def fake_relax(engine,systems,*,steps,**kwargs):
            seen.append(steps)
            return types.SimpleNamespace(n_forward=1,energy=np.zeros(len(systems)),
                cells=[np.asarray(at.cell) for at in systems],positions=[at.positions.copy() for at in systems],
                converged=np.ones(len(systems),dtype=bool),failed=np.zeros(len(systems),dtype=bool))
        engine_module=types.ModuleType('saga.engine')
        engine_module.Sys=types.SimpleNamespace(from_atoms=lambda at:at)
        relax_module=types.ModuleType('saga.relax');relax_module.relax=fake_relax
        for cap in (500,37):
            for algo in ('lbfgs','fire'):
                cfg.relax_algo=algo
                cfg.slab_relax_steps=cfg.ads_relax_steps=cfg.ref_relax_steps=cap
                for cls in (chemistry.Evaluator,chemistry.BatchedEvaluator):
                    obj=cls.__new__(cls);obj.config=cfg;obj.calc=EMT();obj.engine=object()
                    obj.stats={k:0 for k in ('slab_relax','probe_relax','reference_relax','forwards')}
                    for stage in ('slab','ads','ref'):
                        at=Atoms('Ni',positions=[[0,0,0]],cell=[10,10,10])
                        with patch.object(chemistry,'LBFGS',Optimizer),patch('ase.optimize.FIRE',Optimizer),patch.dict(sys.modules,{'saga.engine':engine_module,'saga.relax':relax_module}):
                            obj._relax_many([at],stage)
                        self.assertEqual(seen[-1],cap)
        self.assertEqual(len(seen),24)

    def test_preset_and_builders_unchanged(self):
        cfg=Config.for_reaction('nh3_full')
        self.assertEqual(cfg.objectives['NH'],(-.2,.8))
        self.assertEqual(cfg.objectives['NH2'],(0.,1.))
        self.assertEqual(cfg.references['NH'],[('N2',.5),('H2',.5)])
        self.assertEqual(cfg.references['NH2'],[('N2',.5),('H2',1.)])
        for sp,n in [('NH',2),('NH2',3)]:
            at=build_adsorbate(sp);self.assertEqual(len(at),n)
            self.assertEqual(sum(at.numbers==7),1);self.assertEqual(sum(at.numbers==1),n-1)


