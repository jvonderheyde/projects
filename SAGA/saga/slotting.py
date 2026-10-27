"""Window reservation independent of scalar weighted acquisition."""
import numpy as np
from .objective import ranking_key

def window_quality(mu,sd,validity,cfg,generation):
    mu=np.asarray(mu,float);sd=np.asarray(sd,float)
    lo=np.array([cfg.objectives[a][0] for a in cfg.adsorbates]);hi=np.array([cfg.objectives[a][1] for a in cfg.adsorbates]);half=(hi-lo)/2
    rng=np.random.default_rng(np.random.SeedSequence([cfg.seed,generation,741093]))
    # Common draws reduce ranking noise; outputs are independent latent Gaussians.
    normal=rng.standard_normal((cfg.window_draws,len(lo)))
    draws=mu[:,None,:]+sd[:,None,:]*normal[None,:,:]
    violation=np.maximum(np.maximum(lo-draws,draws-hi),0.)/half
    return np.asarray(validity)/(1+violation.max(axis=2).mean(axis=1))

def pick_window(mu,sd,validity,cfg,generation,exclude=()):
    if not len(mu):return None
    q=window_quality(mu,sd,validity,cfg,generation)
    q[list(exclude)]=-np.inf
    return int(np.argmax(q)) if np.isfinite(q).any() else None

def finalist_shortlist(rows,cfg):
    eligible=sorted(rows,key=ranking_key,reverse=True);picked=[];seen=set()
    def add(pool,n):
        added=0
        for row in pool:
            if row['cand'].id in seen:continue
            picked.append(row);seen.add(row['cand'].id);added+=1
            if added>=n or len(picked)>=cfg.finalist_count:break
    if not cfg.finalist_count:return []
    add([r for r in eligible if r['assessment']['all_mean_windows_met']],min(4,cfg.finalist_count))
    if len(picked)<cfg.finalist_count:add(eligible,min(2,cfg.finalist_count-len(picked)))
    def near(row):
        violations=[max(cfg.objectives[a][0]-row['stats'][a]['mean'],row['stats'][a]['mean']-cfg.objectives[a][1],0)/((cfg.objectives[a][1]-cfg.objectives[a][0])/2) for a in cfg.adsorbates]
        return (-max(violations),*ranking_key(row))
    if len(picked)<cfg.finalist_count:add(sorted(eligible,key=near,reverse=True),min(2,cfg.finalist_count-len(picked)))
    if len(picked)<cfg.finalist_count:add(eligible,cfg.finalist_count-len(picked))
    return picked
