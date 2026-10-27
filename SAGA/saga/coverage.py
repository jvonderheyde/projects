"""Coverage of actual evaluated candidates; independent of predicted fitness."""
from collections import Counter
from itertools import combinations
import numpy as np

def support(cand):
    return tuple(sorted({e for layer in cand.counts.values() for e,n in layer.items() if n}))

def coverage_counts(rows, cfg):
    elements = {e:0 for e in cfg.elements}
    facets, subsets, pairs = Counter(), Counter(), Counter()
    for row in rows:
        cand=row['cand']; present=support(cand)
        for e in present: elements[e]+=1
        facets[int(cand.facet)]+=1
        subsets[(int(cand.facet),present)]+=1
        pairs.update((int(cand.facet),*pair) for pair in combinations(present,2))
    return dict(elements=elements,facets=facets,subsets=subsets,pairs=pairs)

def pick_coverage(proposals, rows, cfg, rng):
    if not proposals:return None
    c=coverage_counts(rows,cfg)
    ranks=[]
    for cand in proposals:
        present=support(cand);facet=int(cand.facet)
        pair_counts=[c['pairs'][(facet,*p)] for p in combinations(present,2)]
        ranks.append((min(c['elements'][e] for e in present),c['facets'][facet],
                      c['subsets'][(facet,present)],
                      float(np.mean(pair_counts)) if pair_counts else 0.,
                      sum(c['elements'][e] for e in present)))
    lowest=min(ranks); ties=[i for i,r in enumerate(ranks) if r==lowest]
    return int(rng.choice(ties))
