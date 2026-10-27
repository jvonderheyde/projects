"""Archive-based search with post-evaluation survival and independent validation."""

from pathlib import Path
import hashlib
import json
import time
import uuid
import numpy as np
from .genome import (
    candidate_dict,
    candidate_from_dict,
    genome_key,
    random_candidate,
    mutate,
    crossover,
    to_features,
)
from .statistics import empty_observations, merge_observations, summarize
from .objective import assess, ranking_key
from .surrogate import (
    Surrogate,
    acquisition,
    select_batch,
    validity_prediction,
)


def _write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _model_identity(cfg):
    if cfg.demo:
        return {"kind": "synthetic-demo-v1"}
    path = Path(cfg.model_path or "")
    if not path.is_file():
        raise ValueError("a readable MACE model file is required")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "size_bytes": path.stat().st_size}


class ActiveLoop:
    def __init__(self, config, evaluator=None):
        config.validate()
        self.config = config
        self.out = Path(config.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        checkpoint = self.out / "checkpoint.json"
        if not config.resume and checkpoint.exists():
            raise FileExistsError(
                f"{checkpoint} already exists: choose another output directory or use --resume"
            )
        if config.resume and not checkpoint.exists():
            raise FileNotFoundError(f"no checkpoint to resume: {checkpoint}")
        self.model_identity = (
            {"kind": "injected-test-evaluator"}
            if evaluator is not None
            else _model_identity(config)
        )
        self.source_identity = {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob('*.py'))}
        self.run_id = uuid.uuid4().hex
        streams = np.random.SeedSequence(config.seed).spawn(3)
        self.search_rng, self.eval_rng, self.validation_rng = [
            np.random.default_rng(s) for s in streams
        ]
        self.archive = {}
        self.population = []
        self.history = []
        self.surrogate = Surrogate(config)
        self.generation = 0
        self.next_id = 0
        self.batch_counter = 0
        self.evaluations = 0
        self.no_improve = 0
        self.previous_quality = None
        self.validation = {}
        self.finalist_keys = []
        self.validation_round = 0
        self.validation_generation = None
        self.elapsed = 0.0
        self.validation_complete = False
        self.prequential = []
        if config.resume:
            self._restore(json.loads(checkpoint.read_text()))
        if evaluator is None:
            if config.demo:
                from .demo import DemoEvaluator

                evaluator = DemoEvaluator(config)
            else:
                from .chemistry import make_evaluator

                evaluator = make_evaluator(config)
        from .evaluation_cache import EvaluationCache
        self.evaluator = EvaluationCache(evaluator,config,self.out/'evaluation_cache')
        self._seen_log_keys=set()
        self._repair_log()
        if not config.resume: self._checkpoint()

    def _repair_log(self):
        path=self.out/'run_log.jsonl'
        if not path.exists():return
        data=path.read_bytes(); good=[]; bad=b''
        for line in data.splitlines(keepends=True):
            try: event=json.loads(line)
            except (ValueError,UnicodeDecodeError): bad+=line;continue
            if bad: raise ValueError('invalid nonterminal JSONL record; inspect run_log.jsonl')
            good.append(line.rstrip(b'\n')+b'\n')
            key=self._log_key(event)
            if key is not None:self._seen_log_keys.add(key)
        if bad:
            tail=path.with_name(path.name+'.interrupted_tail')
            with tail.open('ab') as f:f.write(bad)
            path.write_bytes(b''.join(good))

    @staticmethod
    def _log_key(data):
        if 'event_id' in data:return ('evaluation',data['event_id'])
        if data.get('event') in {'generation','selection'}:return (data['event'],data['gen'])
        return None

    @property
    def dataset(self):
        return list(self.archive.values())

    def _log(self, data):
        key=self._log_key(data)
        if key is not None and key in self._seen_log_keys:return
        if key is not None:self._seen_log_keys.add(key)
        with (self.out / "run_log.jsonl").open("a") as f:
            f.write(json.dumps(data, allow_nan=False) + "\n")

    def _row_json(self, row):
        out = {k: v for k, v in row.items() if k != "cand"}
        out.update(candidate_dict(row["cand"]))
        return out

    def _row_restore(self, data):
        c = candidate_from_dict(data, self.config)
        row = {
            k: v
            for k, v in data.items()
            if k not in {"counts", "layers", "facet", "id"}
        }
        row["cand"] = c
        row["features"] = to_features(c, self.config).tolist()
        row["stats"] = summarize(row["observations"], self.config)
        row["assessment"] = assess(row["stats"], self.config)
        return row

    def _checkpoint(self):
        _write_json(
            self.out / "checkpoint.json",
            dict(
                version=3,
                run_id=self.run_id,
                config=self.config.to_dict(),
                model_identity=self.model_identity,
                source_identity=self.source_identity,
                gen=self.generation,
                next_id=self.next_id,
                batch_counter=self.batch_counter,
                n_evaluations=self.evaluations,
                n_unique=len(self.archive),
                dataset=[self._row_json(r) for r in self.dataset],
                population=[candidate_dict(c) for c in self.population],
                history=self.history,
                rng_states=[
                    r.bit_generator.state
                    for r in (self.search_rng, self.eval_rng, self.validation_rng)
                ],
                kernels=self.surrogate.kernel_state(),
                no_improve=self.no_improve,
                previous_quality=self.previous_quality,
                elapsed=self.elapsed,
                validation=[self._row_json(r) for r in self.validation.values()],
                finalists=[
                    candidate_dict(self.archive[k]["cand"]) for k in self.finalist_keys
                ],
                validation_round=self.validation_round,
                validation_generation=self.validation_generation,
                validation_complete=self.validation_complete,
                prequential=self.prequential,
            ),
        )

    def _restore(self, state):
        if state.get("version") != 3:
            raise ValueError(
                "Rebuilt SAGA requires a version-3 checkpoint; start a new run for old results"
            )
        if state.get('source_identity') != self.source_identity:
            raise ValueError('SAGA source/search policy changed or legacy checkpoint lacks its hash; use a fresh output directory')
        old = state["config"]
        new = self.config.to_dict()
        allowed = {"resume", "out_dir", "progress", "n_generations", "model_path", "max_batch_atoms", "max_batch_edges", "autotune", "device"}
        mismatches = [
            k
            for k in new
            if k not in allowed
            and json.dumps(old.get(k), sort_keys=True)
            != json.dumps(new[k], sort_keys=True)
        ]
        if mismatches:
            raise ValueError("resume settings differ: " + ", ".join(mismatches))
        if state["model_identity"] != self.model_identity:
            raise ValueError("model content differs from checkpoint")
        if self.config.n_generations < state["gen"]:
            raise ValueError("cannot rewind generations; use a separate run")
        self.run_id = state["run_id"]
        self.generation = state["gen"]
        self.next_id = state["next_id"]
        self.batch_counter = state["batch_counter"]
        self.evaluations = state["n_evaluations"]
        for data in state["dataset"]:
            row = self._row_restore(data)
            self.archive[genome_key(row["cand"], self.config)] = row
        self.population = [
            candidate_from_dict(c, self.config) for c in state["population"]
        ]
        self.history = state["history"]
        self.elapsed = state["elapsed"]
        for rng, s in zip(
            (self.search_rng, self.eval_rng, self.validation_rng), state["rng_states"]
        ):
            rng.bit_generator.state = s
        self.surrogate.restore_kernels(state["kernels"])
        self.no_improve = state["no_improve"]
        self.previous_quality = state["previous_quality"]
        self.validation = {}
        for data in state.get("validation", []):
            row = self._row_restore(data)
            self.validation[genome_key(row["cand"], self.config)] = row
        self.finalist_keys = [
            genome_key(candidate_from_dict(c, self.config), self.config)
            for c in state.get("finalists", [])
        ]
        self.validation_round = state.get("validation_round", 0)
        self.validation_generation = state.get("validation_generation")
        self.validation_complete = state.get("validation_complete", False)
        self.prequential = state.get("prequential", [])

    def _record(self, cand, result, target):
        cfg = self.config
        key = genome_key(cand, cfg)
        row = target.get(key)
        if row is None:
            row = {
                "cand": cand,
                "features": to_features(cand, cfg).tolist(),
                "gen": self.generation,
                "observations": empty_observations(cfg),
                "n_evals": 0,
            }
            target[key] = row
        merge_observations(row["observations"], result, cfg)
        row["n_evals"] += 1
        row["stats"] = summarize(row["observations"], cfg)
        row["assessment"] = assess(row["stats"], cfg)
        # Familiar fields retained for inspection/export; means are arrangement-weighted.
        row["energies"] = {a: row["stats"][a]["mean"] for a in cfg.adsorbates}
        row["se"] = {a: row["stats"][a]["se"] for a in cfg.adsorbates}
        row["n_samples"] = {a: row["stats"][a]["n"] for a in cfg.adsorbates}
        row["composite"] = row["assessment"]["composite"]
        return row

    def _evaluate(self, cands, phase="search", predictions=None):
        if not cands:
            return []
        cfg = self.config
        for c in cands:
            key = genome_key(c, cfg)
            if key in self.archive:
                c.id = self.archive[key]["cand"].id
            elif c.id is None:
                c.id = f"c{self.next_id:06d}"
                self.next_id += 1
        target = self.validation if phase == "validation" else self.archive
        rng = self.validation_rng if phase == "validation" else self.eval_rng
        results = self.evaluator.evaluate_many(
            cands, rng, label=f"{phase} gen {self.generation}"
        )
        if len(results) != len(cands):
            raise ValueError("evaluator returned incorrect number of results")
        self.batch_counter += 1
        self.evaluations += len(cands)
        rows = [self._record(c, res, target) for c, res in zip(cands, results)]
        captured = getattr(self.evaluator, "captured", {})
        captures = []
        for i, (c, row) in enumerate(zip(cands, rows)):
            event_id = f"{self.run_id}:b{self.batch_counter}:{c.id}"
            event = dict(
                event="evaluation",
                phase=phase,
                event_id=event_id,
                gen=self.generation,
                id=c.id,
                stats=row["stats"],
                assessment=row["assessment"],
            )
            if predictions is not None and i < len(predictions[0]):
                mu, sd = predictions
                errors = {}
                for j, a in enumerate(cfg.adsorbates):
                    actual = row["stats"][a]["mean"]
                    se = row["stats"][a]["se"]
                    if actual is not None and se is not None:
                        errors[a] = {
                            "error": float(actual - mu[i, j]),
                            "standardized_error": float(
                                (actual - mu[i, j])
                                / np.sqrt(sd[i, j] ** 2 + se**2 + cfg.noise_floor**2)
                            ),
                        }
                event["prequential"] = errors
                self.prequential.append(
                    {"id": c.id, "gen": self.generation, "errors": errors}
                )
            self._log(event)
            if c.id in captured:
                cap = dict(captured[c.id])
                cap.update(
                    run_id=self.run_id,
                    evaluation_id=event_id,
                    composite=row["composite"],
                )
                captures.append(cap)
        if cfg.collect and captures:
            from .collect import write_candidates

            write_candidates(
                captures,
                str(self.out / cfg.collect_dir / phase),
                self.batch_counter,
                per_layer=cfg.atoms_per_layer,
            )
        return rows

    def _random_pool(self, n, exclude=()):
        seen = set(exclude)
        out = []
        for _ in range(max(200, 30 * n)):
            if len(out) >= n:
                break
            c = random_candidate(self.config, self.search_rng)
            k = genome_key(c, self.config)
            if k not in seen:
                seen.add(k)
                out.append(c)
        return out

    def _bootstrap(self):
        cfg = self.config
        if cfg.component_mode=='exact' and cfg.component_scope=='global' and cfg.max_components==1:
            from .genome import from_matrix
            pool=[]
            for facet in cfg.facets:
                for i in range(len(cfg.elements)):
                    matrix=np.zeros((cfg.n_layers,len(cfg.elements)),dtype=int)
                    matrix[:,i]=cfg.atoms_per_layer
                    pool.append(from_matrix(matrix,facet,cfg))
        else:
            pool = self._random_pool(cfg.bootstrap_size * 10)
        X = np.array([to_features(c, cfg) for c in pool])
        # Pure farthest-point initial design, with explicitly masked selections.
        idx = select_batch(X, np.zeros(len(X)), cfg.bootstrap_size, cfg)
        init = [pool[i] for i in idx]
        self._evaluate(init)
        self.population = init[: cfg.population_size]
        self._summary(0, len(init), 0)
        self._checkpoint()

    def _proposals(self):
        cfg = self.config
        seen = set(self.archive)
        seen.update(genome_key(c, cfg) for c in self.population)
        # Include unevaluated survivors so promising proposals aren't forgotten.
        out = [c for c in self.population if genome_key(c, cfg) not in self.archive]
        n = max(cfg.offspring_size, cfg.batch_size)
        for _ in range(20 * n):
            if len(out) >= n:
                break
            if not self.population or self.search_rng.random() < cfg.random_fraction:
                c = random_candidate(cfg, self.search_rng)
            else:
                a, b = [
                    self.population[int(self.search_rng.integers(len(self.population)))]
                    for _ in range(2)
                ]
                c = crossover(a, b, cfg, self.search_rng)
                if self.search_rng.random() < cfg.mutation_rate:
                    c = mutate(c, cfg, self.search_rng)
                if genome_key(c, cfg) in seen:
                    c = mutate(c, cfg, self.search_rng)
            k = genome_key(c, cfg)
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
        if len(out) < n:
            out += self._random_pool(n - len(out), exclude=seen)
        return out

    def _fit(self):
        cfg = self.config
        rows = self.dataset
        X = np.array([r["features"] for r in rows])
        Y = np.array(
            [
                [
                    (
                        r["stats"][a]["mean"]
                        if r["stats"][a]["mean"] is not None
                        else np.nan
                    )
                    for a in cfg.adsorbates
                ]
                for r in rows
            ]
        )
        noise = np.array(
            [
                [
                    r["stats"][a]["se"] if r["stats"][a]["se"] is not None else np.nan
                    for a in cfg.adsorbates
                ]
                for r in rows
            ]
        )
        self.surrogate.fit(X, Y, noise, generation=self.generation)

    def _rank_proposals(self, proposals):
        cfg = self.config
        X = np.array([to_features(c, cfg) for c in proposals])
        if not len(proposals):
            return X, np.zeros(0), None
        mu, sd = self.surrogate.predict(X)
        ucb, _, _ = acquisition(mu, sd, cfg)
        valid = validity_prediction(X, self.dataset, cfg)
        # Acquisition only. Population diversity is carried by batch_diversity
        # in _survive; element coverage by the reserved slot in run().
        return X, ucb * valid, (mu, sd)

    def _element_counts(self):
        """How many archive rows contain each element (coverage, not fitness)."""
        from .coverage import coverage_counts
        return coverage_counts(self.dataset, self.config)["elements"]

    def _coverage_pick(self, proposals, quality):
        """Reserve one genuine evaluation for under-sampled chemistry/facets."""
        from .coverage import pick_coverage
        return pick_coverage(proposals, self.dataset, self.config, self.search_rng)

    def _refinement(self, n):
        # Refine competitive uncertain candidates, not the same top means forever.
        cfg = self.config
        rows = self.dataset
        eligible = [r for r in rows if r["assessment"]["eligible"]]
        threshold = max(
            (r["assessment"]["conservative_score"] for r in eligible), default=0.0
        )
        centers = np.array([sum(cfg.objectives[a]) / 2 for a in cfg.adsorbates])
        choices = []
        for r in rows:
            # Persistent structural failures need new chemistry, not repeated elite slots.
            if (
                min(r["stats"][a]["valid_fraction"] for a in cfg.adsorbates)
                < cfg.min_valid_fraction
            ):
                continue
            if any(r["stats"][a]["mean"] is None for a in cfg.adsorbates):
                continue
            mu = np.array([r["stats"][a]["mean"] for a in cfg.adsorbates])
            se = np.array([r["stats"][a]["se"] for a in cfg.adsorbates])
            nearest = centers + np.maximum(
                np.abs(mu - centers) - cfg.confidence_z * se, 0.0
            )
            from .objective import composite_from_energies

            upper = float(composite_from_energies(nearest, cfg))
            if upper >= threshold:
                ambiguity = upper - r["assessment"]["conservative_score"]
                choices.append((ambiguity, r["assessment"]["composite"], r))
        choices.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [r["cand"] for _, _, r in choices[:n]]

    def _survive(self, proposals):
        cfg = self.config
        ranked = sorted(self.dataset, key=ranking_key, reverse=True)
        keep = [r["cand"] for r in ranked[: cfg.elite_size]]
        seen = {genome_key(c, cfg) for c in keep}
        candidates = []
        for c in [r["cand"] for r in ranked] + proposals:
            k = genome_key(c, cfg)
            if k not in seen:
                seen.add(k)
                candidates.append(self.archive[k]["cand"] if k in self.archive else c)
        if candidates and len(keep) < cfg.population_size:
            self._fit()
            X, q, _ = self._rank_proposals(candidates)
            if keep:
                from .surrogate import distances

                q = q + cfg.batch_diversity * distances(
                    X, np.array([to_features(c, cfg) for c in keep]), cfg
                ).min(1)
            idx = select_batch(X, q, cfg.population_size - len(keep), cfg)
            keep.extend(candidates[i] for i in idx)
        self.population = keep

    def _coverage_summary(self):
        from .coverage import coverage_counts
        c = coverage_counts(self.dataset, self.config)
        return dict(elements=c['elements'],facets=dict(c['facets']),
                    distinct_element_subsets=len(c['subsets']),
                    distinct_element_pair_facets=len(c['pairs']))

    def _summary(self, gen, n_new, n_reval):
        rows = self.dataset
        best = max(rows, key=ranking_key)
        quality = max(
            (r["assessment"]["conservative_score"] for r in rows), default=0.0
        )
        # Compare adjacent updated evidence, not a historical lucky maximum.
        if (
            self.previous_quality is None
            or quality > self.previous_quality + self.config.improvement_tol
        ):
            self.no_improve = 0
        else:
            self.no_improve += 1
        self.previous_quality = quality
        summary = dict(
            gen=gen,
            n_unique=len(rows),
            n_evaluations=self.evaluations,
            n_new=n_new,
            n_reval=n_reval,
            best_id=best["cand"].id,
            best_composite=best["composite"],
            best_conservative_score=quality,
            n_eligible=sum(r["assessment"]["eligible"] for r in rows),
            coverage=self._coverage_summary(),
        )
        self.history.append(summary)
        self._log(dict(event="generation", **summary))
        if self.config.progress:
            print(
                f"gen {gen}: unique={len(rows)} eligible={summary['n_eligible']} new={n_new} refine={n_reval} best={best['composite']:.4f} conservative={quality:.4f}",
                flush=True,
            )

    def run(self):
        start = time.perf_counter()
        cfg = self.config
        if not self.archive:
            self._bootstrap()
        for gen in range(self.generation + 1, cfg.n_generations + 1):
            self.generation = gen
            self._fit()
            # k=1 has a finite exhaustive pool: every layer must contain the
            # same single element. More generations cannot create new chemistry.
            if (cfg.component_mode == 'exact' and cfg.component_scope == 'global'
                    and cfg.max_components == 1
                    and len(self.archive) >= len(cfg.elements) * len(cfg.facets)):
                self.generation = gen-1
                self._log(dict(event='finite_pool_exhausted', gen=self.generation))
                break
            proposals = self._proposals()
            X, q, pred = self._rank_proposals(proposals)
            reserved=[]; roles=[]
            if cfg.coverage_explore and proposals:
                pick=self._coverage_pick(proposals,q)
                if pick is not None:reserved.append(pick);roles.append('coverage')
            if cfg.window_slot and len(reserved)<cfg.batch_size and pred is not None:
                from .slotting import pick_window
                pick=pick_window(pred[0],pred[1],validity_prediction(X,self.dataset,cfg),cfg,self.generation,exclude=reserved)
                if pick is not None:reserved.append(pick);roles.append('window')
            idx=select_batch(X,q,cfg.batch_size,cfg,initial=reserved)
            roles+=['acquisition']*(len(idx)-len(roles))
            batch = [proposals[i] for i in idx]
            reval = self._refinement(
                cfg.reval_elites + max(0, cfg.batch_size - len(batch))
            )
            selected_pred = (pred[0][idx], pred[1][idx]) if batch else None
            self._evaluate(batch + reval, predictions=selected_pred)
            self._log(dict(event='selection',gen=gen,slots=[dict(role=role,id=c.id) for role,c in zip(roles,batch)]))
            self._survive(proposals)
            self._summary(gen, len(batch), len(reval))
            now = time.perf_counter()
            self.elapsed += now - start
            start = now
            self._checkpoint()
            if cfg.early_stop and self.no_improve >= cfg.patience:
                break
        original_config=self.evaluator.config
        try:
            if cfg.validation_n_probes is not None:
                from dataclasses import replace
                self.evaluator.config=replace(cfg,n_probes=cfg.validation_n_probes)
            self._validate_finalists()
        finally:self.evaluator.config=original_config
        self.elapsed += time.perf_counter() - start
        self._checkpoint()
        report = self._report()
        _write_json(self.out / "report.json", report)
        return report

    def _validate_finalists(self):
        cfg = self.config
        if not cfg.finalist_count or not cfg.finalist_repeats:
            return
        if self.validation_generation != self.generation:
            # Freeze the shortlist before observing any validation outcomes.
            rows = sorted(self.dataset, key=ranking_key, reverse=True)
            eligible = [r for r in rows if r["assessment"]["eligible"]]
            from .slotting import finalist_shortlist
            self.finalist_keys = [genome_key(r['cand'],cfg) for r in finalist_shortlist(eligible,cfg)]
            self.validation = {}
            self.validation_round = 0
            self.validation_complete = False
            self.validation_generation = self.generation
            self._checkpoint()
        finalists = [self.archive[k]["cand"] for k in self.finalist_keys]
        if not finalists:
            self.validation_complete=True
            return
        for r in range(self.validation_round, cfg.finalist_repeats):
            self._evaluate(finalists, phase="validation")
            self.validation_round = r + 1
            self._checkpoint()
        self.validation_complete = True

    def _report(self):
        search = sorted(self.dataset, key=ranking_key, reverse=True)
        validation = sorted(self.validation.values(), key=ranking_key, reverse=True)
        eligible = [r for r in validation if r["assessment"]["eligible"]]

        def brief(row):
            return dict(
                **candidate_dict(row["cand"]),
                energies=row["energies"],
                se=row["se"],
                stats=row["stats"],
                **row["assessment"],
            )

        error_summary = {}
        for a in self.config.adsorbates:
            errors = [r["errors"][a] for r in self.prequential if a in r["errors"]]
            if errors:
                error_summary[a] = {
                    "n": len(errors),
                    "rmse_eV": float(
                        np.sqrt(np.mean([r["error"] ** 2 for r in errors]))
                    ),
                    "fraction_abs_z_le_1_96": float(
                        np.mean([abs(r["standardized_error"]) <= 1.96 for r in errors])
                    ),
                }
        return dict(
            version=3,
            run_id=self.run_id,
            config=self.config.to_dict(),
            model_identity=self.model_identity,
            source_identity=self.source_identity,
            final_generation=self.generation,
            n_unique=len(self.archive),
            n_evaluations=self.evaluations,
            best=brief(eligible[0]) if eligible else None,
            best_source="independent_finalist_validation" if eligible else None,
            search_best=brief(search[0]),
            validation_status=(
                "complete"
                if self.validation_complete
                else "no_completed_eligible_validation"
            ),
            finalists=[brief(r) for r in validation],
            search_top=[brief(r) for r in search[: max(self.config.finalist_count, 5)]],
            prequential_diagnostics=error_summary,
            generations=self.history,
            timing={"wall_seconds": self.elapsed},
            evaluator_stats_current_session=getattr(self.evaluator, "stats", {}),
            notes=[
                "Objective: Gaussian window score of equally weighted valid arrangement means.",
                "Structural filtering changes the estimand to valid adsorption states; failure rates are reported separately.",
                "Standard errors use regularized arrangement/placement variance; interval coverage is not guaranteed.",
                "Independent validation reduces search selection bias; choosing its best still introduces shortlist selection bias.",
                "MACE model error is not included in the sampling error bars.",
            ],
        )
