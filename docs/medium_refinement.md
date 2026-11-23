# Medium refinement toward -2.0

The previous published Medium fitness was -1.7792980368828077. It is preserved
in solutions/starts/medium_1_779298.json. ALNS seed 2000 improved it to
-1.8851873884592505; that intermediate is solutions/starts/medium_1_885187.json.
The first subsequent SA stage, seed 2100, reached **-1.8875669244497324**.
The complete verified SA record is now solutions/medium.json. Full scoring at
transition-matrix tolerance 1e-12 gives the same fitness. Remaining satellites
are 40, 39, 40; cropping follows the unchanged mission evaluator.

The user stopped further search at this value. -2.0 was not reached. No claim
of impossibility or global optimality follows from these experiments.

## Reproduce the two improving stages

```bash
python scripts/run_alns.py --start solutions/starts/medium_1_779298.json --target -2.0 --evaluations 100000 --pool-size 16 --min-destroy 2 --max-destroy 4 --initial-temperature 0.002 --seed 2000 --output results/medium-reproduce-alns
python scripts/seek_targets.py --case medium --start solutions/starts/medium_1_885187.json --target -2.0 --attempts 1 --steps 15000 --initial-temperature 0.004 --seed 2100 --output results/medium-reproduce-sa
python scripts/verify_solution.py solutions/medium.json
```

The saved resumed_from paths refer to original local experiment locations.
Use the tracked warm starts above to reproduce the stages after checkout.
The initial ALNS stage used its recorded score budget; final SA used 15005
candidate scores. Earlier warm-start discovery and verification costs are extra.

## Additional trials and interruptions

medium_target2_experiments.csv records complete runs and last verified
checkpoints. For interrupted runs scores_lower_bound is only the last archived
count, not all work performed. Complete records contain actual score counts.
ALNS counts partial destruction/insertion scores; SA and Tabu count different
proposal types. This is parameter exploration, not an equal-work comparison.

ALNS seed 2001 used groups 4–8, pool 10 and initial temperature 0.006.
Tabu seed 2002 used 30000 scores, neighborhood 32, tenure 13, radius 0.8 and
24 intervals. ALNS 2003 tried block recombination from a saved 2000 checkpoint
but stopped after a Windows checkpoint replacement error. The last checkpoint
remains valid; atomic publication now retries temporary PermissionError failures.
Tests also ensure a persistent lock preserves the existing checkpoint.

ALNS 2004 continued from seed 2000 with groups 2–4, pool 16, temperature 0.001
and a planned 200000-score budget. ALNS 2005 used groups 2–3, pool 24,
temperature 0.004, block probability 0.25 and a planned 150000-score budget.
Both were stopped by the user before completion. SA completed six 15000-step
stages starting at seed 2100; later stages did not improve primary fitness.
An experimental planar/vertical assignment chain was also stopped before its
first result was written, so no final result or exact score count is reported.

Block recombination is optional and checks exact separation of the computed
linear model. It is not used in the two published improving stages. Do not
attribute the published Medium gain to this optional operator. See
alns_algorithm.md for its feasibility checks and rejection on the coupled HALO
model. All 103 tests passed before publication of the saved results.
