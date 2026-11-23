# Coverage-ranked grid search with LP feasibility

The new verified Hard record is **-1.5365853658536586**, improved from
-1.5223081499107673 by 0.01427721594289122 in fitness magnitude.
All 40 satellites remain in every observation. The independent full evaluator
and transition matrices recomputed at tolerance 1e-12 give the same fitness.
The old record is preserved in solutions/starts/hard_1_522308.json.

Implementation: mission/lp_grid_search.py. Runner: scripts/run_lp_grid_search.py.
The search logic is implemented here; linear programs use OR-Tools GLOP.
The orbital model, grid mapping, cropping and objective are unchanged.

## What changed in the search

Earlier optimizers relied on a limited sample of continuous states. This search
first identifies the weakest observation and enumerates its 3D grid cells.
For each target cell, a linear program seeks a continuous six-component state
whose projections remain inside the observation cube at every epoch and whose
decision variables stay within bounds. Whitening with the existing constraint
geometry improves numerical conditioning. The recovered state is independently
checked against the original projection and decision-bound rows.

Next, temporarily remove one satellite from the baseline counters. Precompute
the exact baseline support after inserting a point at every 2D cell in XY, XZ
and YZ. The score of a 3D target cell is the sum of three table lookups. Sort
feasible target cells by that score. A target whose support sum cannot exceed
the incumbent's weakest observation cannot improve the primary fitness and is
pruned without scoring a complete formation.

For a promising cell, a second LP minimizes the largest displacement of that
satellite's projected coordinates at the other epochs relative to its old state.
Evaluate that witness and, if needed, the catalog witness in the full formation.
Accept only a strict primary improvement, verified by the full evaluator. The
default stops at the first improvement; a larger --improvements value repeats
the procedure from the new archive.

## The successful update

The catalog contained 2856 recovered feasible cells out of 21^3=9261 targets.
The first two satellite replacements had no higher-scoring critical cell in
this catalog. For satellite index 2 (the third satellite), the target cell
(0, 7, 0) at observation index 1 led to a verified improvement.
Construction and the reference-conditioned query used 9262 LP solves in total.
The engine's formation-score counter is only 2: the initial formation and the
accepted candidate. This does not mean there were only two computations: all
catalog LPs, insertion tables, grid ranking and verification are additional work.

The new observation totals are approximately 1.713860797145,
1.536585365854 and 1.574657941701. The middle observation still sets fitness.
The update improves primary fitness while allowing the other observations to
decrease, provided the true max-min objective improves.

## Limits and correctness claims

This is a coordinate search heuristic, not an exact global best-response solver.
An epsilon of 1e-5 keeps candidates inside grid and observation boundaries.
That excludes some boundary-adjacent states. GLOP uses numerical tolerances;
only recovered states passing explicit checks enter the catalog. Returning no
witness is not used as a global mission infeasibility certificate.

Each critical cell gets a catalog witness and at most one reference-conditioned
witness per satellite. Other continuous states in that same cell can behave
differently at the other epochs. The method therefore cannot prove absence of
an improving satellite move or global optimality when a scan finds none.
Requiring survival at every epoch is also stricter than the mission evaluator,
which permits cropping. -1.55 has not been reached by the published update.

LP-call and improvement limits bound the search. A tiny LP budget preserves the
incumbent and reports lp_limit; it never publishes a partial formation. The best
archive never worsens. Saved verification counts cover engine checks; runner
input, checkpoint and output checks are extra.

For grid width G, catalog construction uses G^3 small LPs per visited epoch.
Per satellite, tables cost O(G^2*(B+N)) with set unions, where B is baseline
support and N satellite count. Sorting C catalog cells costs O(C log C).
Promising cells require further LPs and full or incremental formation checks.
These costs are distinct from the final formation-score count.

## Reproduce

```bash
python -m pip install -r requirements-lp.txt
python scripts/run_lp_grid_search.py --start solutions/starts/hard_1_522308.json --improvements 1 --lp-limit 50000 --epsilon 0.00001 --output results/lp_grid_search/reproduce
python scripts/verify_solution.py solutions/hard.json
```

The requirement supplies the shared OR-Tools package, including GLOP. The runner
also recognizes this workspace's optional .runtime/cpsat installation.
The original resumed_from path in the saved record points to solutions/hard.json
before replacement; the tracked archive above preserves its exact state.

108 tests passed in the isolated LP publication tree with OR-Tools enabled. LP tests compare insertion tables
with independent full baseline sets, check cell recovery and bounds, preserve
archives under tiny budgets, protect existing output, and reproduce an actual
Hard improvement from the archived incumbent. The final saved record was also
re-evaluated separately with tighter transition-matrix integration tolerance.

Official GLOP interface documentation:
https://developers.google.com/optimization/lp/lp_example.
