# Grid-aware Tabu Search

Implementation: mission/tabu.py; command: scripts/run_tabu.py.
The objective and orbital model are unchanged. Lower fitness is better.

Each iteration generates a bounded batch of feasible one-satellite moves.
Chord proposals sample interiors of intervals separated by grid transitions,
avoiding objective evaluations for unchanged projected cells. Optional cyclic
metric projections seek neighboring cells, including coordinated shifts in
several axes. These projections are a bounded feasibility heuristic: failure
does not establish infeasibility. Other observation epochs can have cell slack.

Candidates use the incremental FormationState evaluator. Select the best
admissible candidate by (negative worst observation, negative total coverage),
even if it worsens the current state. Remember the previous satellite cell
signature to forbid immediate reversal for the configured tenure. Aspiration
admits a tabu destination only when it improves the best archived key.
Always return the archive, not the last visited state.

The evaluation budget includes the initial formation and every scored proposal,
including rejected tabu proposals. Final full verification is reported separately.
Geometric attempts are bounded to eight times the neighborhood size per batch;
they cost runtime but are not objective evaluations. Stop on the budget, target,
or a batch without scored neighbors. A finite budget guarantees termination,
not global optimality. Archive fitness never worsens.

With N satellites, M observations and B baseline counters per plane, a scored
one-satellite replacement costs O(M*(N+B)) on average with hash counters.
Chord construction also sorts up to O(M*G) grid transitions. Projection proposals
perform at most 128 cycles over the constraints. Neighborhood size and those
geometric costs must be included when comparing wall-clock performance.

Tests cover full versus incremental scores, decision bounds, requested cells,
reverse memory, aspiration, preserved incumbents, deterministic seeds and output
overwrite protection. Generic solution verification checks the formation and
fitness; it does not independently certify the tabu execution metadata.

## Published Hard experiment

solutions/hard.json contains seed 904, fitness -1.5139797739440808.
Full re-evaluation with transition-matrix tolerance 1e-12 gives the same score.
The previous SA result is preserved in solutions/starts/hard_1_505651.json.
The exact seed-904 input is solutions/starts/hard_tabu_seed901.json; it includes
its own search history and original local provenance path. The earlier input
for seed 901 is preserved as solutions/starts/hard_tabu_seed900.json.

```bash
python scripts/run_tabu.py --start solutions/starts/hard_tabu_seed901.json --seed 904 --evaluations 30000 --target -1.515 --neighbors 32 --tenure 13 --radius 0.8 --intervals 24 --output results/tabu/reproduce-904.json
python scripts/verify_solution.py solutions/hard.json
```

This is a warm-start result after multiple experiments, not a comparison with
SA under equal total work. The final stage used 30000 scored evaluations plus
one final verification. Neither -1.515 nor -1.55 has been reached. Subsequent
trials 905 through 908 did not improve the primary fitness. ALNS is a proposed
next experiment and is not implemented in this change.
