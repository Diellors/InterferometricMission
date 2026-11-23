# Adaptive Large Neighborhood Search

`mission/alns.py` implements an ALNS adaptation using Python's standard library.
The orbital model, grid and fitness normalization are unchanged. The algorithm
builds on the destroy/repair framework of Ropke and Pisinger:
https://doi.org/10.1287/trsc.1050.0135. Application to this mission is an
experimental adaptation, not a performance guarantee from that paper.

## Destroy and repair

Choose 4–8 satellites by default and remove their contributions from the baseline
counters. Removed satellites contribute neither pair baselines nor self-pairs.
These partial formations are internal search states, not valid saved solutions.

Three destroy heuristics compete: random selection, nearby projected cells at
the weakest observation, and low exclusive-baseline contribution there. Each
combines with a local or wide repair proposal distribution, giving six operators.
For each removed satellite, score its original state plus new feasible candidates
and insert the candidate with the best partial lexicographic key. Local repair
samples changed-cell intervals along feasible chords. Wide repair also samples
new interior and boundary states. The group is repaired in random order.

Complete formations are accepted with a simulated annealing criterion on the
primary fitness. Equal primary scores are accepted, permitting plateau walks.
The archive uses the primary score followed by total coverage as a tie breaker.
An accepted move can worsen the current state; the returned archive cannot worsen.

Operator rewards are 5 for archive improvement, 2 for current-key improvement,
0.5 for another accepted move and 0.05 for rejection. Every 25 complete iterations,
update used weights as 0.8*weight + 0.2*mean_reward, with a floor of 0.1. Unused
operators retain their weights. Restart from the archive every 150 iterations
and repeat geometric cooling from the initial to the final temperature.

## Budget and verification

`evaluations` counts the initial full score, each destruction score and every
partial insertion score. This is not the number of completed formations.
An unfinished final repair is discarded, but its scores remain counted.
The final independent full fitness verification is reported separately.
The CLI additionally verifies input, checkpoints and the saved result outside
the search budget. These checks must be included in wall-clock comparisons.

With E score budget, N satellites, M observations and B baseline support per
plane, hash-counter removal/insertion costs O(M*(N+B)) per satellite, plus copies
of states/counters. Destroy selection has additional cost, including O(M*N^2)
in the redundancy heuristic. Proposal geometry and sorting add runtime beyond
score counting. Budgets are finite; no optimum or strict improvement is guaranteed.

Tests compare partial counters with independently constructed directed baseline
sets, restore original formations, insert different formations, and check budgets,
adaptation, restarts, reproducibility, validation and archive preservation.

## Running

```bash
python scripts/run_alns.py --start solutions/hard.json --evaluations 30000 --seed 1500 --output results/alns/my-run
python scripts/verify_solution.py results/alns/my-run/result.json
```

The output directory must be new. `checkpoint.json` is updated every 25 complete
iterations; `result.json` is written after completion and verification.
Use `--min-destroy 2 --max-destroy 4 --pool-size 16` for a smaller neighborhood
with more candidate insertions. Default target is -1.55; smaller fitness is better.

Recorded experiments are summarized in `alns_experiments.csv`. They all start
from the published Hard incumbent -1.5139797739440808. Different temperatures
and pool sizes mean this is a parameter exploration, not a controlled comparison
with simulated annealing. The published incumbent is only replaced after a
verified improvement.

## Outcome

Seeds 1500–1502 did not improve the incumbent. Seed 1503, with groups of 2–4,
pool size 16 and initial temperature 0.001, reached -1.5223081499107673.
It used 99998 search scores, including incomplete final repair
work. In total these four runs used 189984 scores.
This crosses -1.515, but not -1.55. All final archives pass full verification;
the new record also matches at transition-matrix tolerance 1e-12.

The previous incumbent is in solutions/starts/hard_1_513979.json. Use that file
with seed 1503, --evaluations 100000 --pool-size 16 --min-destroy 2
--max-destroy 4 --initial-temperature 0.001 to repeat the winning final stage.
The record's original resumed_from path refers to solutions/hard.json before
its replacement. Warm-start discovery costs precede the reported ALNS budgets.

## Optional planar/vertical block proposals

`--block-probability` defaults to zero, preserving the original proposal stream.
For Medium's planar DRO reference, the computed transition matrices have exactly
zero cross-block entries between (x, y, vx, vy) and (z, vz). The optional operator
combines independently signed blocks from two donor satellites. This reuses
feasible projected planar/vertical profiles while changing their association.
Every proposed combination is still checked at all observation epochs.
This property concerns the linear model used here; it is not a statement about
arbitrary nonlinear orbits. Hard's coupled HALO matrices fail the separation
check and the runner rejects a nonzero block probability for that model.
