"""Check feasible sampling and incremental scores against the unchanged model."""
import random
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from mission.optimizer import evolve, initial_population, mutate_satellite, quality_key, resume_population
from mission.problems import load_problem
from mission.sampling import fit_state, local_state, sample_boundary_state, sample_state
from mission.search_state import FormationState


class GeometryTests(unittest.TestCase):
    def test_sampled_states_stay_feasible_in_all_epochs(self):
        for case in ['easy', 'medium', 'hard']:
            problem = load_problem(case)
            rng = random.Random(17)
            for _ in range(100):
                state = sample_state(problem, rng)
                for epoch, matrix in enumerate(problem.stms):
                    scale = 1 if epoch == 0 else problem.inflation_factor
                    self.assertTrue(all(abs(sum(a*b for a,b in zip(row, state))) < scale
                                        for row in matrix[:3]))
                self.assertTrue(all(abs(v) <= (1 if j < 3 else 10) for j,v in enumerate(state)))

    def test_initial_population_retains_all_satellites(self):
        for case in ['easy', 'medium', 'hard']:
            problem = load_problem(case)
            for x in initial_population(problem, 5, random.Random(7)):
                self.assertTrue(all(item['remaining'] == problem.n_sat
                                    for item in problem.observations(x)))

    def test_fit_preserves_feasible_state_and_shortens_extreme_state(self):
        problem = load_problem('hard')
        state = sample_state(problem, random.Random(3))
        self.assertEqual(fit_state(problem, state), state)
        projected = fit_state(problem, [v * 1000 for v in state])
        for epoch, matrix in enumerate(problem.stms):
            scale = 1 if epoch == 0 else problem.inflation_factor
            self.assertTrue(all(abs(sum(a*b for a,b in zip(row, projected))) < scale
                                for row in matrix[:3]))

    def test_easy_counting_bound(self):
        self.assertAlmostEqual(load_problem('easy').fitness_lower_bound(), -1 / 7)
        self.assertLess(load_problem('medium').fitness_lower_bound(), -1.5)

    def test_chord_and_boundary_samples_remain_feasible(self):
        for case in ['easy', 'medium', 'hard']:
            problem = load_problem(case)
            rng = random.Random(19)
            state = sample_state(problem, rng)
            for _ in range(30):
                moved = local_state(problem, state, rng, max_displacement=0.2, targeted=bool(_ % 2))
                self.assertIsNotNone(moved)
                for epoch, matrix in enumerate(problem.stms):
                    scale = 1 if epoch == 0 else problem.inflation_factor
                    for row in matrix[:3]:
                        self.assertLess(abs(sum(a*b for a,b in zip(row, moved))), scale)
                        delta = sum(a*(b-c) for a,b,c in zip(row, moved, state)) / scale
                        self.assertLessEqual(abs(delta), 0.2 + 1e-10)
                state = moved
            for _ in range(10):
                state = sample_boundary_state(problem, rng)
                for epoch, matrix in enumerate(problem.stms):
                    scale = 1 if epoch == 0 else problem.inflation_factor
                    self.assertTrue(all(abs(sum(a*b for a,b in zip(row, state))) < scale
                                        for row in matrix[:3]))


class IncrementalScoreTests(unittest.TestCase):
    def assert_matches(self, problem, state):
        self.assertEqual(state.key, quality_key(problem, state.x))
        self.assertEqual(state.key[0], problem.fitness(state.x)[0])
        for planes in state.counts:
            for counts in planes:
                self.assertTrue(all(value > 0 for value in counts.values()))

    def test_chained_moves_match_reference_and_preserve_parent(self):
        for case in ['easy', 'medium', 'hard']:
            problem = load_problem(case)
            rng = random.Random(5)
            x = initial_population(problem, 5, rng)[0]
            state = FormationState(problem, x)
            for _ in range(40):
                original_key = state.key
                original_counts = [[dict(counts) for counts in planes] for planes in state.counts]
                trial = mutate_satellite(problem, state.x, rng, max_displacement=0.5)
                updated = state.replace(trial)
                self.assert_matches(problem, updated)
                self.assertEqual(state.key, original_key)
                self.assertEqual(state.counts, original_counts)
                state = updated

    def test_overlaps_leaving_cube_and_multiple_satellites(self):
        problem = load_problem('easy')
        zero = [0.] * 30
        state = FormationState(problem, zero)
        self.assert_matches(problem, state)
        # The boundary is excluded at t=0; repeated projected cells are allowed.
        trial = zero[:]
        trial[0] = 1
        trial[15] = 10
        left = state.replace(trial)
        self.assert_matches(problem, left)
        self.assert_matches(problem, left.replace(zero))
        trial[1], trial[6] = -0.8, 0.8
        self.assert_matches(problem, state.replace(trial))
        self.assertIs(state.replace(zero), state)

    def test_all_satellites_outside_grid_then_one_returns(self):
        problem = load_problem('easy')
        x = [1.] * 15 + [10.] * 15
        state = FormationState(problem, x)
        self.assert_matches(problem, state)
        for component in range(6):
            x[component * problem.n_sat] = 0
        self.assert_matches(problem, state.replace(x))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.problem = load_problem('easy')
        self.population = initial_population(self.problem, 5, random.Random(31))
        x = min(self.population, key=lambda x: self.problem.fitness(x)[0])
        self.record = {'problem': 'tiny-observatory-in-dro', 'x': x,
                       'fitness': self.problem.fitness(x)[0]}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'incumbent.json'
        self.path.write_text(json.dumps(self.record), encoding='utf-8')

    def test_resume_retains_verified_incumbent_without_mutating_population(self):
        before = [row[:] for row in self.population]
        resumed = resume_population(self.problem, self.population, self.path, random.Random(4))
        self.assertEqual(resumed[0], self.record['x'])
        self.assertEqual(self.population, before)
        with contextlib.redirect_stdout(io.StringIO()):
            x, score, history, evaluations = evolve(self.problem, resumed, 3, random.Random(8))
        self.assertLessEqual(score, self.record['fitness'])
        self.assertEqual(self.problem.fitness(x)[0], score)
        self.assertEqual(evaluations, 20)

    def test_resume_rejects_wrong_case_and_incorrect_fitness(self):
        for field, value in [('problem', 'large-observatory-in-halo'), ('fitness', -2),
                             ('x', [0.] * 29), ('x', [True] * 30)]:
            record = dict(self.record)
            record[field] = value
            self.path.write_text(json.dumps(record), encoding='utf-8')
            with self.subTest(field=field), self.assertRaises(ValueError):
                resume_population(self.problem, self.population, self.path, random.Random(9))


if __name__ == '__main__':
    unittest.main()
