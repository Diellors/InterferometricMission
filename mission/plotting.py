"""Formation, baseline and convergence figures; matplotlib is optional."""
import math


def plot_formation(problem, x):
    """Return a figure without showing, saving or closing other figures."""
    observations = problem.observations(x)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2 * len(observations), 3, figsize=(12, 16), squeeze=False)
    names = ['XY', 'XZ', 'YZ']
    size = problem.grid_size
    worst = min(item['total'] for item in observations)
    fig.suptitle(f'{problem.case.upper()} | {problem.n_sat} satellites | fitness = {-worst:.6f}',
                 fontsize=15)
    for epoch, observation in enumerate(observations):
        for plane, data in enumerate(observation['planes']):
            horizontal, vertical = names[plane]
            ax = axes[2 * epoch][plane]
            if data['cells']:
                a, b = zip(*sorted(data['cells']))
                ax.scatter(a, b, marker='s', s=45, color='tomato', label='Occupied cells')
            center = size // 2
            ax.scatter([center], [center], marker='+', s=90, color='black', label='Mothership')
            ax.set(xlim=(-0.5, size - 0.5), ylim=(-0.5, size - 0.5),
                   xlabel=f'{horizontal} cell', ylabel=f'{vertical} cell',
                   title=f't={epoch}T | {names[plane]} | '
                         f'{len(data["cells"])} cells, {observation["remaining"]} retained')
            ax.set_xticks(range(0, size, max(1, size // 5)))
            ax.set_yticks(range(0, size, max(1, size // 5)))
            ax.grid(alpha=0.2)
            ax.set_aspect('equal')
            if epoch == 0 and plane == 0:
                ax.legend(fontsize=7, loc='upper left')
            ax = axes[2 * epoch + 1][plane]
            if data['baselines']:
                a, b = zip(*sorted(data['baselines']))
                ax.scatter(a, b, marker='s', s=9, color='steelblue')
            edge = size - 0.5
            ax.set(xlim=(-edge, edge), ylim=(-edge, edge),
                   xlabel=f'Delta {horizontal} cells', ylabel=f'Delta {vertical} cells',
                   title=f'{len(data["baselines"])} baselines | fill = {data["fill"]:.6f}')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)
            ax.set_aspect('equal')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def validate_history(history):
    """Accept a best-so-far series including generation zero."""
    if not isinstance(history, (list, tuple)) or not history:
        raise ValueError('history must be a nonempty list of fitness values')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in history):
        raise ValueError('history must contain only finite numbers')
    if any(b > a + 1e-12 for a, b in zip(history, history[1:])):
        raise ValueError('Best-so-far fitness must not increase')
    return list(history)


def plot_convergence(history, *, title='Optimization convergence'):
    """Plot minimized fitness on its original scale; lower is better."""
    values = validate_history(history)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    generations = list(range(len(values)))
    ax.step(generations, values, where='post', color='steelblue', linewidth=2,
            label='Best fitness')
    ax.scatter([0, generations[-1]], [values[0], values[-1]], color='tomato', zorder=3)
    ax.set(title=title, xlabel='Generation (0 = initial population)',
           ylabel='Fitness (lower is better)', xlim=(-0.1, max(1, generations[-1]) + 0.1))
    ax.text(0.98, 0.95, f'Initial: {values[0]:.6f}\nFinal: {values[-1]:.6f}\n'
            f'Improvement: {values[0] - values[-1]:.6f}', transform=ax.transAxes,
            ha='right', va='top', bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'gray'})
    ax.grid(alpha=0.25)
    ax.legend(loc='lower left')
    fig.tight_layout()
    return fig
