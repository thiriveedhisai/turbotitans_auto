"""Plot a simulated run: course, path coloured by speed, events, outcome."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tt_course_sim.course import preview  # noqa: E402


def plot_run(course, traj, events, result, path):
    fig, ax = preview(path, course)
    if traj:
        arr = np.array([(r[1], r[2], r[3]) for r in traj])
        sc = ax.scatter(arr[:, 0], arr[:, 1], c=np.abs(arr[:, 2]), cmap='viridis', s=4,
                        vmin=0, vmax=max(1.0, float(np.abs(arr[:, 2]).max())))
        fig.colorbar(sc, ax=ax, shrink=0.6, label='speed (m/s)')
    style = {'COLLISION': ('rx', 10), 'FALL': ('kX', 16), 'STUCK': ('ms', 12),
             'TIMEOUT': ('ms', 12)}
    for e in events:
        mk, ms = style.get(e['kind'], ('k.', 6))
        ax.plot(e['x'], e['y'], mk, ms=ms, mew=2)
    ax.set_title(f"{result.get('outcome', '?').upper()}  t={result.get('time_s')} s  "
                 f"collisions={result.get('collisions')}  "
                 f"furthest={result.get('furthest_zone')}  rails={result.get('rails')}")
    fig.savefig(path, dpi=70, bbox_inches='tight')
    plt.close(fig)
