import csv
import os
from typing import List, Sequence

import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


@dataclass
class LearningRun:
    N: int
    evaluation_rewards: List[float]
    dt = 0.05
    T: float = field(init=False)

    def __post_init__(self):
        self.T = self.N * self.dt


def parse_csv(filepath: str) -> List[LearningRun]:
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=",")
        learning_runs = [LearningRun(int(row[0]), eval(row[1])) for row in reader]
    return learning_runs


def plot_data(learning_runs: Sequence[LearningRun]):
    cmap = plt.get_cmap('viridis')

    Ns = np.array([learning_run.N for learning_run in learning_runs])
    N_min = min(Ns)
    N_max = max(Ns)
    normalised_Ns = (Ns - N_min) / (N_max - N_min)

    # dict of N to color
    color_dict = {
        Ns[i]: cmap(normalised_N)
        for i, normalised_N in enumerate(normalised_Ns)
    }

    ax = plt.figure(figsize=(20, 10))

    for i, N in enumerate(Ns):
        plt.plot(learning_runs[i].evaluation_rewards, '-', color=color_dict[N], label=N)

    plt.grid()

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", 0.5, pad="3%")

    # plt.legend(loc=4)
    Ns_step: float = float(np.mean(np.ediff1d(Ns)))
    norm = Normalize(vmin=N_min, vmax=N_max)
    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    plt.colorbar(
        scalar_mappable,
        ticks=np.linspace(N_min, N_max, len(Ns)),
        boundaries=np.linspace(N_min - Ns_step / 2, N_max + Ns_step / 2, len(Ns) + 1),
        cax=cax
     )

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_filename = "data.csv"

    data_directory = "data"

    data_filepath = os.path.join(data_directory, data_filename)

    data = parse_csv(data_filepath)

    print(data)
    plot_data(data)
