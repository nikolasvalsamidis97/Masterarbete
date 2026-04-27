import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import mass_loss_rate_advanced_timestep_convergence as base


base.SYSTEM = base.advanced.AdvancedSystem(
    test_family="convergence",
    planet_key="super_earth_rocky",
    star_key="G1",
    distance_au=0.08,
)
base.advanced.SELECTED_ATOMIC_SPECIES = ["Na I"]


if __name__ == "__main__":
    base.main()
