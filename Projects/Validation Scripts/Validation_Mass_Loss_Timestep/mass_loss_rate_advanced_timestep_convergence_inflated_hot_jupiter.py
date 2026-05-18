import importlib.util
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))


BASE_SCRIPT_PATH = pathlib.Path(__file__).with_name("mass_loss_rate_advanced_timestep_convergence.py")
BASE_SPEC = importlib.util.spec_from_file_location(
    "mass_loss_rate_advanced_timestep_convergence_base",
    BASE_SCRIPT_PATH,
)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise ImportError(f"Could not load base convergence script from {BASE_SCRIPT_PATH}")
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)


base.SYSTEM = base.advanced.AdvancedSystem(
    test_family="convergence",
    planet_key="inflated_hot_jupiter",
    star_key="A0",
    distance_au=0.05,
)
base.advanced.SELECTED_ATOMIC_SPECIES = ["Na I"]


if __name__ == "__main__":
    base.main()
