# Validation Mass Loss Timestep

Benchmarks timestep sensitivity for the advanced mass-loss calculation.

## Scripts

- `mass_loss_rate_advanced_timestep_convergence.py`: base convergence benchmark.
- `mass_loss_rate_advanced_timestep_convergence_inflated_hot_jupiter.py`: inflated hot-Jupiter variant.
- `mass_loss_rate_advanced_timestep_convergence_super_earth.py`: super-Earth variant.
- `mass_loss_rate_advanced_timestep_confidence_suite.py`: multi-case confidence suite.

## Run

From the repository root:

```bash
python "Projects/Validation Scripts/Validation_Mass_Loss_Timestep/mass_loss_rate_advanced_timestep_convergence.py"
```

Generated validation outputs are written under `results/`, which is ignored by Git.
