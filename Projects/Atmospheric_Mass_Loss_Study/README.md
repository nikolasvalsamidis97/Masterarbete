# Atmospheric Mass Loss Study

Runs the advanced atmospheric mass-loss calculations and produces summary plots and tables.

## Scripts

- `Mass_loss_rate_advanced.py`: main mass-loss workflow. It supports `MLA_*` environment variables for choosing families, species, and output behavior.
- `make_finished_family_plots.py`: plotting helper for completed mass-loss families.
- `make_mass_loss_summary_table.py`: table helper for finished mass-loss outputs.

## Run

From the repository root:

```bash
python Projects/Atmospheric_Mass_Loss_Study/Mass_loss_rate_advanced.py
```

Generated outputs are written under `results/`, which is ignored by Git.
