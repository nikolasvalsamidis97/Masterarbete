# Photon Pressure Project

This repository contains a modular framework for computing photon-pressure effects using atomic data, spectral information, and project-specific run scripts.

## Repository Structure

### `project_classes/`

Core physical and computational classes: atoms, molecules, broadening profiles, stars, planets, planetary systems, and photon-pressure calculations.

### `project_utils/`

Small shared utilities used by the core classes and project scripts.

### `Templates/`

Versioned templates and reference inputs. This includes atom, molecule, planet, star, and system templates, plus `Templates/TS/` for theoretical spectra.

### `Projects/`

Project-specific scripts. Each folder is a separate project with one main calculation script, plus any closely associated plotting or table helpers. Validation projects live under `Projects/Validation Scripts/`. Every project folder has a local `README.md` with its purpose and run command, plus a local `results/` folder with `Tables/` and `Plots` subfolders.

The `results/` folders are ignored by Git, so generated tables, plots, caches, and intermediate outputs stay local.

### `exomol_data/`

Local ExoMol/RADIS data cache. This is ignored by Git.

### `Illustrations/`

Local thesis illustrations. This is ignored by Git.

## Getting Started

1. Clone the repository.
2. Review `project_classes/` for the reusable model components.
3. Use scripts in `Projects/<project_name>/` to run calculations, plotting, and table generation.
4. Put reusable template inputs under `Templates/`.
