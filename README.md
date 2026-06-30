# Radiation-Pressure-Driven Loss of Planetary Atmospheres

This repository contains the numerical framework and project scripts used for the master thesis **Radiation-pressure-driven loss of planetary atmospheres**. The code computes photon pressure, beta-coefficients, atmospheric beta=1 diagnostics, and idealized radiation-pressure-driven mass-loss estimates for atoms and molecules in stellar and planetary environments.

The thesis document is available on Overleaf: <https://www.diva-portal.org/smash/get/diva2:2067232/FULLTEXT01.pdf>

The repository is organized so that the reusable physical model lives in `project_classes/`, shared helpers live in `project_utils/`, reusable input definitions live in `Templates/`, and thesis-specific calculations live in `Projects/`.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `project_classes/` | Core physical objects used by all calculations. |
| `project_utils/` | Shared path helpers, plotting-data writers, and small utilities. |
| `Templates/` | Versioned atom, molecule, planet, star, system, and stellar-spectrum inputs. |
| `Projects/` | One project folder per thesis calculation or plot workflow. |
| `Projects/**/results/` | Local generated tables and plots. These folders are ignored by Git. |
| `exomol_data/` | Local ExoMol/RADIS cache. Ignored by Git. |
| `Illustrations/` | Local thesis illustration assets. Ignored by Git. |

Every project folder has its own `README.md` with the relevant script names and run command. Validation and regression checks are grouped under `Projects/Validation Scripts/`.

## Core Classes

| Class | File | Role |
| --- | --- | --- |
| `Atom` | [`project_classes/Atom.py`](project_classes/Atom.py) | Retrieves atomic line data from NIST, cleans transition tables, computes line-center cross-sections, and evaluates LTE partition functions for atomic species. |
| `Molecule` | [`project_classes/Molecule.py`](project_classes/Molecule.py) | Defines molecular species, wavelength ranges, masses, and ExoMol/HITRAN cache metadata used by molecule calculations. |
| `BroadeningProfile` | [`project_classes/BroadeningProfile.py`](project_classes/BroadeningProfile.py) | Builds Gaussian, Lorentzian, or Voigt line-broadening profiles for atomic transitions and applies them to atomic cross-sections. |
| `BroadeningProfileMolecule` | [`project_classes/BroadeningProfileMolecule.py`](project_classes/BroadeningProfileMolecule.py) | Builds molecular cross-section spectra on a shared wavelength grid, using cached ExoMol/HITRAN line data and temperature-dependent weights. |
| `Star` | [`project_classes/Star.py`](project_classes/Star.py) | Reads stellar spectra, converts wavelengths, applies rotational broadening, stores stellar parameters, and can scale spectra from photometric targets. |
| `PhotonPressure` | [`project_classes/PhotonPressure.py`](project_classes/PhotonPressure.py) | Combines a star with atomic or molecular cross-sections to calculate transmitted flux, photon pressure, and beta-values. |
| `Planet` | [`project_classes/Planet.py`](project_classes/Planet.py) | Represents an isothermal hydrostatic atmosphere and computes gravity, scale height, number density, and slant column density. |
| `PlanetarySystem` | [`project_classes/PlanetarySystem.py`](project_classes/PlanetarySystem.py) | Combines a planet and star with an orbital distance, then computes Hill, Roche-lobe, and gravity-equality radii and height grids. |

## Templates And Data

`Templates/Atoms/`, `Templates/Molecules/`, `Templates/Planets/`, `Templates/Stars/`, and `Templates/Systems/` define the reusable species, planet, star, and system inputs used by the project scripts.

`Templates/TS/Spectral_type/` contains the BT-NextGen spectral type grid used by the stellar templates. These spectra are tracked in Git so that a fresh clone can run the available template-based scripts without a separate stellar-spectrum download.

`Templates/TS/Beta pic Spectra/` contains the beta Pictoris spectra used for the Fernandez et al. comparison.

## Python Dependencies

Starting from a clean Python installation, the thesis code uses the following external packages:

| Package | Used for |
| --- | --- |
| `numpy` | Arrays, numerical grids, interpolation, and vectorized calculations. |
| `pandas` | Tables, CSV/TXT processing, and generated result summaries. |
| `scipy` | Numerical integration and special functions used in line profiles. |
| `matplotlib` | Plot generation. |
| `astropy` | Units, constants, tables, spectra I/O, and astronomy utilities. |
| `astroquery` | NIST atomic line-data queries. |
| `synphot` | Synthetic photometry and stellar-spectrum scaling. |
| `requests` | HTTP requests for SVO filter data and related remote resources. |
| `periodictable` | Atomic masses. |
| `molmass` | Molecular masses. |
| `radis` | ExoMol/HITRAN molecular line-data access. |
| `tables` | HDF5/PyTables support for cached molecular data. |
| `pytictoc` | Timing utility used by one validation/stress-test script. |

Install all external dependencies with:

```bash
python -m pip install numpy pandas scipy matplotlib astropy astroquery synphot requests periodictable molmass radis tables pytictoc
```

`pip` automatically skips reinstalling packages that are already available in the active Python environment.

## Running Scripts

Run scripts from the repository root so relative paths resolve consistently:

```bash
python Projects/Fernandez_Beta_Comparison/comparison_Fernandez.py
```

Generated outputs are written below each project's local `results/` folder, for example:

```text
Projects/Fernandez_Beta_Comparison/results/Plots/
Projects/Atmospheric_Mass_Loss_Study/results/Tables/
```

These result folders are ignored by Git. Re-run the relevant project script to regenerate figures or tables.

Some workflows are computationally heavy, especially the full beta grids and mass-loss calculations. For those projects, check the local project README and script-level environment variables before running the full workflow.

## Thesis Figure And Script Index

The table below links the thesis figures and plots to the project scripts that generate them. Figures that are diagrams or local illustrations rather than Python-generated plots are marked as thesis assets.

| Thesis item | Content | Source script or asset |
| --- | --- | --- |
| Fig. 2.1 | Atmospheric escape mechanisms | Thesis illustration asset in `Illustrations/`, not a project script. |
| Fig. 3.1 | Gaussian, Lorentzian, and Voigt profile comparison | [`Projects/Broadening_Profile_Plots/BroadeningProfilesPlots.py`](Projects/Broadening_Profile_Plots/BroadeningProfilesPlots.py) |
| Fig. 3.2 | Voigt-broadened normalized intensity as a function of column density | [`Projects/Broadening_Profile_Plots/OpticalDepth_Ncol.py`](Projects/Broadening_Profile_Plots/OpticalDepth_Ncol.py) |
| Fig. 3.3 | Photon pressure per absorber versus column density for Na I | [`Projects/Photon_Pressure_Column_Plot/photon_pressure_vs_column_density.py`](Projects/Photon_Pressure_Column_Plot/photon_pressure_vs_column_density.py) |
| Fig. 3.4 | Rotational broadening of the stellar Na I line | [`Projects/Broadening_Profile_Plots/RotationalBroadeningPlots.py`](Projects/Broadening_Profile_Plots/RotationalBroadeningPlots.py) |
| Fig. 3.5 | Spherically symmetric atmosphere geometry | Thesis illustration asset in `Illustrations/`, not a project script. |
| Fig. 3.6 | Mass-loss estimate geometry | Thesis illustration asset in `Illustrations/`, not a project script. |
| Fig. 4.1 | Atom and Molecule class diagrams | Thesis diagram based on [`project_classes/Atom.py`](project_classes/Atom.py) and [`project_classes/Molecule.py`](project_classes/Molecule.py). |
| Fig. 4.2 | BroadeningProfile class diagram | Thesis diagram based on [`project_classes/BroadeningProfile.py`](project_classes/BroadeningProfile.py) and [`project_classes/BroadeningProfileMolecule.py`](project_classes/BroadeningProfileMolecule.py). |
| Fig. 4.3 | Star class diagram | Thesis diagram based on [`project_classes/Star.py`](project_classes/Star.py). |
| Fig. 4.4 | PhotonPressure class diagram | Thesis diagram based on [`project_classes/PhotonPressure.py`](project_classes/PhotonPressure.py). |
| Fig. 4.5 | Planet class diagram | Thesis diagram based on [`project_classes/Planet.py`](project_classes/Planet.py). |
| Fig. 4.6 | PlanetarySystem class diagram | Thesis diagram based on [`project_classes/PlanetarySystem.py`](project_classes/PlanetarySystem.py). |
| Fig. 5.1 | BT-NextGen and Fernandez beta Pictoris spectra | [`Projects/BTNextGen_Spectrum_Plot/plotspectra.py`](Projects/BTNextGen_Spectrum_Plot/plotspectra.py) and [`Projects/Spectra_Fernandez_Comparison/spectra_comparison.py`](Projects/Spectra_Fernandez_Comparison/spectra_comparison.py) |
| Fig. 5.2 | Full beta comparison against Fernandez et al. (2006) | [`Projects/Fernandez_Beta_Comparison/comparison_Fernandez.py`](Projects/Fernandez_Beta_Comparison/comparison_Fernandez.py) |
| Fig. 5.3 | Zoomed beta comparison for beta greater than 1 | [`Projects/Fernandez_Beta_Comparison/comparison_Fernandez.py`](Projects/Fernandez_Beta_Comparison/comparison_Fernandez.py) |
| Fig. 5.4 | Beta versus column density for Na I and selected stellar models | [`Projects/Beta_Ncol_Depth_Study/simple_vs_depth.py`](Projects/Beta_Ncol_Depth_Study/simple_vs_depth.py) |
| Fig. 5.5 | Effect of line and rotational broadening on beta versus column density | [`Projects/Beta_Ncol_Broadening_Study/effect_of_broad.py`](Projects/Beta_Ncol_Broadening_Study/effect_of_broad.py) |
| Fig. 5.6 | Boltzmann-weighted Fe, Fe II, and Fe III beta curves | [`Projects/Beta_Ncol_Weighting_Study/effect_of_weights.py`](Projects/Beta_Ncol_Weighting_Study/effect_of_weights.py) and [`Projects/Beta_Ncol_Weighting_Study/plot_weights_study.py`](Projects/Beta_Ncol_Weighting_Study/plot_weights_study.py) |
| Fig. 5.7 | Atomic beta heatmap versus excitation temperature | [`Projects/Beta_Tgas_Excitation_Study/Beta_Tgas.py`](Projects/Beta_Tgas_Excitation_Study/Beta_Tgas.py) and [`Projects/Beta_Tgas_Excitation_Study/plot_Beta_Tgas.py`](Projects/Beta_Tgas_Excitation_Study/plot_Beta_Tgas.py) |
| Fig. 5.8 | Molecular beta heatmap versus excitation temperature | [`Projects/Beta_Tgas_Excitation_Study/Beta_Tgas.py`](Projects/Beta_Tgas_Excitation_Study/Beta_Tgas.py) and [`Projects/Beta_Tgas_Excitation_Study/plot_Beta_Tgas.py`](Projects/Beta_Tgas_Excitation_Study/plot_Beta_Tgas.py) |
| Fig. 5.9 | Critical height `r_beta=1 / R_p` for four example planets | [`Projects/Atmospheric_Teff_Beta1_Distance_Study/Teff_planet_beta1_dist.py`](Projects/Atmospheric_Teff_Beta1_Distance_Study/Teff_planet_beta1_dist.py) and [`Projects/Atmospheric_Teff_Beta1_Distance_Study/make_four_example_plot.py`](Projects/Atmospheric_Teff_Beta1_Distance_Study/make_four_example_plot.py) |
| Fig. 5.10 | Strongest critical-height responders for the four example planets | [`Projects/Atmospheric_Teff_Beta1_Distance_Study/make_four_strongest_responders_plot.py`](Projects/Atmospheric_Teff_Beta1_Distance_Study/make_four_strongest_responders_plot.py) |
| Fig. 5.11 | Threshold stellar temperature and orbital distance for `r_beta=1 <= r_exo` | [`Projects/Atmospheric_Rp_Exobase_Beta1_Study/R_p_exo_beta1.py`](Projects/Atmospheric_Rp_Exobase_Beta1_Study/R_p_exo_beta1.py) and [`Projects/Atmospheric_Rp_Exobase_Beta1_Study/make_R_p_exo_beta1_summary.py`](Projects/Atmospheric_Rp_Exobase_Beta1_Study/make_R_p_exo_beta1_summary.py) |
| Fig. 5.12 | Mass-loss rate versus orbital distance | [`Projects/Atmospheric_Mass_Loss_Study/Mass_loss_rate_advanced.py`](Projects/Atmospheric_Mass_Loss_Study/Mass_loss_rate_advanced.py) and [`Projects/Atmospheric_Mass_Loss_Study/make_finished_family_plots.py`](Projects/Atmospheric_Mass_Loss_Study/make_finished_family_plots.py) |
| Fig. 5.13 | Mass-loss rate versus surface gravity | [`Projects/Atmospheric_Mass_Loss_Study/Mass_loss_rate_advanced.py`](Projects/Atmospheric_Mass_Loss_Study/Mass_loss_rate_advanced.py) and [`Projects/Atmospheric_Mass_Loss_Study/make_finished_family_plots.py`](Projects/Atmospheric_Mass_Loss_Study/make_finished_family_plots.py) |
| Fig. 5.14 | Mass-loss rate versus reference pressure `P0` | [`Projects/Atmospheric_Mass_Loss_Study/Mass_loss_rate_advanced.py`](Projects/Atmospheric_Mass_Loss_Study/Mass_loss_rate_advanced.py) and [`Projects/Atmospheric_Mass_Loss_Study/make_finished_family_plots.py`](Projects/Atmospheric_Mass_Loss_Study/make_finished_family_plots.py) |
| Fig. 5.15 | Solar-System analogue mass-loss rates | [`Projects/Atmospheric_Mass_Loss_Study/Mass_loss_rate_advanced.py`](Projects/Atmospheric_Mass_Loss_Study/Mass_loss_rate_advanced.py) and [`Projects/Atmospheric_Mass_Loss_Study/make_finished_family_plots.py`](Projects/Atmospheric_Mass_Loss_Study/make_finished_family_plots.py) |
| Fig. 5.16 | Real reference-system mass-loss rates | [`Projects/Atmospheric_Mass_Loss_Study/Mass_loss_rate_advanced.py`](Projects/Atmospheric_Mass_Loss_Study/Mass_loss_rate_advanced.py) and [`Projects/Atmospheric_Mass_Loss_Study/make_finished_family_plots.py`](Projects/Atmospheric_Mass_Loss_Study/make_finished_family_plots.py) |

## Table-Oriented Projects

Several projects generate thesis tables rather than figures:

| Project | Output |
| --- | --- |
| [`Projects/Planet_Template_Tables/`](Projects/Planet_Template_Tables/) | Planet template property and composition tables. |
| [`Projects/Real_Mass_Loss_System_Table/`](Projects/Real_Mass_Loss_System_Table/) | Real reference-system table for the mass-loss study. |
| [`Projects/Beta_Teff_Template_Table/`](Projects/Beta_Teff_Template_Table/) | Beta as a function of stellar effective temperature. |
| [`Projects/Beta_Bigtable_Study/`](Projects/Beta_Bigtable_Study/) | Large beta tables for atoms and molecules. |
| [`Projects/Fernandez_Beta_Table/`](Projects/Fernandez_Beta_Table/) | This-work beta values used in the Fernandez comparison. |
| [`Projects/Atmospheric_Exobase_Calculation/`](Projects/Atmospheric_Exobase_Calculation/) | Exobase-height table used by atmospheric beta=1 and mass-loss workflows. |
| [`Projects/Atmospheric_Mass_Loss_Study/make_mass_loss_summary_table.py`](Projects/Atmospheric_Mass_Loss_Study/make_mass_loss_summary_table.py) | Summary tables for mass-loss families and real systems. |

## Git Notes

The repository tracks source code, templates, and the stellar spectra needed by the available scripts. Generated outputs stay local:

```gitignore
Projects/**/results/
Illustrations/
exomol_data/
```

After restructuring or regenerating tracked inputs, stage the new structure with:

```bash
git add -A
git commit -m "Restructure thesis project layout"
git push
```
