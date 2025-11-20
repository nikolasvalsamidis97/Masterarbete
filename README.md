# Photon Pressure Project

This repository contains a modular framework for computing photon pressure effects using atomic data, spectral information, and customizable run sequences. The project is organized into distinct folders, each encapsulating a specific functional part of the pipeline.

## Repository Structure

### **TS/**

Contains *theoretical spectra* retrieved from **SVO2**. These serve as input data for radiation field calculations and diagnostic comparisons.

### **project_classes/**

This directory includes Python modules that define the main physical and computational components of the project:

* **Atom construction** using parameters obtained from **NIST**.
* **Broadening profiles** (Lorentz, Gauss and Voigt).
* **Star models**, providing theoretical SED.
* **Photon pressure models**, computing the force excerted on a particle, coming from the SED.

### **project_func/**

Helper functions and utilities used across the codebase, including:

* General-purpose helper functions.
* Error-handling and error-propagation utilities.
* Additional shared computational routines.

### **project_run/**

A set of run-sequences enabling different workflows or simulation modes. These scripts orchestrate:

* Initialization of atomic and stellar models.
* Loading of spectral or observational data.
* Execution of photon-pressure computations.
* Optional plotting or data export.

### **Plots/**

Generated figures, diagnostics, and visualization outputs.

### **trash/**

Deprecated or temporary files not used in the current workflow.

---

## Getting Started

1. Clone the repository.
2. Review the modules in `project_classes` to understand the building blocks.
3. Use scripts in `project_run` to execute predefined simulation setups.
4. Add theoretical spectra to the `TS` folder if extending or modifying test cases.

---

## Requirements

Typical requirements include Python scientific libraries such as:

* NumPy
* SciPy
* Matplotlib
* Astropy

(Exact list may depend on the modules in `project_classes` and helper functions.)

---

## Purpose

This project provides a flexible and modular framework for studying photon pressure using physically grounded atomic data and spectrum-driven radiative interactions.

