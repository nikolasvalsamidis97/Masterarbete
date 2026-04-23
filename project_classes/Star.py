from project_func.errors import _not_quantity
import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.io import ascii
import re
from synphot import SpectralElement, SourceSpectrum, Observation
from synphot.models import Empirical1D
import requests
from io import BytesIO
from astropy.table import Table
# Optional Vega support (only needed if you ever want magsys="vegamag")
from synphot.config import conf as synconf




class Star:
    
    def __init__(self, path: str, radius, mass, vsini, epsilon):
        """
        Creates a star object

        ** Inputs **
        path:       str               Filepath for theoretical spectra
        distance:   Quantity          Distance to star
        radius:     Quantity          Radius of star
        mass:       Quantity          Mass of star
        vsini:      Quantity          Projected rotational velocity
        epsilon:    Quantity          Limb darkening

        ** Functions **
        print_header:     Prints the header of the model

        ** Callables **
        header:           Header of the model
        """
        self.path = path
        self.radius = radius.to(u.m) if isinstance(radius, u.Quantity) else _not_quantity("radius")
        self.mass = mass.to(u.kg) if isinstance(mass, u.Quantity) else _not_quantity("mass")
        self.vsini = vsini.to(u.km / u.s) if isinstance(vsini, u.Quantity) else _not_quantity("vsini")
        self.epsilon = epsilon.to(u.dimensionless_unscaled) if isinstance(epsilon, u.Quantity) else _not_quantity("epsilon")

        self.lam_star, self.flux_star_rot, self.flux_star_unrot = self.read_Spectra()
        self.header = self.set_header(path)

    
    def read_Spectra(self):
        """
        Reads a spectra from a file, rotationally broadens it and returns the flux in vacuum
        Also creates a header for the star

        ** Returns **
        lam:          The lambda array from the stellar spectra       [Å]               [lambda, ]
        flux_rot      Rotatationally broadened spectra                [Flux units]      [F, ]
        flux_unrot    Original spectra                                [Flux units]      [F_orig, ]
        """
        tab = ascii.read(
          self.path,
          format="basic",
          comment="#",
          names=("WAVELENGTH", "FLUX"),
          guess=False
        )

        lam = tab["WAVELENGTH"].value          #u.AA
        flux = tab['FLUX'].value              #(u.erg/u.s/(u.cm**2)/u.AA)
        flux = flux * (u.erg/u.s/(u.cm**2)/u.AA)
        n = self.air_to_vacuum(lam)
        lam = lam * n * u.AA

        mask = np.isfinite(lam) & (lam > 0)
        lam = lam[mask]
        flux = flux[mask]
        sort = np.argsort(lam)
        lam = lam[sort]
        flux_unrot = flux[sort]
        
        lam, flux_rot = self.rot_kernel(lam, flux)

        return lam, flux_rot, flux_unrot
    
    def set_header(self, path):
        hdr = {
            "model": {"value": None, "unit": None},
            "teff":  {"value": None, "unit": "K"},
            "logg":  {"value": None, "unit": "log10(cm/s2)"},
            "meta":  {"value": None, "unit": "dex"},
            "alpha": {"value": None, "unit": "dex"},
        }

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.lstrip().startswith("#"):
                    break

                s = line.lstrip("#").strip()
                if not s:
                    continue

                if hdr["model"]["value"] is None and "=" not in s and s.startswith("BT-"):
                    hdr["model"]["value"] = s
                    continue

                m = re.match(r"^(teff|logg|meta|alpha)\s*=\s*([+-]?\d+(?:\.\d+)?)", s, re.I)
                if m:
                    key = m.group(1).lower()
                    hdr[key]["value"] = float(m.group(2))
        return hdr  

    def print_header(self):
        for k, d in self.header.items():
            if isinstance(d, dict):
                unit = d.get("unit")
                val  = d.get("value")
                print(f"{k:6s}: {val}" if not unit else f"{k:6s}: {val} {unit}")
            else:
                print(f"{k:6s}: {d}")

    def air_to_vacuum(self, lam_air_A):
        s2 = (1e4/lam_air_A)**2
        n_minus_1 = 1e-8*(8342.13 + 2406030/(130 - s2) + 15997/(38.9 - s2))
        n = 1 + n_minus_1
        return n
    
    def rot_kernel(self, lam, flux):
        vsini = self.vsini.to_value(u.km/u.s)
        eps = self.epsilon.to_value(u.dimensionless_unscaled)
        c_kms = const.c.to_value(u.km / u.s)
        
        lam_star = lam.to_value(u.AA)
        flux_star = flux.to_value(u.erg/u.s/(u.cm**2)/u.AA)

        lnlam_star = np.log(lam_star)
        dlnlam_star = np.median(np.diff(lnlam_star)) / 4      # quadruple the resolution of the stars spectra
        N = int(np.floor((lnlam_star[-1] - lnlam_star[0]) / dlnlam_star)) + 1
        lnlam = lnlam_star[0] + dlnlam_star * np.arange(N)

        flux_star_interp = np.interp(lnlam, lnlam_star, flux_star)

        dv = (c_kms * dlnlam_star)
        half = int(np.ceil(vsini / dv))
        dv_axis = (np.arange(-half, half+1) * dv)              # The grid of dv the kernel will use in km/s
        x = dv_axis / vsini
        g = np.zeros_like(x)

        m = np.abs(x) <= 1
        xm = x[m]

        g[m] = (2*(1-eps)*np.sqrt(1-xm**2) + 0.5*np.pi*eps*(1-xm**2)) / (np.pi*vsini*(1-eps/3))

        g_weights = g * dv      
        g_weights /= g_weights.sum()

        flux_rot_interp = np.convolve(flux_star_interp, g_weights, mode="same")
        flux_rot_star = np.interp(lnlam_star, lnlam, flux_rot_interp)

        return lam_star * u.AA, flux_rot_star * u.erg/u.s/(u.cm**2)/u.AA

    def get_bandpass_svo(self, photcalid: str):
        """
        photcalid example inputs:
        "2MASS/2MASS.H/AB"
        "2MASS/2MASS.H/Vega"
        """
        
        url = "https://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?PhotCalID=" + photcalid
        vot_bytes = requests.get(url).content

        tab = Table.read(BytesIO(vot_bytes), format="votable")

        bp_wave = np.array(tab["Wavelength"]) * u.AA
        bp_thru = np.array(tab["Transmission"])

        band = SpectralElement(Empirical1D, points=bp_wave, lookup_table=bp_thru)
        lam_pivot = band.pivot().to(u.AA)
        return band, lam_pivot

    def synthetic_mag(self, photcalid: str, distance: u.Quantity, magsys: str="vegamag", use_rot: bool=True):
        """
        Returns synthetic magnitude of this stars spectrum using given SVO PhotCalID

        distance: Distance to the star from earth
        magsys: "abmag" or "vegamag"
        """

        R_star = self.radius.to_value(u.m)
        d_earth_star = distance.to_value(u.m)

        band, lam_pivot = self.get_bandpass_svo(photcalid) # Bandpass/filter curve

        lam = self.lam_star
        flux = self.flux_star_rot if use_rot else self.flux_star_unrot

        flux_scaled = flux * (R_star / d_earth_star)**2

        # SourceSpectrum: Creates a source spectrum with values given to it
        # Empirical1D: “don't use a blackbody / power law / analytic function — just use these points and interpolate.”
        # "Taper": Use when bandpass extends beyond spectra
        source = SourceSpectrum(Empirical1D, points=lam, lookup_table=flux_scaled)
        obs = Observation(source, band, force="taper")

        if magsys.lower() == "vegamag":
            # synphot needs an explicit Vega spectrum reference
            synconf.vega_file = "https://ssb.stsci.edu/trds/calspec/alpha_lyr_stis_011.fits"
            vega = SourceSpectrum.from_vega()
            return obs.effstim("vegamag", vegaspec=vega), lam_pivot
        
        # effstim: Returns synthetic observed spectra
        return obs.effstim(magsys.lower()), lam_pivot

    def scale_factors_from_targets(self, 
                                   targets: dict, 
                                   distance: u.Quantity,
                                   magsys: dict, 
                                   photcalid_map: dict=None,
                                   use_rot: bool=True,
                                   ):
        """
        Computes scale factors from synthetic magnitudes from current star and given target magnitude. 
        The mean of the scale factors can be used to scale the stellar radius according to
        R_new = sqrt(k_mean) * R_old
        
        photcalid example inputs: "2MASS/2MASS.H/AB" or "2MASS/2MASS.H/Vega"
        distance: Distance to the star from earth
        m_target: target magnitude (from comparison)
        magsys: "abmag" or "vegamag"

        Formula to calc k:
        k = 10^(-0.4 (m_target - m_synthetic))
        """

        k_vals = {}
        lam_pivots = {}
        for survey, filters in targets.items():                       # "2MASS", "J"
            for band, m_target in filters.items():                      # "J", 10.2 
                photcalid = photcalid_map[survey][band]                   # photcalid_map["2MASS"]["J"] -> "2MASS/2MASS.J/Vega"
                ms = magsys[survey]                                       # magsys["2MASS"] -> "vegamag"
                m_synthetic, lam_pivot = self.synthetic_mag(photcalid, distance, magsys=ms, use_rot=use_rot)   # synthetic_mag("2MASS/2MASS.J/Vega", d_to_object, magsys="vegamag", use_rot=True) -> m_synthetic
                lam_pivots[f"{survey}_{band}"] = lam_pivot
                k_vals[f"{survey}_{band}"] = 10**(-0.4 * (m_target - m_synthetic.value))   # k = 10^(-0.4 (m_target - m_synthetic))
        
        k_mean = np.mean(list(k_vals.values()))

        self.old_radius = self.radius
        self.radius = self.radius * np.sqrt(k_mean)

        return k_vals, lam_pivots


    def convert_from_log10(self):
        # For already rotated, vacuum spectra in log 10 flux units
        # If scaling (alpha) provided plug in
        # If distance for calibration provided, plug in to convert to surface flux
        lam = self.lam_star
        flux_log10 = self.flux_star_unrot
        flux = 10**flux_log10.to_value(u.erg/u.s/(u.cm**2)/u.AA) * (u.erg/u.s/(u.cm**2)/u.AA)
        n = self.air_to_vacuum(lam.to_value(u.AA))   # reverting the vacuum to air wavelengths
        lam = lam / n
        self.lam_star = lam

        self.flux_star_unrot = flux
        self.flux_star_rot = self.flux_star_unrot                    # since already broadened
        return lam, flux
    
    def get_mag(self, band: str, distance: u.Quantity, system: str="vegamag", use_rot: bool=True):
        """
        band: SVO PhotCalID, e.g. "TYCHO/TYCHO.B/Vega" or "Generic/Johnson.B/Vega"
        distance: distance to the star
        system: "vegamag" or "abmag"
        """
        m, _ = self.synthetic_mag(band, distance, magsys=system, use_rot=use_rot)
        return m
