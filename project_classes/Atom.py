import numpy as np
import periodictable as pt
from astropy import units as u
from astropy import constants as const
from project_func.errors import _not_quantity
from astroquery.nist import Nist
import pandas as pd

class Atom:
    
    def __init__(self, species: str, lam_min, lam_max, A_ul_min = 0 / u.s):
        """
        species:        Chemical formula of molecule
        lam_min:        Minimum data wavelength                     Quantity
        lam_max:        Maximum data wavelength                     Quantity
        A_ul_min:       Minimum value of spontaneous deexitation    Quantity
        """
        self.species = species
        self.lam_min = lam_min.to(u.AA) if isinstance(lam_min, u.Quantity) else _not_quantity("lam_min")
        self.lam_max = lam_max.to(u.AA) if isinstance(lam_max, u.Quantity) else _not_quantity("lam_max")
        self.A_ul_min = A_ul_min.to(1/u.s) if isinstance(A_ul_min, u.Quantity) else _not_quantity("A_ul_min")

        self.data = self.set_Nist_Data(self.species, self.lam_min, self.lam_max, self.A_ul_min)
        self.mass = pt.elements.symbol(self.species.split()[0]).mass * u.u                          # Removing "I" from the species name for molmass formula parsing. This is because molmass does not recognize the ionization state in the chemical formula.
        self.A_ul, self.A_ul_err, self.lam0, self.g_u, self.g_l, self.E_u, self.E_l, self.J_l, self.fik = self.pandas_to_numpy(self.data)
        self.sig_0, self.sig_0_err = self.calc_central_crossection()
        self._unique_states = self._build_unique_states()

    def get_Name(self):
        print(self.species)

    def pandas_to_numpy(self, data):
        """
        Numpy arrays with dimensions (N_lines, None) ex. (16,)
        """
        Aul = pd.to_numeric(data['A_ul']).to_numpy().reshape(-1, 1) / u.s
        Aul_err = pd.to_numeric(data['Acc'], errors='coerce').fillna(0.0).to_numpy().reshape(-1, 1) / u.s
        lam0 = pd.to_numeric(data['lam_obs']).to_numpy().reshape(-1, 1) * u.AA
        gu = pd.to_numeric(data['g_u']).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled
        gl = pd.to_numeric(data['g_l']).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled
        Eu = pd.to_numeric(data['E_u']).to_numpy().reshape(-1, 1) * u.eV
        El = pd.to_numeric(data['E_l']).to_numpy().reshape(-1, 1) * u.eV
        J_l = pd.to_numeric(data['J_l']).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled
        fik = pd.to_numeric(data['fik']).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled

        return Aul, Aul_err, lam0, gu, gl, Eu, El, J_l, fik

    def set_Nist_Data(self,
                      species,
                      wav_min, 
                      wav_max, 
                      A_ul):
        """
        wav_min:    Minimum wavelength as float in Angstrom
        wav_max:    Maximum wavelength as float in Angstrom
        A_ul:       Minimum A_ul as float in 1/s
        """

        tab = Nist.query(wav_min, 
                        wav_max, 
                        linename = species, 
                        energy_level_unit = 'eV',
                        wavelength_type='vacuum', 
                        output_order='wavelength')
        
        df = tab.to_pandas()
        # Ion identification
        spec = df.get('Spectrum')

        def text_series(column_name: str) -> pd.Series:
            series = df.get(column_name)
            if series is None:
                return pd.Series([""] * len(df), index=df.index, dtype="object")
            return series.fillna("").astype(str)

        def split_two_columns(series: pd.Series, delimiter: str) -> pd.DataFrame:
            parts = series.str.split(delimiter, n=1, expand=True)
            return parts.reindex(columns=[0, 1], fill_value="")

        # Wavelength in nm. If observed wavelength is missing i will use Ritz
        lam_obs = pd.to_numeric(df['Observed'], errors='coerce').astype(float)
        lam_ritz = pd.to_numeric(df['Ritz'], errors='coerce').astype(float)
        lam_obs = lam_obs.fillna(lam_ritz)

        A_ul = pd.to_numeric(df['Aki'], errors='coerce')

        Acc = df['Acc.']
        ACC_FRAC = {                     # Map onto Acc-code. Source: 'https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html#OUTACC' Search for "estimated accuracy"
        'AAA': 0.003,
        'AA': 0.01,
        'A+': 0.02,
        'A': 0.03,
        'B+': 0.07,
        'B': 0.10,
        'C+': 0.18,
        'C': 0.25,
        'D+': 0.40,
        'D': 0.50,
        'E': 0.50
        }
        Acc = (Acc.map(ACC_FRAC) * A_ul).fillna(0.0)
        

        eiek = split_two_columns(text_series('Ei           Ek'), '-')
        ei = eiek[0].str.strip(' []?') # lower energy (Ei)
        ek = eiek[1].str.strip(' []?') # upper energy (Ek)
        Ei = pd.to_numeric(ei, errors='coerce')
        Ek = pd.to_numeric(ek, errors='coerce')

        gigk = split_two_columns(text_series('gi   gk'), '-')
        gi = gigk[0].str.strip()
        gk = gigk[1].str.strip()
        Gi = pd.to_numeric(gi, errors='coerce')
        Gk = pd.to_numeric(gk, errors='coerce')

        lower_parts = text_series('Lower level').str.split('|', n=2, expand=True)
        lower_parts = lower_parts.reindex(columns=[0, 1, 2], fill_value="")
        upper_parts = text_series('Upper level').str.split('|', n=2, expand=True)
        upper_parts = upper_parts.reindex(columns=[0, 1, 2], fill_value="")
        ji = lower_parts[2].str.strip()
        jk = upper_parts[2].str.strip()

        fik = pd.to_numeric(df['fik'], errors='coerce')

        def parse_j(series: pd.Series) -> pd.Series:                                  # For turning spin values (string) into floats
            s = series.astype(str).str.strip().str.strip('()')
            mask = s.str.contains('/', regex=False, na=False)
            out = pd.to_numeric(s, errors='coerce')
            if mask.any():
                parts = s[mask].str.split('/', n=1, expand=True)
                num = pd.to_numeric(parts[0].str.strip(), errors='coerce')
                den = pd.to_numeric(parts[1].str.strip(), errors='coerce')
                out.loc[mask] = num / den
            return out
        
        Ji = parse_j(ji)
        Jk = parse_j(jk)

        if species == 'H':
            spec = 'H'

        output = pd.DataFrame({
                              'Ion'       : spec,             # string
                              'lam_obs'   : lam_obs,          # Å
                              'A_ul'      : A_ul,             # 1/s
                              'Acc'       : Acc,              # Accuracy of A_ul
                              'E_l'       : Ei,               # eV
                              'E_u'       : Ek,               # eV
                              'J_l'       : Ji,               # dimless
                              'J_u'       : Jk,               # dimless
                              'g_l'       : Gi,               # dimless
                              'g_u'       : Gk,               # dimless
                              'fik'       : fik,              # dimless
                              'transition': df['Transition'].astype(str)
                              })

        output = output.replace([np.inf, -np.inf], np.nan)
        required = ['lam_obs', 'A_ul', 'E_l', 'E_u', 'g_l', 'g_u']
        valid_mask = output[required].notna().all(axis=1)
        valid_mask &= output['lam_obs'] > 0.0
        valid_mask &= output['A_ul'] > 0.0
        valid_mask &= output['g_l'] > 0.0
        valid_mask &= output['g_u'] > 0.0
        output = output.loc[valid_mask].copy()

        # Sorting the values with respect to observed wavelength
        out_f = (output
              .sort_values('lam_obs', kind='mergesort')
              .drop_duplicates(subset=['lam_obs', 'transition', 'A_ul'])
              .reset_index(drop=True))

        data = out_f

        return data

    def calc_central_crossection(self):
        # Returns σ_0 in cm^2 km/s such that s = integral(σ_0 * φ) [cm^2]
        
        sig0 = (self.A_ul * (self.lam0**3/(8 * np.pi)) * (self.g_u/self.g_l))                 # σ_v = σ_λ = σ_0 * φ_λ = σ_0 * φ_v * c/λ
        sig0 = sig0.to(u.cm**2 * u.km / u.s)
        ratio = np.divide(
            self.A_ul_err.to_value(1 / u.s),
            self.A_ul.to_value(1 / u.s),
            out=np.zeros_like(self.A_ul.value, dtype=float),
            where=np.abs(self.A_ul.to_value(1 / u.s)) > 0,
        )
        sig0_err = sig0 * ratio

        return sig0, sig0_err

    def _build_unique_states(self):
        lower_states = np.column_stack((
            np.asarray(self.E_l.to_value(u.eV), dtype=float).reshape(-1),
            np.asarray(self.g_l.to_value(u.dimensionless_unscaled), dtype=float).reshape(-1),
        ))
        upper_states = np.column_stack((
            np.asarray(self.E_u.to_value(u.eV), dtype=float).reshape(-1),
            np.asarray(self.g_u.to_value(u.dimensionless_unscaled), dtype=float).reshape(-1),
        ))
        states = np.vstack((lower_states, upper_states))
        finite_mask = np.all(np.isfinite(states), axis=1)
        return np.unique(states[finite_mask], axis=0)

    def partition_function(self, Temp_atm):
        T = Temp_atm.to_value(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")
        T = np.atleast_1d(np.asarray(T, dtype=float))

        if self._unique_states.size == 0:
            raise ValueError(f"No finite atomic states available for partition function: {self.species}")

        E_unique = self._unique_states[:, 0][:, None]
        g_unique = self._unique_states[:, 1][:, None]
        kb_eV_per_K = const.k_B.to_value(u.eV / u.K)
        boltz = g_unique * np.exp(-E_unique / (kb_eV_per_K * T[None, :]))
        Z = np.nansum(boltz, axis=0)

        if np.any((~np.isfinite(Z)) | (Z <= 0.0)):
            raise ValueError(f"Atomic partition function is zero or non-finite for {self.species}")

        return Z
  
