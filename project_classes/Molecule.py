import numpy as np
from molmass import Formula
from astropy import units as u
from project_func.errors import _not_quantity
from astroquery.nist import Nist
import pandas as pd

class Molecule:
  
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
    self.mass = Formula(self.species).mass * u.u
    self.A_ul, self.A_ul_err, self.lam0, self.g_u, self.g_l, self.E_u, self.E_l = self.pandas_to_numpy(self.data)
    self.sig_0, self.sig_0_err = self.calc_central_crossection()

  def get_Name(self):
    print(self.species)

  def pandas_to_numpy(self, data):
    """
    Numpy arrays with dimensions (N_lines, None) ex. (16,)
    """
    Aul = pd.to_numeric(data['A_ul']).to_numpy().reshape(-1, 1) / u.s
    Aul_err = pd.to_numeric(data['Acc']).to_numpy().reshape(-1, 1) / u.s
    lam0 = pd.to_numeric(data['lam_obs']).to_numpy().reshape(-1, 1) * u.AA
    gu = pd.to_numeric(data['g_u']).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled
    gl = pd.to_numeric(data['g_l']).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled
    Eu = pd.to_numeric(data['E_u']).to_numpy().reshape(-1, 1) * u.eV
    El = pd.to_numeric(data['E_l']).to_numpy().reshape(-1, 1) * u.eV

    return Aul, Aul_err, lam0, gu, gl, Eu, El

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

    # Wavelength in nm. If observed wavelength is missing i will use Ritz
    lam_obs = pd.to_numeric(df['Observed'], errors='coerce')
    lam_ritz = pd.to_numeric(df['Ritz'], errors='coerce')
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
    Acc = Acc.map(ACC_FRAC) * A_ul
    

    eiek = df['Ei           Ek'].str.split('-', n=1, expand=True)
    ei = eiek[0].str.strip(' []?') # lower energy (Ei)
    ek = eiek[1].str.strip(' []?') # upper energy (Ek)
    Ei = pd.to_numeric(ei, errors='coerce')
    Ek = pd.to_numeric(ek, errors='coerce')

    gigk = df['gi   gk'].str.split('-', n=1, expand=True)
    gi = gigk[0].str.strip()
    gk = gigk[1].str.strip()
    Gi = pd.to_numeric(gi, errors='coerce')
    Gk = pd.to_numeric(gk, errors='coerce')

    ji = df['Lower level'].str.split('|',n=2 , expand=True)[2].str.strip()
    jk = df['Upper level'].str.split('|',n=2 , expand=True)[2].str.strip()

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
                          'transition': df['Transition'].astype(str)
                          })

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
    sig0_err = sig0 * (self.A_ul_err/self.A_ul)

    return sig0, sig0_err
    