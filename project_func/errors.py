

def _not_quantity(name: str):
  raise TypeError(f"{name} must be an astropy Quantity with units (e.g., 500*u.nm).")