import copy

import astropy.constants as const
import astropy.units as u


MOLECULE_TEMPLATES = {
    "CO": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "CO/12C-16O/Li2015",
            "database": "Li2015",
            "localdatabase": "exomol_data",
        },
    },
    "SO": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "SO/32S-16O/SOLIS",
            "database": "SOLIS",
            "localdatabase": "exomol_data",
        },
    },
    "CH4": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "CH4/12C-1H4/MM",
            "database": "MM",
            "localdatabase": "exomol_data",
        },
    },
    "NH3": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "NH3/14N-1H3/CoYuTe",
            "database": "CoYuTe",
            "localdatabase": "exomol_data",
        },
    },
    "TiO": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "TiO/48Ti-16O/ToTo",
            "database": "ToTo",
            "localdatabase": "exomol_data",
        },
    },
    "SiO": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "SiO/28Si-16O/SiOUVenIR",
            "database": "SiOUVenIRJT",
            "localdatabase": "exomol_data",
        },
    },
    "NO": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "NO/14N-16O/XABC",
            "database": "XABC",
            "localdatabase": "exomol_data",
        },
    },
    "H2O": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "H2O/1H2-16O/POKAZATEL",
            "database": "POKAZATEL",
            "localdatabase": "exomol_data",
        },
    },
    "CO2": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "CO2/12C-16O2/Dozen",
            "database": "Dozen",
            "localdatabase": "exomol_data",
        }
    },
    "OH": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "OH/16O-1H/MYTHOS",
            "database": "MYTHOS",
            "localdatabase": "exomol_data",
        },
    },
    "O2": {
        "source": "hitran",
        "fetch_kwargs": {
            "path": "O2/16O2/HITRAN",
            "database": "HITRAN",
            "localdatabase": "exomol_data",
            "databank_name": "HITRAN-O2-loc",
        },
    },        
    "H2S": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "H2S/1H2-32S/AYT2",
            "database": "AYT2",
            "localdatabase": "exomol_data",
        },
    },
    "SO2": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "SO2/32S-16O2/ExoAmes",
            "database": "ExoAmes",
            "localdatabase": "exomol_data",
        },
    },
    "HCN": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "HCN/1H-12C-14N/Harris",
            "database": "Harris",
            "localdatabase": "exomol_data",
        },
    },
    "C3": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "C3/12C3/AtLast",
            "database": "AtLast",
            "localdatabase": "exomol_data",
        },
    },
    "OCS": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "OCS/16O-12C-32S/OYT8",
            "database": "OYT8",
            "localdatabase": "exomol_data",
        },
    },

}
  