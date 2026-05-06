from project_func.Templates.Atoms.top_10_doubly_ionized_atoms import TOP_10_DOUBLY_IONIZED_ATOMS
from project_func.Templates.Atoms.top_10_neutral_atoms import TOP_10_NEUTRAL_ATOMS
from project_func.Templates.Atoms.top_10_singly_ionized_atoms import TOP_10_SINGLY_IONIZED_ATOMS


TOP_10_ATOMS = (
    TOP_10_NEUTRAL_ATOMS
    + TOP_10_SINGLY_IONIZED_ATOMS
    + TOP_10_DOUBLY_IONIZED_ATOMS
)

# Backward-compatible alias
atoms = TOP_10_ATOMS
