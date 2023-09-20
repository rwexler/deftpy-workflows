import os

from ase.io import write
from emmet.core.provenance import Database
from mp_api.client import MPRester
from numpy.random import default_rng
from pymatgen.core import Species, Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPScanRelaxSet

rng = default_rng(42)

alkali_metals = ["Li", "Na", "K", "Rb", "Cs"]
alkaline_earth_metals = ["Be", "Mg", "Ca", "Sr", "Ba"]
group_13_metals = ["Al", "Ga", "In", "Tl"]
group_14_metals = ["Sn", "Pb"]
pnictogen_metals = ["Bi"]
chalcogen_metals = ["Po"]

RUNSCRIPT = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=dragon
#SBATCH --exclusive

ulimit -s unlimited

srun -n 64 /software/vasp.6.4.1/bin/vasp_std
"""


def get_material_docs(api_key, metals, nonmetal_species):
    with MPRester(api_key) as mpr:
        docs = mpr.summary.search(
            chemsys=f"*-{nonmetal_species.element.symbol}",
            deprecated=False,
            possible_species=[nonmetal_species.to_pretty_string()],
            theoretical=False,
        )

    material_docs = []
    for doc in docs:
        structure = doc.structure
        symbol_set = structure.symbol_set

        if not set(metals) & set(symbol_set):
            continue

        material_docs.append(doc)

    return material_docs


def get_high_spin_magmom(structure):
    high_spin_magmom = []
    for site in structure:
        element = Element(site.specie.symbol)
        if element.block == "s":
            high_spin_magmom.append(1)
        elif element.block == "p":
            high_spin_magmom.append(3)
        elif element.block == "d":
            high_spin_magmom.append(5)
        elif element.block == "f":
            high_spin_magmom.append(7)
        else:
            raise ValueError(f"Unknown block {element.block}")
    return high_spin_magmom


def get_lmaxmix(structure):
    # if the structure has transition metals, return 4
    # if the structure has lanthanides or actinides, return 6
    # otherwise, return 2
    lmaxmix = 2
    for site in structure:
        element = Element(site.specie.symbol)
        if element.is_transition_metal:
            lmaxmix = 4
        elif element.is_lanthanoid or element.is_actinoid:
            lmaxmix = 6
    return lmaxmix


def get_minimum_bond_distance(structure):
    minimum_bond_distance = float("inf")
    for i, site1 in enumerate(structure):
        for j, site2 in enumerate(structure):
            if i == j:
                continue
            bond_distance = structure.get_distance(i, j)
            if bond_distance < minimum_bond_distance:
                minimum_bond_distance = bond_distance

    return minimum_bond_distance


def generate_vasp_inputs(structure, directory, task_ids, icsd_ids, energy_above_hull, band_gap, percent=0.1, rattle_structure=True):
    magmom = get_high_spin_magmom(structure)
    structure.add_site_property("magmom", magmom)
    lmaxmix = get_lmaxmix(structure)
    vasp_input_set = MPScanRelaxSet(
        structure,
        user_incar_settings={
            # Level of Theory
            "ENCUT": 520,  # Plane-wave kinetic energy cutoff (recommended starting value)

            # SCF
            "ALGO": "All",  # SCF convergence algorithm
            "ISMEAR": 0,  # Smearing method
            "SIGMA": 0.01,  # Smearing parameter
            "LREAL": False,  # Use real-space projection for the augmentation charge
            "LCHARG": False,  # Writes the charge density to the CHGCAR file
            "NELM": 150,  # Maximum number of SCF iterations
            "NELMIN": 4,  # Minimum number of SCF iterations
            "LMAXMIX": lmaxmix,

            # Geometry Optimization
            "EDIFFG": -0.03,  # Maximum net force for convergence
            "NSW": 200,  # Maximum number of geometry optimization steps
            "ISYM": 0,  # Symmetry constraints
            "SYMPREC": 1e-8,  # Precision for symmetry detection

            # Parallel Performance
            "NCORE": 4,  # Number of cores that work on one band

            # Miscellaneous
            "EFERMI": "MIDGAP",

            # Remove
            "ENAUG": None,
            "LAECHG": None,
            "LELF": None,
            "LVTOT": None,
            "LWAVE": None,
        },
        user_kpoints_settings=Kpoints.automatic_density_by_vol(structure, 100),
        user_potcar_functional="PBE_54",
    )
    vasp_input_set.write_input(directory, include_cif=True)

    # Rattle structure
    if rattle_structure:
        os.rename(os.path.join(directory, "POSCAR"), os.path.join(directory, "POSCAR.higher_symmetry"))
        atoms = AseAtomsAdaptor.get_atoms(structure)
        minimum_bond_distance = get_minimum_bond_distance(structure)
        stdev = percent * minimum_bond_distance
        atoms.rattle(stdev=stdev)
        atoms.set_cell(atoms.cell + rng.normal(scale=stdev, size=atoms.cell.shape))
        write(os.path.join(directory, "POSCAR"), atoms, format="vasp")

    # Write metadata
    if task_ids:
        with open(os.path.join(directory, "task_ids.txt"), "w") as f:
            f.write("\n".join(str(task_id) for task_id in task_ids) + "\n")

    if icsd_ids:
        with open(os.path.join(directory, "icsd_ids.txt"), "w") as f:
            f.write("\n".join(str(icsd_id) for icsd_id in icsd_ids) + "\n")

    if energy_above_hull:
        with open(os.path.join(directory, "energy_above_hull.txt"), "w") as f:
            f.write(f"{energy_above_hull}\n")

    if band_gap:
        with open(os.path.join(directory, "band_gap.txt"), "w") as f:
            f.write(f"{band_gap}\n")

    if rattle_structure:
        with open(os.path.join(directory, "rattle.txt"), "w") as f:
            f.write(f"percent {percent}\n")
            f.write(f"minimum_bond_distance {minimum_bond_distance}\n")
            f.write(f"stdev {stdev}\n")

    # Write runscript
    with open(os.path.join(directory, "runscript"), "w") as f:
        f.write(RUNSCRIPT.format(job_name=directory.split("/")[-1]))


def main():
    api_key = os.getenv("MATERIALS_PROJECT_API_KEY")
    metals = alkali_metals
    nonmetal_species = Species(symbol="O", oxidation_state=-2)
    directory = "binary_alkali_metal_oxides"
    material_docs = get_material_docs(api_key, metals, nonmetal_species)
    for material_doc in material_docs:
        structure = material_doc.structure

        formula_pretty = material_doc.formula_pretty
        symmetry_symbol = material_doc.symmetry.symbol.replace("/", "__")
        material = f"{formula_pretty}_{symmetry_symbol}"

        task_ids = [task_id.string for task_id in material_doc.task_ids]
        icsd_ids = material_doc.database_IDs[Database.ICSD]
        energy_above_hull = material_doc.energy_above_hull
        band_gap = material_doc.band_gap

        vasp_input_directory = os.path.join(directory, material)
        generate_vasp_inputs(
            structure,
            vasp_input_directory,
            task_ids,
            icsd_ids,
            energy_above_hull,
            band_gap,
        )


if __name__ == "__main__":
    main()
