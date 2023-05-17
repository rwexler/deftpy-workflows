import os

from ase.io import write
from emmet.core.provenance import Database
from mp_api.client import MPRester
from numpy.random import default_rng
from pymatgen.core import Element, Species
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPScanRelaxSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

rng = default_rng(42)

alkali_metals = [str(Element.from_Z(i)) for i in range(1, 119) if Element.from_Z(i).is_alkali]


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


def generate_vasp_inputs(structure, directory, task_ids, icsd_ids, energy_above_hull):
    vasp_input_set = MPScanRelaxSet(
        structure,
        user_incar_settings={
            "ISMEAR": 0,
            "ISYM": 0,
            "NPAR": 8,
            "SIGMA": 0.03,
            "SYMPREC": 1.0e-8,
        },
        user_potcar_functional="PBE_54",
    )
    vasp_input_set.write_input(directory, include_cif=True)

    # Rattle structure
    os.rename(os.path.join(directory, "POSCAR"), os.path.join(directory, "POSCAR.higher_symmetry"))
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.rattle()
    atoms.set_cell(atoms.cell + rng.normal(scale=0.001, size=atoms.cell.shape))
    write(os.path.join(directory, "POSCAR"), atoms, format="vasp")

    with open(os.path.join(directory, "task_ids.txt"), "w") as f:
        f.write("\n".join(str(task_id) for task_id in task_ids) + "\n")

    with open(os.path.join(directory, "icsd_ids.txt"), "w") as f:
        f.write("\n".join(str(icsd_id) for icsd_id in icsd_ids) + "\n")

    with open(os.path.join(directory, "energy_above_hull.txt"), "w") as f:
        f.write(f"{energy_above_hull}\n")


def main():
    api_key = os.getenv("MATERIALS_PROJECT_API_KEY")
    metals = alkali_metals
    nonmetal_species = Species(symbol="O", oxidation_state=-2)
    directory = "binary_alkali_metal_oxides"
    material_docs = get_material_docs(api_key, metals, nonmetal_species)
    for material_doc in material_docs:
        structure = material_doc.structure
        conventional_standard_structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

        formula_pretty = material_doc.formula_pretty
        symmetry_symbol = material_doc.symmetry.symbol.replace("/", "__")
        material = f"{formula_pretty}_{symmetry_symbol}"

        task_ids = [task_id.string for task_id in material_doc.task_ids]
        icsd_ids = material_doc.database_IDs[Database.ICSD]
        energy_above_hull = material_doc.energy_above_hull

        vasp_input_directory = os.path.join(directory, material)
        generate_vasp_inputs(
            conventional_standard_structure,
            vasp_input_directory,
            task_ids,
            icsd_ids,
            energy_above_hull
        )


if __name__ == "__main__":
    main()
