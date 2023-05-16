import os

from mp_api.client import MPRester
from pymatgen.core import Element, Species
from pymatgen.io.vasp.sets import MPScanRelaxSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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


def generate_vasp_inputs(structure, directory):
    vasp_input_set = MPScanRelaxSet(
        structure,
        user_incar_settings={
            "ISMEAR": 0,
            "ISYM": 0,
            "SIGMA": 0.03,
        },
        user_potcar_functional="PBE_54",
    )
    vasp_input_set.write_input(directory, include_cif=True)


def main():
    api_key = os.getenv("MATERIALS_PROJECT_API_KEY")
    metals = alkali_metals
    nonmetal_species = Species(symbol="O", oxidation_state=-2)
    directory = "binary_alkali_metal_oxides"
    material_docs = get_material_docs(api_key, metals, nonmetal_species)
    for material_doc in material_docs:
        structure = material_doc.structure
        conventional_standard_structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
        vasp_input_directory = os.path.join(directory, material_doc.formula_pretty)
        generate_vasp_inputs(conventional_standard_structure, vasp_input_directory)
        break


if __name__ == "__main__":
    main()
