import os
from glob import glob

import covalent as ct
from ase.calculators.vasp import Vasp
from ase.io import read
from pymatgen.analysis.defects.core import Defect
from pymatgen.analysis.defects.generators import VacancyGenerator
from pymatgen.analysis.defects.thermo import DefectEntry
from pymatgen.core import DummySpecies, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from shakenbreak.input import Distortions

remote_workdir = "your remote work directory"
vasp_pp_path = "your pseudopotential path"
command = "your VASP command"
executor = ct.executor.SlurmExecutor(
    username="your username",
    address="your address",
    ssh_key_file="your ssh key file",
    remote_workdir=remote_workdir,
    options={
        "job-name": "covalent",
        "nodes": 1,
        "ntasks": 64,
        "mem-per-cpu": "3G",
        "ntasks-per-core": 1,
        "hint": "nomultithread",
    },
    prerun_commands=[
        "export VASP_PP_PATH=" + vasp_pp_path,
        ". /opt/intel/oneapi/setvars.sh --force",
        "ulimit -s unlimited",
    ],
)


@ct.electron(executor=executor)
def relax_system(system, calculator, isif=3):
    system_relaxation_directory = f"{calculator.directory}/system_relaxation"

    # Run first system relaxation
    first_system_relaxation_directory = f"{system_relaxation_directory}/run_1"
    system.calc = calculator
    system.calc.set(directory=first_system_relaxation_directory, isif=isif)
    system.get_potential_energy()

    # Run second system relaxation
    second_system_relaxation_directory = f"{system_relaxation_directory}/run_2"
    system.calc = calculator
    system.calc.set(directory=second_system_relaxation_directory, isif=isif)
    system.get_potential_energy()

    return system


@ct.electron
def get_defect_entry_from_defect(
        defect: Defect,
        defect_supercell: Structure,
        charge_state: int,
        dummy_species: DummySpecies = DummySpecies("X"),
):
    """
    Generate DefectEntry object from Defect object.
    This is used to describe a Defect using a certain simulation cell.
    """
    # Dummy species (used to keep track of the defect coords in the supercell)
    # Find its fractional coordinates & remove it from supercell
    symbol = dummy_species.symbol
    dummy_site = next((site for site in defect_supercell if site.species.elements[0].symbol == symbol), None)
    sc_defect_frac_coords = dummy_site.frac_coords
    defect_supercell.remove(dummy_site)

    computed_structure_entry = ComputedStructureEntry(
        structure=defect_supercell,
        energy=0.0,  # needs to be set, so set to 0.0
    )

    return DefectEntry(
        defect=defect,
        charge_state=charge_state,
        sc_entry=computed_structure_entry,
        sc_defect_frac_coords=sc_defect_frac_coords,
    )


@ct.electron
def generate_defect_entries(relaxed_system, symbol="O", charge=0, min_length=9):
    relaxed_system_structure = AseAtomsAdaptor.get_structure(relaxed_system)
    defect_generator = VacancyGenerator()

    defects = [
        x for x in defect_generator.get_defects(relaxed_system_structure)
        if x.site.specie.symbol == symbol
    ]

    defect_entries = []
    for defect in defects:
        defect.user_charges = [charge]
        defect_supercell = defect.get_supercell_structure(
            min_length=min_length,
            dummy_species=DummySpecies("X"),
        )
        defect_entry = get_defect_entry_from_defect(
            defect=defect,
            charge_state=charge,
            defect_supercell=defect_supercell,
        )
        defect_entries.append(defect_entry)
    return defect_entries


@ct.electron
def distort_defects(defect_entries):
    distorted_defects = {}
    for defect_entry in defect_entries:
        distortions = Distortions(defects=[defect_entry])
        defect_dicts, distortion_metadata = distortions.apply_distortions()
        for defect_id, defect_dict in defect_dicts.items():
            defect_distortions_dict = defect_dict["charges"][defect_entry.charge_state]["structures"]["distortions"]

            distorted_defects.update({
                f"{defect_id}/{distortion_id}": AseAtomsAdaptor.get_atoms(distorted_defect_structure)
                for distortion_id, distorted_defect_structure in defect_distortions_dict.items()
            })

    return distorted_defects


@ct.electron(executor=executor)
@ct.lattice
def relax_defects(distorted_defects, calculator, isif=2):
    relaxed_defects = []
    for defect_distortion_id, distorted_defect in distorted_defects.items():
        defect_relaxation_directory = f"{calculator.directory}/defect_relaxation/{defect_distortion_id}"
        distorted_defect.calc = calculator
        distorted_defect.calc.set(directory=defect_relaxation_directory, isif=isif)
        distorted_defect.get_potential_energy()
        relaxed_defects.append(distorted_defect)
    return relaxed_defects


@ct.lattice
def run_defect_calculations(system, calculator):
    relaxed_system = relax_system(system, calculator)
    defect_entries = generate_defect_entries(relaxed_system)
    distorted_defects = distort_defects(defect_entries)
    relaxed_defects = relax_defects(distorted_defects, calculator)
    return relaxed_defects


def main():
    directories = sorted(glob("your vasp input directories"))
    for directory in directories:
        s = read(os.path.join(directory, "POSCAR"))
        s.set_initial_magnetic_moments([0.6] * len(s))
        c = Vasp(
            directory=os.path.join(remote_workdir, directory),
            command=command,
            setups="recommended",
            pp="PBE",
        )
        c.read_incar(os.path.join(directory, "INCAR"))
        dispatch_id = ct.dispatch(run_defect_calculations)(s, c)
        result = ct.get_result(dispatch_id=dispatch_id, wait=True)
        print(f"{directory} calculation status = {result.status}")


if __name__ == "__main__":
    main()
