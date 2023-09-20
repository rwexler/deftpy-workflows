""" Defect calculation workflow """
import sys
import os

from pymatgen.analysis.defects.generators import VacancyGenerator
from pymatgen.core import Element
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Incar, Kpoints
from pymatgen.io.vasp.sets import MPScanRelaxSet

from quacc.utils.defects import make_defects_from_bulk

RUNSCRIPT = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=dragon
#SBATCH --exclusive

ulimit -s unlimited

srun -n 64 /software/vasp.6.4.1/bin/vasp_gam
"""


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


def main():
    # Make vacancies from bulk Fm-3m Li2O
    contcar_path = "output/binary_alkali_metal_oxides/Li2O_Fm-3m/rerun/CONTCAR"
    atoms = AseAtomsAdaptor.get_atoms(Structure.from_file(contcar_path))
    vacancies = make_defects_from_bulk(
        atoms=atoms,
        defect_gen=VacancyGenerator,
        defect_charge=0,
        rm_species=["O"],
    )

    # Read INCAR
    incar_path = "output/binary_alkali_metal_oxides/Li2O_Fm-3m/rerun/INCAR"
    incar = Incar.from_file(incar_path)
    incar["ISIF"] = 2

    # Write input files
    for i in range(len(vacancies)):
        vacancy_stats = vacancies[i].info["defect_stats"]

        # Make directory
        directory = f"output" \
                    f"/binary_alkali_metal_oxides" \
                    f"/Li2O_Fm-3m" \
                    f"/rerun" \
                    f"/vacancies" \
                    f"/{vacancy_stats['defect_symbol']}_{vacancy_stats['defect_charge']}" \
                    f"/{vacancy_stats['distortions']}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Write POSCAR
        vacancy = AseAtomsAdaptor.get_structure(vacancies[i])
        vacancy.to(filename=f"{directory}/POSCAR")

        # Write INCAR
        magmom = get_high_spin_magmom(vacancy)
        vacancy.add_site_property("magmom", magmom)
        incar["MAGMOM"] = magmom
        incar.write_file(f"{directory}/INCAR")

        # Write KPOINTS
        Kpoints().write_file(f"{directory}/KPOINTS")  # default: gamma centered, 1x1x1

        # Write POTCAR using MPScanRelaxSet
        MPScanRelaxSet(vacancy, user_potcar_functional="PBE_54").potcar.write_file(f"{directory}/POTCAR")

        # Write runscript
        job_name = f"Li2O_Fm-3m" \
                   f"--{vacancy_stats['defect_symbol']}" \
                   f"--{vacancy_stats['defect_charge']}" \
                   f"--{vacancy_stats['distortions']}"
        with open(f"{directory}/runscript", "w") as f:
            f.write(RUNSCRIPT.format(job_name=job_name))


if __name__ == "__main__":
    main()
