from glob import glob

import covalent as ct
from ase.calculators.vasp import Vasp
from ase.io import read

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
        "ntasks": 8,
        "mem-per-cpu": "4G",
        "time": "00:10:00",
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
def calculate_potential_energy(system):
    return system.get_potential_energy()


@ct.lattice
def calculate_defect_formation_energy(system, calculator):
    system_directory = calculator.directory + "/system"
    defect_directory = calculator.directory + "/vacancy"

    system.calc = calculator
    system.calc.set(directory=system_directory)
    system_potential_energy = calculate_potential_energy(system)

    defect = system[:-1].copy()  # defect = neutral oxygen vacancy
    defect.calc = calculator
    defect.calc.set(directory=defect_directory)
    defect_potential_energy = calculate_potential_energy(defect)

    o2_potential_energy = -9.88511959
    defect_formation_energy = defect_potential_energy + 0.5 * o2_potential_energy - system_potential_energy

    return defect_formation_energy


if __name__ == "__main__":
    directories = sorted(glob("*_mp-*"))
    for directory in directories:
        s = read(directory + "/POSCAR")
        s.set_initial_magnetic_moments([0.6] * len(s))
        c = Vasp(
            directory=remote_workdir + "/" + directory,
            command=command,
            setups="recommended",
            pp="PBE",
        )
        c.read_incar(directory + "/INCAR")
        c.read_kpoints(directory + "/KPOINTS")
        dispatch_id = ct.dispatch(calculate_defect_formation_energy)(s, c)
        result = ct.get_result(dispatch_id=dispatch_id, wait=True)
        print(f"Calculation status = {result.status} \n Defect formation energy = {result.result:.4f} eV")
