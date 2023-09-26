#!/bin/bash
mkdir rerun
cp CONTCAR INCAR KPOINTS POTCAR runscript rerun/
cd rerun/
mv CONTCAR POSCAR
sbatch runscript
