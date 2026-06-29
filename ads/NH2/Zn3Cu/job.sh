#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --account=mit197
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -o vasp.shared.%j.out
#SBATCH -e vasp.shared.%j.err
#SBATCH -p shared
#SBATCH --export=ALL

# Load modules
module load cpu/0.17.3b
module load aocc/3.2.0
module load openmpi/4.1.3
module load vasp6/6.2.1

# Set executable and environment
exe=`which vasp_std`
export OMP_NUM_THREADS=1

# Run VASP
mpirun --mca btl_openib_if_include "mlx5_2:1" --mca btl self,vader $exe >& vasp.log
