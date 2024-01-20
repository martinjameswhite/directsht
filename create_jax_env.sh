#!/bin/bash
#
# Create a JAX environment at NERSC.
#
# Need to module load these in the scripts that are calling
# code running under this environment as well.
#
module load cudatoolkit
module load cudnn/
module load python
# Verify the versions of cudatoolkit and cudnn are compatible with JAX
# module list
#
# Set up the environment, cloning from nersc-mpi4py to get the
# proper MPI environment.
conda create --name jax-env --clone nersc-mpi4py
conda activate jax-env
conda update --all -y
conda install -c conda-forge numpy scipy ipykernel -y
python3 -Xfrozen_modules=off -m ipykernel \
        install --user --name jax-env --display-name JAX-env
#
conda install -c conda-forge numba healpy -y
#
python3 -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html



# test installation
# python -c "from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform); import jax.numpy as jnp; print(jnp.exp(2.34))"
