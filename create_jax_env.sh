#!/bin/bash
#
# Create a JAX environment at NERSC.
#
# Need to module load these in the scripts that are calling
# code running under this environment as well.
module load cudatoolkit/11.7
module load cudnn/8.9.1_cuda11
module load python
# Verify the versions of cudatoolkit and cudnn are compatible with JAX
module list
#
# Set up the environment, cloning from nersc-mpi4py to get the
# proper MPI environment.
conda create --name jax-env --clone nersc-mpi4py
conda activate jax-env
conda update --all -y
conda install numpy scipy ipykernel -y
python3 -m ipykernel install --user --name jax-env --display-name JAX-env
#
conda install -c conda-forge numba healpy -y
#conda install -c confa-forge ipython jupyter -y
#
#
# Activate the environment before using pip to install JAX(lib).
#conda activate jax-env
# Install the JAX library.
python3 -m pip install "jaxlib==0.4.7"
# Install a compatible wheel
python3 -m pip install --no-cache-dir "jax[cuda11_cudnn82]==0.4.7" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#
