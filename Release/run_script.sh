#!/bin/sh

mpirun --mca btl self,vader,tcp -np 4 -hostfile hosts_NIRFAST.txt mc_cuda_ompi
#mpirun --mca btl self,sm,tcp -np 7 -hostfile hosts_IBIB.txt CUDA_OMPI_ALL_FEATURES
