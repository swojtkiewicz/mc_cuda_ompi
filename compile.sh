#!/bin/sh

#get current path
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
#move inside current path
cd "$parent_path"

#compile CUDA
nvcc -O3 -c "./src/SimulationEngine.cu" -I"/usr/local/cuda/include" -I"./src" -gencode=arch=compute_70,code=\"sm_70,compute_70\" -Xcompiler -fopenmp --machine 64 --ptxas-options=-v -cudart static -odir "./" 2>&1 | tee nvcc_compile.log

#compile C++
mpicxx -c "./src/utilities.cpp" -m64 -fopenmp -fmessage-length=0 -I"./src"

mpicxx -c "./src/main.cpp" -m64 -fopenmp -fmessage-length=0 -I"/usr/local/cuda/include" -I"/usr/inlude/openmpi-x86_64" -I"./src"

#link
mpicxx -m64 "./utilities.o" "./SimulationEngine.o" "./main.o" -L"/usr/local/cuda/lib64" -L"/usr/local/cuda/lib" -L"/usr/lib64/openmpi/lib" -o "./Release/mc_cuda_ompi" -lcuda -lcurand -lcudart -lmpi -lgomp
