################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/main.cpp \
../src/utilities.cpp 

CU_SRCS += \
../src/SimulationEngine.cu 

CPP_DEPS += \
./src/main.d \
./src/utilities.d 

CU_DEPS += \
./src/SimulationEngine.d 

OBJS += \
./src/SimulationEngine.o \
./src/main.o \
./src/utilities.o 


# Each subdirectory must supply rules for building sources it contributes
src/SimulationEngine.o: ../src/SimulationEngine.cu src/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	nvcc --expt-extended-lambda -I/usr/local/cuda/include -O3 --use_fast_math -c -Xcompiler="-std=c++0x -fopenmp -Wall" -gencode arch=compute_70,code=compute_70 --machine 64 --ptxas-options=-v -cudart static -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp src/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	mpicxx -std=c++0x -I/usr/local/cuda/include -I/usr/include/openmpi-x86_64 -O3 -Wall -c -fmessage-length=0 -m64 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-src

clean-src:
	-$(RM) ./src/SimulationEngine.d ./src/SimulationEngine.o ./src/main.d ./src/main.o ./src/utilities.d ./src/utilities.o

.PHONY: clean-src

