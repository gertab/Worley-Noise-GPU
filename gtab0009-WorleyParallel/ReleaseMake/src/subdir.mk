################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/Tests.cu \
../src/WorleyParallel.cu 

OBJS += \
./src/Tests.o \
./src/WorleyParallel.o 

CU_DEPS += \
./src/Tests.d \
./src/WorleyParallel.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -DNDEBUG -O3 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -DNDEBUG -O3 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


