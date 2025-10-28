#include <cuda_runtime.h>
#include <stdio.h>

__global__ void device_kernel_add() {
  printf("Does add on device");
}

__global__ void device_kernel_mul() {
  printf("Does mul on device");
}

void host_func_add() {
  printf("Does add on host");
}

void host_func_mul() {
  printf("Does mul on host");
}

int main()
{
  helloCPU();
  
  helloGPU<<<1, 1>>>();
  
  cudaDeviceSynchronize();
}
