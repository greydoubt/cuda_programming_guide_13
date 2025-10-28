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

// everything will be executed on the default stream

int main()
{
  timer.start();

  uint64_t * host_ptr, * device_ptr;

  const uint64_t size = 64;

  int grid 1;
  
  int block 1;

  //cudaError_t cudaMemcpy  (   void *    dst,
  //  const void *    src,
  //  size_t    count,
  //  enum cudaMemcpyKind   kind   
  //)   
  //https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_g48efa06b81cc031b2aa6fdc2e9930741.html

  cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
  
  device_kernel_add<<<grid, block>>>();

  host_func_add();

  device_kernel_mul<<<grid, block>>>();

  host_func_mul();
  
  cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost);
  
  cudaFreeHost(host_ptr);
  cudaFree    (device_ptr);
  timer.stop(" ");
  //cudaDeviceSynchronize();
}
