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
  //timer.start();

  uint64_t * host_ptr, * device_ptr;

  const uint64_t size = 64;

  int grid = 1;
  int block = 1;

  cudaStream_t s_1, s_2;
  cudaStreamCreate(&s_1);
  cudaStreamCreate(&s_2);

  cudaMemcpyAsync(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost, s_2);
  cudaMemcpyAsync(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost, s_1);

  device_kernel_add<<<grid, block, 0, s_1>>>();
  device_kernel_add<<<grid, block, 0, s_2>>>();
  device_kernel_mul<<<grid, block, 0, s_1>>>();
  device_kernel_mul<<<grid, block, 0, s_2>>>();

  cudaMemcpyAsync(device_ptr, host_ptr, size, cudaMemcpyHostToDevice, s_2);
  cudaMemcpyAsync(device_ptr, host_ptr, size, cudaMemcpyHostToDevice, s_1);

  cudaStreamSynchronize(s_1);
  cudaStreamSynchronize(s_2);

  cudaStreamDestroy(s_1);
  cudaStreamDestroy(s_2);

}

//custom created streams run concurrently (s_1 and s_2 run independently of each other), but tasks scheduled on each stream execute sequentially, respecting the order in which they were scheduled.
