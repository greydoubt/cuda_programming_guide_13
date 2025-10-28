#include <cuda_runtime.h>
#include <stdio.h>

//Kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
  //...
  //Invoke kernel with N threads
    vecAdd<<<1, N>>>(A, B, C);
//... 
}
