#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int getMaxThreads(
    const int max_regs_per_thread, 
    int cuda_device) {
  cudaDeviceProp d;
  cudaGetDeviceProperties(&d, cuda_device);
  return d.regsPerBlock / max_regs_per_thread;
}
