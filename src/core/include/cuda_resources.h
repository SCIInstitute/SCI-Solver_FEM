#ifndef __CUDA_RESOURCES_H__
#define __CUDA_RESOURCES_H__

#include <stdio.h>
#include <cutil.h>

int getMaxThreads(const int max_regs_per_thread, int cuda_device);

#endif //end __CUDA_RESOURCES_H__
