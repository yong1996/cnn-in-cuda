#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

// includes, system
#include <string>


#include "layer.h"


#ifndef FN_H
#define FN_H



// __global__ void FullyConLayerForward_kernel();
__global__ void FullyConLayerForward_kernel(float* Md, float* Nd, float* Pd, float* B, int M_height_in, int M_width_N_height_in, int N_width_in , int height_out, int width_out);
__global__ void FullyConLayerForward_bias();
__global__ void FullyConLayerBackward();
__global__ void FullyConLayerBackward_bias();

#endif