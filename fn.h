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

__global__ void FullyConLayerForward();
__global__ void FullyConLayerForward_bias();
__global__ void FullyConLayerBackward();
__global__ void FullyConLayerBackward_bias();

#endif