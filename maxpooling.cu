#include <windows.h>
#include <iostream>

#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

#include "layer.h"
#include "maxpooling.h"


// includes, system
#include <string>

// includes, kernels
#include <cnn_kernel.cu>


// poolingLayer_forward_GPU_naive<<<numBlocks,threadsPerBlock>>>(input_pointer, Inputimage_height,
//     Inputimage_width, Output_pointer, Outputimage_channel, pool_size);

