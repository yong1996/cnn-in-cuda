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


__global__ void MaxPool2dForward_Kernel_1(int stride, int poolSize, float input[6][24][24], float output[6][6][6]){
    int m = blockIdx.x;
    int h = blockIdx.y / W_grid  + threadIdx.y;
    int w = blockIdx.y % W_grid + threadIdx.x;
    float acc = 0.;
    //for (int c = 0;  c < C; c++) {		// sum over all input channels, in this case, the channel is 1
       for (int p = 0; p < K; p++){
            for (int q = 0; q < K; q++){
                acc += input[h+p][w+q] * weight[m][p][q];
            }  
       }		
          
    //}
    output[m][h][w] = acc;
}


