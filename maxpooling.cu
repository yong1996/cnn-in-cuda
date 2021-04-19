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

// input_pointer, output_pointer, inputimage_height, inputimage_width, outputimage_channel, pool_size 
__global__ void MaxPool2dForward_Kernel_1(float input[6][24][24], float output[6][6][6], int H_in, int W_in, int M, int pool_size){
    int n, m, h, w, p, q;
	int H_out = H_in/pool_size;
	int W_out = W_in/pool_size;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0){
        W_grid = 1;
    }
		
	n = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;
	//h and w is not center point of calculating, it's upper left corner point of Input image
	float acc = 0;
	for (p = 0; p < pool_size; p++) { // loop over KxK input samples
		for (q = 0; q < pool_size; q++)
			if(h < H_out && w < W_out)
				acc = acc + X[n*(M*H_in*W_in)+ m*(H_in*W_in) +
				              (pool_size * h + p)*(W_in) + (pool_size * w + q)] / (pool_size * pool_size);
                acc = acc + input[h+p][w+q] / (pool_size * pool_size);
	}
	__syncthreads();
	if(h < H_out && w < W_out)
	{
		// Y[n*(M*H_out*W_out)+ m*(H_out*W_out) + h*(W_out) + w] = acc;
		output[m][h][w] = acc;

	}
}

