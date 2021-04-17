// includes, system
#include <string>

// includes, kernels
#include <cnn_kernel.cu>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "mnist.h"
#include "cnn.cpp"


//define the kernel size
#define TILE_WIDTH = 4;  //for small example





void learn(){


    //example for convLayer 1:
    int W_grid, H_grid;
    int W_out = 27, H_out = 27;
    int M = 6;  // The first (x) dimension in the grid maps to the M output feature maps
    W_grid = W_out/TILE_WIDTH; 	// number of horizontal tiles per output map
    H_grid = H_out/TILE_WIDTH; 	// number of vertical tiles per output map
    Y = H_grid * W_grid; //The second (y) dimension in the grid maps to the tiles in the output feature maps
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, Y, 1);
    ConvLayerForward_Kernel<<< gridDim, blockDim>>>(…);

     
    
}

void test(){
    return 0;
}

// cnn -- iteration, lr
// int main(int argc, char** argv) {
//     // loaddata();
// 	learn();
// 	test();

// 	return 0;
// }