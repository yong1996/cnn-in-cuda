#define USE_MNIST_LOADER
#define MNIST_DOUBLE


// includes, system
#include <string>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// //opencv for testing
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// using namespace cv;

#include "mnist.h"
#include "layer.h"
#include "util.h"


//define the kernel size
#define TILE_WIDTH 4  //for small example


// set Layer
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(5*5, 6, 24*24*6);
static Layer l_s1 = Layer(4*4, 1, 6*6*6);
static Layer l_f = Layer(6*6*6, 10, 10);

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;


static inline void loadData(){
    mnist_load("MNIST_data/train-images.idx3-ubyte", "MNIST_data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("MNIST_data/t10k-images.idx3-ubyte", "MNIST_data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}



// void learn(){
   
// }

// void test(){
//     return 0;
// }


void forward(const double data[28][28]){

    
    float input[28][28];

    for (int i = 0; i<28; i++){
        for (int j = 0; j<28; j++){
            input[i][j] = data[i][j];
        }
    }


    //example for convLayer 1:

    l_input.setInput((float *)input);

    int W_grid, H_grid;
    int W_out = 24, H_out = 24;
    int M = 6;  // The first (x) dimension in the grid maps to the M output feature maps
    W_grid = W_out/TILE_WIDTH; 	// number of horizontal tiles per output map
    H_grid = H_out/TILE_WIDTH; 	// number of vertical tiles per output map
    int Y = H_grid * W_grid; //The second (y) dimension in the grid maps to the tiles in the output feature maps
    int C = 1, K = 5;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, Y, 1);
    // ConvLayerForward_Kernel<<< gridDim, blockDim>>>(int C = 1, W_grid, int K = 5, (float (*)[28])l_input.output,  (float (*)[24][24])l_c1.preact,(float (*)[5][5])l_c1.weight);
    ConvLayerForward_Kernel<<< gridDim, blockDim>>>(C, W_grid, K, (float (*)[28])l_input.output,  (float (*)[24][24])l_c1.preact,(float (*)[5][5])l_c1.weight);
    //apply_sigmoid <<<64,64>>>(l_c1.preact, l_c1.output, l_c1.size);
    printf("%f", l_c1.preact[1]);

    write_ppm("test.ppm", 24, 24, 255, l_c1.preact[0]);

                                               

}
// cnn -- iteration, lr
// int main(int argc, char** argv) {
//     // loaddata();
// 	learn();
// 	test();

// 	return 0;
// }




int main(){
    // loadData()

    printf("test 1\n");
    forward(train_set[0].data);

    return 0;
}