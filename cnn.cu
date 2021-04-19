#define USE_MNIST_LOADER
#define MNIST_DOUBLE


// includes, system
#include <string>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// //opencv for testing
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// using namespace cv;

#include "mnist.h"
#include "layer.h"
#include "layer.cu"
//#include "maxpooling.h"
//#include "util.h"


//define the kernel size
#define TILE_WIDTH 16  //for small example


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

    printf("run forward\n");

    
    float input[28][28];

    for (int i = 0; i<28; i++){
        for (int j = 0; j<28; j++){
            input[i][j] = data[i][j];
            printf("%.2f ",data[i][j]);
        }
        printf("\n");
    }

    printf("**************************************\n");


    //example for convLayer 1:

    l_input.setInput((float *)input);

    //printf("input image: %f\n", &l_input.output[0][0]);
    

    int W_grid, H_grid;
    int W_out = 24, H_out = 24;
    int M = 6;  // The first (x) dimension in the grid maps to the M output feature maps
    W_grid = ceilf(W_out/TILE_WIDTH); 	// number of horizontal tiles per output map
    if (W_grid = 0) W_grid = 1;
    H_grid = H_out/TILE_WIDTH; 	// number of vertical tiles per output map
    int Y = H_grid * W_grid; //The second (y) dimension in the grid maps to the tiles in the output feature maps
    int C = 1, K = 5;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 gridDim(M, Y, 1);
    // ConvLayerForward_Kernel<<< gridDim, blockDim>>>(int C = 1, W_grid, int K = 5, (float (*)[28])l_input.output,  (float (*)[24][24])l_c1.preact,(float (*)[5][5])l_c1.weight);
    // ConvLayerForward_Kernel_1<<< gridDim, blockDim>>>((float (*)[28])l_input.output,  (float (*)[24][24])l_c1.preact,(float (*)[5][5])l_c1.weight);


    int bz = ceil((float)28/TILE_WIDTH)*ceil((float)28/TILE_WIDTH);
    dim3 gridDim(1, 6, bz);

    ConvLayerForward_Kernel_1<<<gridDim,blockDim>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight, 6, 28, 24, 24, 5, 6);
    //fp_preact_c1<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);


    float *result = (float *)malloc(sizeof(float) * 24*24*6);

    cudaMemcpy(result,
		l_c1.preact,
		24*24*6 * sizeof(float),
		cudaMemcpyDeviceToHost);
    //apply_sigmoid <<<64,64>>>(l_c1.preact, l_c1.output, l_c1.size);

    for (int i = 0; i < 6; i++){
        for (int j = 0; j <24; j++){
            for (int z = 0; z < 24; z++){
                printf("%.2f",*(result + i+j+z));
            }
            printf("\n");
        }

        printf("-----------------------------------\n");
    }

    

    //write_ppm("test.ppm", 24, 24, 255, l_c1.preact[0]);

    //for sigmoid layer:
    // apply_sigmoid<<<64,64>>> (l_c1.preact, l_c1.output, l_c1.bytes);

    // MaxPool2dForward_Kernel_1<<<>>>((float(*)[24][24])l_c1.output, (float(*)[6][6]l_s1.preact, (float(*)[4][4]l_s1.weight)));
    // apply_sigmoid<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.bytes);


    // //for fully connected layer
    // FullyConLayerForward<<<64, 64>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
	// FullyConLayerForward_bias<<<64, 64>>>(l_f.preact, l_f.bias);
	// apply_sigmoid<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);

                                               

}
// cnn -- iteration, lr
// int main(int argc, char** argv) {
//     // loaddata();
// 	learn();
// 	test();

// 	return 0;
// }




int main(){
    loadData();

    printf("test 2\n");
    forward(train_set[0].data);

    return 0;
}