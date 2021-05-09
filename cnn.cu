#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define USE_MNIST_LOADER
#define MNIST_DOUBLE


// includes, system
#include <string>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>



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

void forward(const double data[28][28]){

    // printf("run forward\n");

    
    float input[28][28];

    for (int i = 0; i<28; i++){
        for (int j = 0; j<28; j++){
            input[i][j] = data[i][j];
            // printf("%.2f ",data[i][j]);
            // printf("%d ",(int)ceil(data[i][j]));
        }
        // printf("\n");
    }


    l_input.clear();
	l_c1.clear();
	l_s1.clear();
	l_f.clear();

    // printf("**************************************\n");


    //example for convLayer 1:

    l_input.setInput((float *)input);

    //printf("input image: %f\n", &l_input.output[0][0]);
    

    int W_grid, H_grid;
    int W_out = 24, H_out = 24;
    //int M = 6;  // The first (x) dimension in the grid maps to the M output feature maps
    W_grid = ceilf(W_out/TILE_WIDTH); 	// number of horizontal tiles per output map
    if (W_grid == 0) W_grid = 1;
    H_grid = H_out/TILE_WIDTH; 	// number of vertical tiles per output map
    //int Y = H_grid * W_grid; //The second (y) dimension in the grid maps to the tiles in the output feature maps
    //int C = 1, K = 5;
    int bz = ceil((float)28/TILE_WIDTH)*ceil((float)28/TILE_WIDTH);
    dim3 gridDim(1, 6, bz);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    ConvLayerForward_Kernel_1<<<gridDim,blockDim>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight, 1, 28, 28, 24, 5, 6);

    apply_sigmoid <<<64,64>>>(l_c1.preact, l_c1.output, l_c1.bytes);
   
 

    // for pooling layer example:
    bz = ceil((float)6/TILE_WIDTH)*ceil((float)6/TILE_WIDTH);
    dim3 gridDimPool(1, 6, bz);
    dim3 blockDimPool(TILE_WIDTH, TILE_WIDTH, 1);
    //MaxPool2dForward_Kernel_1<<<gridDimPool,blockDimPool>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, 24, 24, 6, 4);
    MaxPool2dForward_Kernel_1(float input[6][24][24], float output[6][6][6], (float (*)[4][4])l_s1.weight, l_s1.bias ,24, 24, 6, 4);






    FullyConLayerBackward_kernel<<<gridDimfc,blockDimfc>>>(
            l_f.output,
            l_f.d_preact,
            (float (*)[6][6]) l_s1.preact,
            (float (*)[6][6][6]) l_f.d_weight,
            l_f.bias
        );
    
    //pooling backward:
    dim3 gridDimPool(TILE_WIDTH,TILE_WIDTH);
	bz = ceil((float)24/TILE_WIDTH)*ceil((float)24/TILE_WIDTH);
	if( bz == 0 )bz = 1;
	dim3 blockDimPool(1, 6, bz);
    //input_pointer, Inputimage_height, Inputimage_width, output_pointer, Outputimage_channel, pool_size
    poolingLayer_backward_GPU<<<gridDimPool,blockDimPool>>>((float (*)[24][24])l_c1.d_output, 24, 24, (float (*)[6][6])l_s1.preact, 6,  4);
    
    // //convolutional backward kernel
    // bz = ceil((float)28/TILE_WIDTH)*ceil((float)28/TILE_WIDTH);
    // dim3 gridDim(1, 6, bz);
    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // // ConvLayerBackward_Kernel<<<gridDim,blockDim>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight, l_c1.bias, 1, 24, 24, 6, 5, 6);
    
    // ConvLayerBackward_Kernel<<<gridDim,blockDim>>>(
    //     (float (*)[28])l_input.output, 
    //     (float (*)[24][24])l_c1.d_output, 
    //     (float (*)[24][24])l_c1.preact, 
    //     (float (*)[24][24])l_c1.d_preact, 
    //     (float (*)[5][5])l_c1.d_weight, 
    //     1, 24, 24, 6, 5, 6);
    bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
	bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);


    apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
	apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);


}

static void learn(){

    printf("test 09\n");
    for(int i=0; i< 10; i++){
    // // for(int i=0; i<10; i++){
    //     printf("label: %d \n", train_set[i].label);

        l_f.bp_clear();
		l_s1.bp_clear();
		l_c1.bp_clear();
        
        forward(train_set[i].data);
        makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
        backward();

        printf("label: %d \n", train_set[i].label);
    
    float *r1 = (float *)malloc(sizeof(float) * 10);

    cudaMemcpy(r1, l_f.preact, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    

    printf("ConvLayerForward_Kernel: \n");
    for (int i = 0; i < 10; i++){
        printf("%.2f ",*(r1 + i));
    }
    printf("\n preact-----------------------------------\n");

    float *r2 = (float *)malloc(sizeof(float) * 10);

    cudaMemcpy(r2, l_f.output, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    

    printf("ConvLayerForward_Kernel: \n");
    for (int i = 0; i < 10; i++){
        printf("%.2f ",*(r2 + i));
    }
    printf("\n output -----------------------------------\n\n");



    }  

    
}


static unsigned int classify(double data[28][28])
{
	float res[10];

	forward(data);

	unsigned int max = 0;

    cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	// cudaMemcpy(res, l_f.d_preact, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	printf("Error Rate: %.2lf%%\n", double(error) / double(test_cnt) * 100.0);
}


int main(){

    loadData();
    for (int i = 0; i<1; i++){
        learn();
    }
    
    //test();
    printf("finish\n");

    return 0;
}