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
    

    int bz;

    int W_grid, H_grid;
    int W_out = 24, H_out = 24;
    //int M = 6;  // The first (x) dimension in the grid maps to the M output feature maps
    W_grid = ceilf(W_out/TILE_WIDTH); 	// number of horizontal tiles per output map
    if (W_grid == 0) W_grid = 1;
    H_grid = H_out/TILE_WIDTH; 	// number of vertical tiles per output map
    //int Y = H_grid * W_grid; //The second (y) dimension in the grid maps to the tiles in the output feature maps
    //int C = 1, K = 5;
    bz = ceil((float)28/TILE_WIDTH)*ceil((float)28/TILE_WIDTH);
    dim3 gridDim(1, 6, bz);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    ConvLayerForward_Kernel_1<<<gridDim,blockDim>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight, l_c1.bias, 1, 28, 28, 24, 5, 6);
    apply_sigmoid <<<64,64>>>(l_c1.preact, l_c1.output, l_c1.bytes);
   

    // fp_preact_c1<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
	// fp_bias_c1<<<64, 64>>>((float (*)[24][24])l_c1.preact, l_c1.bias);
	// apply_sigmoid<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.bytes);

    // float *result = (float *)malloc(sizeof(float) * 24*24*6);

    // cudaMemcpy(result,
	// 	l_c1.preact,
	// 	24*24*6 * sizeof(float),
	// 	cudaMemcpyDeviceToHost);
    

    // for (int i = 0; i < 6; i++){
    //     for (int j = 0; j <24; j++){
    //         for (int z = 0; z < 24; z++){
    //             printf("%.2f",*(result + i+j+z));
    //         }
    //         printf("\n");
    //     }

    //     printf("-----------------------------------\n");
    // }

    


    // for pooling layer example:
    bz = ceil((float)6/TILE_WIDTH)*ceil((float)6/TILE_WIDTH);
    dim3 gridDimPool(1, 6, bz);
    dim3 blockDimPool(TILE_WIDTH, TILE_WIDTH, 1);
    //MaxPool2dForward_Kernel_1<<<gridDimPool,blockDimPool>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, 24, 24, 6, 4);
    MaxPool2dForward_Kernel_1<<<gridDimPool,blockDimPool>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight, l_s1.bias ,24, 24, 6, 4);

    apply_sigmoid <<<64,64>>>(l_s1.preact, l_s1.output, l_s1.bytes);



    

    // float *result = (float *)malloc(sizeof(float) * 24*24*6);

    // cudaMemcpy(result,
	// 	l_c1.output,
	// 	24*24*6 * sizeof(float),
	// 	cudaMemcpyDeviceToHost);
    

    // for (int i = 0; i < 6; i++){
    //     for (int j = 0; j <24; j++){
    //         for (int z = 0; z < 24; z++){
    //             printf("%.2f  ",*(result + i+j+z));
    //         }
    //         printf("\n");
    //     }

    //     printf("-----------------------------------\n");
    // }

    // float *pool_result = (float *)malloc(sizeof(float) * 6*6*6);

    // cudaMemcpy(pool_result,
	// 	l_s1.preact,
	// 	6*6*6 * sizeof(float),
	// 	cudaMemcpyDeviceToHost);
    

    // for (int i = 0; i < 6; i++){
    //     for (int j = 0; j <6; j++){
    //         for (int z = 0; z < 6; z++){
    //             printf("%.2f  ",*(pool_result + i+j+z));
    //         }
    //         printf("\n");
    //     }

    //     printf("-----------------------------------\n");
    // }
    // printf("===========================\n\n");








    // for fully connected layer
    bz = ceil((float)10/TILE_WIDTH);
    dim3 gridDimfc(1, 10, 1);
    dim3 blockDimfc(6, 6, 6);

    FullyConLayerForward_kernel<<<gridDimfc,blockDimfc>>>((float (*)[6][6])l_s1.output, (float (*)[6][6][6])l_f.weight, l_f.preact, l_f.bias, 1, 6, 10, 1, 10);
    

    // int Output_width = 10;
    // int Output_height = 1;
    // dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
	// dim3 numBlocks(ceil((float)Output_width/TILE_WIDTH),ceil((float)Output_height/TILE_WIDTH));//bx = O_WIDTH, by = O_HEIGH
    // gemm_h_bias<<<numBlocks,threadsPerBlock>>>((float (*)[6][6])l_s1.output, (float (*)[6][6][6])l_f.weight, l_f.preact, l_f.bias, 1, 6, 10, 1, 10);

    
    
    // fp_preact_s1<<<64, 64>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
	// fp_bias_s1<<<64, 64>>>((float (*)[6][6])l_s1.preact, l_s1.bias);
	// apply_sigmoid<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.bytes);

	// fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
	// fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
	apply_sigmoid<<<64, 64>>>(l_f.preact, l_f.output, l_f.bytes);





    // float *result = (float *)malloc(sizeof(float) * 10);

    // cudaMemcpy(result, l_f.preact, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    

    // printf("ConvLayerForward_Kernel: \n");
    // for (int i = 0; i < 10; i++){
    //     printf("%.2f ",*(result + i));
    // }
    // printf("\n-----------------------------------\n");


    // printf("===========================\n\n");
    // printf("l_f.weight\n");
    // float *fwr = (float *)malloc(sizeof(float) * 10*6*6*6);

    // cudaMemcpy(fwr,
	// 	l_f.weight,
	// 	10*6*6*6 * sizeof(float),
	// 	cudaMemcpyDeviceToHost);
    

    // for (int lk = 0; lk < 10; lk++){
    //     for (int llk = 0; llk < 6; lli++){
    //         for (int lj = 0; lj <6; lj++){
    //             for (int lz = 0; lz < 6; lz++){
    //                 // printf("%.2f  ",*(fwr + lk*10*6*6 + llk*6*6 + lj*6 +lz));
    //                 printf("%.2f  ",*(fwr + lk+ llk+ lj+lz));
    //             }
    //             printf("\n");
    //         }

    //         printf("-----------------%d, %d------------------\n", lk, llk);
    //     }
    //     printf("******************* %d ******************* \n", lk);
    // }
    // printf("===========================\n\n");

}

void backward(){
    // int bz = ceil((float)10/TILE_WIDTH);
    // dim3 gridDimfc(1, 10, 1);
    // dim3 blockDimfc(6, 6, 6);

    // FullyConLayerBackward_kernel<<<gridDimfc,blockDimfc>>>(
    //         l_f.output,
    //         l_f.d_preact,
    //         (float (*)[6][6]) l_s1.preact,
    //         (float (*)[6][6][6]) l_f.d_weight,
    //         l_f.bias
    //     );
    
    // //pooling backward:
    // dim3 gridDimPool(TILE_WIDTH,TILE_WIDTH);
	// bz = ceil((float)24/TILE_WIDTH)*ceil((float)24/TILE_WIDTH);
	// if( bz == 0 )bz = 1;
	// dim3 blockDimPool(1, 6, bz);
    // //input_pointer, Inputimage_height, Inputimage_width, output_pointer, Outputimage_channel, pool_size
    // poolingLayer_backward_GPU<<<gridDimPool,blockDimPool>>>((float (*)[24][24])l_c1.d_output, 24, 24, (float (*)[6][6])l_s1.preact, 6,  4);
    
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


    bp_weight_f<<<64, 64>>>((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
	bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);

	bp_output_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
	bp_preact_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
	bp_weight_s1<<<64, 64>>>((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
	bp_bias_s1<<<64, 64>>>(l_s1.bias, (float (*)[6][6])l_s1.d_preact);

	bp_output_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
	bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
	bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
	bp_bias_c1<<<64, 64>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);


    apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
	apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);


}

static void learn(){

    printf("test 2\n");
    for(int i=0; i< train_cnt - 10; i++){
    //for(int i=0; i<10; i++){
    //     printf("label: %d \n", train_set[i].label);

        l_f.bp_clear();
		l_s1.bp_clear();
		l_c1.bp_clear();
        
        forward(train_set[i].data);
        makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
        backward();

     }

     for (int i = train_cnt - 10; i < train_cnt; i++){
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
    
    test();
    printf("finish\n");

    return 0;
}