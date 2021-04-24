#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

// includes, system
#include <string>


#ifndef LAYER_H
#define LAYER_H



//const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer{
    public:
        int M;
        int N;
        int bytes;

        float *output;
        float *preact;

        float *bias;
        float *weight;

        float *d_output;
        float *d_preact;
        float *d_weight;

        Layer(int in_width, int in_height, int bytes);
        ~Layer(); // free the memory allocation

        void setInput(float *data); // set data
        void clear();     //reset the GPU memory
        void bp_clear();  
};

// Layer::Layer(int in_width, int in_height, int in_depth): width(in_width), height(in_height), size(in_size){

// }

__device__ float sigmoid(float v);
__global__ void apply_sigmoid(float *input, float *output, const int N);

//__global__ void ConvLayerForward_Kernel_1(int C, int W_grid, int K, float input[28][28], float output[6][24][24], float weight[6][5][5]);
__global__ void ConvLayerForward_Kernel_1(float input[28][28], float output[6][24][24], float weight[6][5][5], int C, int H_in, int W_in, int W_out, int K, int M);
__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]);
__global__ void ConvLayerForward_Kernel_bias_1(float input[6][24][24], float bias[1]);


//average pooling
__global__ void MaxPool2dForward_Kernel_1(float input[6][24][24], float output[6][6][6], int H_in, int W_in, int M, int pool_size);
__global__ void MaxPool2dBackward_Kernel_1(int stride, int poolSize, float input[6][24][24], float output[6][6][6]);



// FullyConnect
__global__ void FullyConLayerForward_kernel(float input[6][6][6], float weight[10][6][6][6], float output[10], float bias[10], int H_in, int W_in, int W_we , int H_out, int W_out);
__global__ void FullyConLayerBackward_kernel(float input[6][6][6], float weight[10][6][6][6], float output[10], float bias[10], int H_in, int W_in, int W_we , int H_out, int W_out);


#endif

