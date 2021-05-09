#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

// includes, system
#include <string>


#ifndef LAYER_H
#define LAYER_H



const static float dt = 1.0E-01f;
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
__global__ void backward_sigmoid(float* X, int size_in);
__global__ void apply_grad(float *output, float *grad, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);

//__global__ void ConvLayerForward_Kernel_1(int C, int W_grid, int K, float input[28][28], float output[6][24][24], float weight[6][5][5]);
// __global__ void ConvLayerForward_Kernel_1(float input[28][28], float output[6][24][24], float weight[6][5][5], int C, int H_in, int W_in, int W_out, int K, int M);
__global__ void ConvLayerForward_Kernel_1(float input[28][28], float output[6][24][24], float weight[6][5][5], float bias[6], int C, int H_in, int W_in, int W_out, int K, int M);
__global__ void ConvLayerForward_Kernel_bias_1(float input[6][24][24], float bias[1]);

// __global__ void ConvLayerBackward_Kernel(float input[28][28], float d_output[6][24][24], float output[6][24][24], float weight[6][5][5], float bias[6], int C, int H_in, int W_in, int W_out, int K, int M);
__global__ void ConvLayerBackward_Kernel(
	float input[28][28], 
	float d_output[6][24][24], 
	float preact[6][24][24], 
	float d_preact[6][24][24], 
	float d_weight[6][5][5], 
	int C, int H_in, int W_in, int W_out, int K, int M);

//pooling
//__global__ void MaxPool2dForward_Kernel_1(float input[6][24][24], float output[6][6][6], int H_in, int W_in, int M, int pool_size);
__global__ void MaxPool2dForward_Kernel_1(float input[6][24][24], float output[6][6][6], float weight[1][4][4], float bias[1] ,int H_in, int W_in, int M, int pool_size);
__global__ void poolingLayer_backward_GPU(float input[6][24][24], int H_in, int W_in, float output[6][6][6], int M, int pool_size);

// FullyConnect
__global__ void gemm_h_bias(float input[6][6][6], float weight[10][6][6][6], float output[10], float bias[10], int H_in, int W_in, int W_we , int H_out, int W_out);
__global__ void FullyConLayerForward_kernel(float input[6][6][6], float weight[10][6][6][6], float output[10], float bias[10], int H_in, int W_in, int W_we , int H_out, int W_out);

// __global__ void FullyConLayerBackward_kernel(float input[6][6][6], float weight[10][6][6][6], float output[10], float bias[10], int H_in, int W_in, int W_we);
// __global__ void FullyConLayerBackward_kernel(
// 	float output[6][6][6], 
// 	float weight[10][6][6][6], 
// 	float d_output[10], 
// 	float preact[10], 
// 	float d_preact[10], 
// 	float bias[10], 
// 	int H_in, int W_in, int W_we);

__global__ void FullyConLayerBackward_kernel(
	float lf_output[10],
	float l_f_d_preact[10],
	float ls1_preact[6][6][6],
	float lf_d_weight[10][6][6][6],
	float lf_bias[10]
	);

//test
__global__ void softmax(float *error, float *output, unsigned int label, unsigned int size);
__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]);
__global__ void fp_bias_c1(float preact[6][24][24], float bias[6]);
__global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4]);
__global__ void fp_bias_s1(float preact[6][6][6], float bias[1]);
__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6]);
__global__ void fp_bias_f(float preact[10], float bias[10]);


__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6]);
__global__ void bp_bias_f(float bias[10], float d_preact[10]);
__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]);
__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]);
__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24]);
__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6]);
__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]);
__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]);
__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]);
__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]);

#endif

