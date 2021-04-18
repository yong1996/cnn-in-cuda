#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

// includes, system
#include <string>


#ifndef LAYER_H
#define LAYER_H
#endif


const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer{
    public:
        int M;
        int N;
        int bytes;

        float *bias;
        float *weigth;

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
__global__ void ConvLayerForward_Kernel(int C, int W_grid, int K, float *input[28][28], float output[6][24][24], float weight[6][5][5]);










