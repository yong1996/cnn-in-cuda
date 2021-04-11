#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

// includes, system
#include <string>

// includes, kernels
#include <cnn_kernel.cu>



#ifndef LAYER_H
#define LAYER_H
#endif


const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer{
    public:
        int width;
        int height;
        int number;

        float *bias;
        float *weigth;

        float *d_output;
        float *d_preact;
        float *d_weight;

        Layer(int in_width, int in_height, int in_number);
        ~Layer(); // free the memory allocation

        void setOutput; // set data
        void clear;     //reset the GPU memory
        void bp_clear;  
}

Layer::Layer(int in_width, int in_height, int in_depth): width(in_width), height(in_height), depth(in_depth){

}










