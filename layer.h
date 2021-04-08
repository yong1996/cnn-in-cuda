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

class CNN_layer{
    private:
        int width;
        int height;
        int depth;

        float *bias;
        float *weigth;
    public:
        CNN_layer(int in_width, int in_height, int in_depth);
        ~CNN_layer(); // free the memory allocation

        void setOutput;
        void clear;
        void bp_clear;
}

CNN_layer::CNN_layer(int width, int height, int depth): width(in_width), height(in_height), depth(in_depth){

}










