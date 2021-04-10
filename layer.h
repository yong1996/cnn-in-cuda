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

class layer{
    private:
        int width;
        int height;
        int depth;

        float *bias;
        float *weigth;
    public:
        layer(int in_width, int in_height, int in_depth);
        ~layer(); // free the memory allocation

        void setOutput; // set data
        void clear;     //reset the GPU memory
        void bp_clear;  
}

layer::layer(int width, int height, int depth): width(in_width), height(in_height), depth(in_depth){

}










