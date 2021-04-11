#include "layer.h"


Layer::Layer(int in_width, int in_height, int in_depth): width(in_width), height(in_height), depth(in_depth){

    float h_bias[in_height];
    float h_weight[N][M];

    output = NULL;
    preact = NULL;
    bias = NULL:
    weight = NULL;

    for (int i = 0; i < height; i++){
        h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);  // initial bias
        for (int j = 0; j < width; j++){
            h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);  // initial weight
        }
    }

    //TODO: cudaMalloc


}


// de-constructor
Layer::~Layer(){

    // TODO: free cuda memory

}


