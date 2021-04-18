#include "layer.h"

#define MASK_WIDTH = 3;

// Layer constructor:
Layer::Layer(int in_width, int in_height, int in_size): M(in_width), N(in_height), bytes(in_size){

    float h_bias[N];
    float h_weight[N][M];


    output = NULL;
    preact = NULL;
    bias = NULL:
    weight = NULL;

    for (int i = 0; i < N; i++){
        h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);  // initial bias
        for (int j = 0; j < M; j++){
            h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);  // initial weight
        }
    }

    cudaMalloc(&output, sizeof(float) * bytes);
	cudaMalloc(&preact, sizeof(float) * bytes);

	cudaMalloc(&bias, sizeof(float) * N);
	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * bytes);
	cudaMalloc(&d_preact, sizeof(float) * bytes);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// de-constructor
Layer::~Layer(){

    // TODO: free cuda memory
    cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);

}

void Layer:: setInput(float *data){
    cudaMemcpy(output, data, sizeof(float)*bytes, cudaMemcpyHostToDevice);
}

// void Layer:: clear(){

// }
// void Layer:: bp_clear(){

// }


__device__ float sigmoid(float v){
    return 1/(1 + exp(-v));
}

__global__ void apply_sigmoid(float *input, float *output, const int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;
    // TODO:

    for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = sigmoid(input[idx]);
	}


}

// convLayer 1 the weight is 6*3*3  output is 6*24*24
__global__ void ConvLayerForward_Kernel(int C, int W_grid, int K, float input[28][28], float output[6][24][24], float weight[6][5][5]){

    
    // int m, h, w, c, q, p;
    // float Pvalue = 0;
    // int n_start_point = i - (MASK_WIDTH/2);

    // for (int m = 0; m < 6; m++){          // for each output feature map
    //     for (int h = 0; h < 27; h++ ){
    //         for(int w = 0; w < 27; w++){
    //             output[m][h][w] = 0;
    //             for (int c = 0; c < 1; c++){ // for each channel
    //                 for (int p = 0; p < MASK_WIDTH; p++){
    //                     for (int q = 0; q < MASK_WIDTH; q++){
    //                         output[m][h][w] += input[h + p][w + q] * weight[m][p][q];
    //                     }
    //                 }
    //             }
    //         }

    //     }

    // }

    int m = blockIdx.x;
    int h =  blockIdx.y / W_grid  + threadIdx.y;
    int w = blockIdx.y % W_grid + threadIdx.x;
    float acc = 0.;
    //for (int c = 0;  c < C; c++) {		// sum over all input channels, in this case, the channel is 1
       for (int p = 0; p < K; p++)		// loop over KxK  filter
          for (int q = 0; q < K; q++)  
             acc += X[h+p][w+q] * weight[m][p][q];
    //}
    output[m][h][w] = acc;

}


