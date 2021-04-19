#include "layer.h"



// Layer constructor:
Layer::Layer(int in_width, int in_height, int in_size): M(in_width), N(in_height), bytes(in_size){

    float h_bias[N];
    float h_weight[N][M];


    output = NULL;
    preact = NULL;
    bias = NULL;
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
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;
    // TODO:
    for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = sigmoid(input[idx]);
	}
}

// convLayer 1 the weight is 6*3*3  output is 6*24*24
// __global__ void ConvLayerForward_Kernel_1(int C, int W_grid, int K, float input[28][28], float output[6][24][24], float weight[6][5][5]){

//     int m = blockIdx.x;
//     int h = blockIdx.y / W_grid * 16 + threadIdx.y;
//     int w = blockIdx.y % W_grid * 16 + threadIdx.x;
//     float acc = 0.;
//     //for (int c = 0;  c < C; c++) {		// sum over all input channels, in this case, the channel is 1
//        for (int p = 0; p < K; p++){
//             for (int q = 0; q < K; q++){
//                 acc += input[h+p][w+q] * weight[m][p][q];
//             }  
//        }		
          
//     //}
//     output[m][h][w] = acc;

// }

#define TILE_WIDTH 16

//input_pointer,  Output_pointer, W_pointer, Inputimage_channel, Inputimage_height, Inputimage_width , Outputimage_width, W_width_height, Outputimage_channel
__global__ void ConvLayerForward_Kernel_1(float input[28][28], float output[6][24][24], float weight[6][5][5], int C, int H_in, int W_in, int W_out, int K, int M){

    int H_out = H_in - K + 1;
	int n, m, h, w, c, p, q;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;
	n = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;
	//h and w is not center point, it's upper left corner point of Input image
	float acc = 0;
	for (c = 0; c < C; c++) { // sum over all input channels
		for (p = 0; p < K; p++) // loop over KxK filter
			for (q = 0; q < K; q++)
				if(h < H_out && w < W_out)
					//acc = acc + input[n*(C*H_in*W_in) + c*(H_in*W_in) + (h+p)*(W_in) + (w+q)] * weight[m*(C*K*K) + c*(K*K) + p*(K) + q];
                    acc += input[h+p][w+q] * weight[m][p][q];
	}
	if(h < H_out && w < W_out)
	{
		//output[n*(M*H_out*W_out) + m*(H_out*W_out) + h*(W_out) + w] = acc;
        output[m][h][w] = acc;
    }


}


__global__ void ConvLayerForward_Kernel_bias_1(float preact[6][24][24], float bias[1]){
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		preact[i1][i2][i3] += bias[i1];
	}
}


