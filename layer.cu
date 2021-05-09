#include "layer.h"


#define TILE_WIDTH 16


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

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * bytes);
	cudaMemset(preact, 0x00, sizeof(float) * bytes);
}


void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * bytes);
	cudaMemset(d_preact, 0x00, sizeof(float) * bytes);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}


__device__ float sigmoid(float s){
    return 1/(1 + exp(-s));
}

__global__ void apply_sigmoid(float *input, float *output, const int N){
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;
    // TODO:
    for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = sigmoid(input[idx]);
	}
}

__global__ void backward_sigmoid(float* X, int size_in)
{
	int t = blockIdx.x * 1024 + threadIdx.x;

	if(t < size_in)
	{
		double tmp = 1 / (1 + exp(-X[t]));
		tmp = (1-tmp)*tmp;
		X[t] = X[t]*tmp;
	}
}

// softmax
__global__ void softmax(float *error, float *output, unsigned int label, unsigned int size){
	int tid = threadIdx.x;
	
}

__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] += dt * grad[idx];
	}
}



#define TILE_WIDTH 16

//input_pointer,  Output_pointer, W_pointer, Inputimage_channel, Inputimage_height, Inputimage_width , Outputimage_width, W_width_height, Outputimage_channel
__global__ void ConvLayerForward_Kernel_1(float input[28][28], float output[6][24][24], float weight[6][5][5], float bias[6], int C, int H_in, int W_in, int W_out, int K, int M){

    int H_out = H_in - K + 1;
	int n, m, h, w, c, p, q;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;
	n = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;
	float acc = 0;
	for (c = 0; c < C; c++) { // sum over all input channels
		for (p = 0; p < K; p++) // loop over KxK filter
			for (q = 0; q < K; q++)
				if(h < H_out && w < W_out)
                    acc += input[h+p][w+q] * weight[m][p][q];
					//acc = acc + X[n*(C*H_in*W_in) + c*(H_in*W_in) + (h+p)*(W_in) + (w+q)] * W[m*(C*K*K) + c*(K*K) + p*(K) + q];
	}
	__syncthreads();
	if(h < H_out && w < W_out)
	{
        output[m][h][w] = acc + bias[m];
    }
}


// input_pointer, output_pointer, inputimage_height, inputimage_width, outputimage_channel, pool_size 
__global__ void MaxPool2dForward_Kernel_1(float input[6][24][24], float output[6][6][6], float weight[1][4][4], float bias[1] ,int H_in, int W_in, int M, int pool_size){
    int n, m, h, w, p, q;
	int H_out = H_in/pool_size;
	int W_out = W_in/pool_size;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0){
        W_grid = 1;
    }
		
	n = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;
	//h and w is not center point of calculating, it's upper left corner point of Input image
	float acc = 0;
	for (p = 0; p < pool_size; p++) { // loop over KxK input samples
		for (q = 0; q < pool_size; q++)
			if(h < H_out && w < W_out)
				// acc = acc + input[n*(M*H_in*W_in)+ m*(H_in*W_in) +
				//               (pool_size * h + p)*(W_in) + (pool_size * w + q)] / (pool_size * pool_size);
                acc = acc + input[m][pool_size * h+p][pool_size * w+q] * weight[0][p][q];
	}
	__syncthreads();
	if(h < H_out && w < W_out)
	{
		// Y[n*(M*H_out*W_out)+ m*(H_out*W_out) + h*(W_out) + w] = acc;
		output[m][h][w] = acc + bias[0];

	}
}




// input_height, input_width, weight_width, output_height, output_width
//      1             6          10          1              10
__global__ void FullyConLayerForward_kernel(float input[6][6][6], float weight[10][6][6][6], float output[10], float bias[10], int H_in, int W_in, int W_we , int H_out, int W_out) {
    int n, m, h, w, y, p, q, o;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;

	n = blockIdx.x;
	m = blockIdx.y;  // 10
	h = threadIdx.x;  // 6
	w = threadIdx.y;  // 6
	y = threadIdx.z;  // 6

	float Pvalue = 0;
	for (o = 0; o < 6; o++) {
		for (p = 0; p < 6; p++) {
			for (q = 0; q < 6; q++){
				if(h < 6 && w < 6 && y < 6)
				// Pvalue += input[y][h+p][w+q] * weight[m][y][h+p][w+q];
				// Pvalue += input[h][w][y] * weight[m][h+o][w+p][y+q];
				Pvalue += input[h][w][y] * weight[m][o][p][q];
			}
		}
	}
	__syncthreads();

    if(m < W_out)
		output[m] += Pvalue + bias[m]; // Output
}



__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
	}
}

__global__ void fp_bias_f(float preact[10], float bias[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		preact[idx] += bias[idx];
	}
}


__global__ void FullyConLayerBackward_kernel(
	float lf_output[10],
	float l_f_d_preact[10],
	float ls1_preact[6][6][6],
	float lf_weight[10][6][6][6],
	float lf_d_weight[10][6][6][6],
	float lf_bias[10]
	) {

	int n, m, h, w, y, p, q, o;

	n = blockIdx.x;
	m = blockIdx.y;  // 10
	h = threadIdx.x;  // 6
	w = threadIdx.y;  // 6
	y = threadIdx.z;  // 6

	l_f_d_preact[m] *= lf_output[m] * (1- lf_output[m]);
	
	// ls1_d_preact[m] = l_f_d_preact[m] * lf_output[m] * (1- lf_output[m]);

	lf_bias[m] += dt + l_f_d_preact[m];
	__syncthreads();


	lf_d_weight[m][h][w][y] = l_f_d_preact[m] * ls1_preact[h][w][y] ;
	lf_d_weight[m][h][w][y] += lf_weight[m][h][w][y];
	__syncthreads();
}


//input_pointer, Inputimage_height, Inputimage_width, output_pointer, Outputimage_channel, pool_size
__global__ void poolingLayer_backward_GPU(float input[6][24][24], int H_in, int W_in, float output[6][6][6], int M, int pool_size)

{
	int n, m, h, w, p, q;
	int H_out = H_in/pool_size;
	int W_out = W_in/pool_size;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;
	n = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;

	//h and w is not center point of calculating, it's upper left corner point of Input image
	float acc = 0;
	for (p = 0; p < pool_size; p++) { // loop over KxK input samples
		for (q = 0; q < pool_size; q++)
			if(h < H_out && w < W_out)
			input[m][h+p][w+q] = output[m][h][w] / (pool_size * pool_size);
	}
	__syncthreads();

}



__global__ void ConvLayerBackward_Kernel(
	float input[28][28], 
	float d_output[6][24][24], 
	float preact[6][24][24], 
	float d_preact[6][24][24], 
	float d_weight[6][5][5], 
	int C, int H_in, int W_in, int W_out, int K, int M) {

    int H_out = H_in - K + 1;
	int n, m, h, w, c, p, q;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;
	n = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;

	float d = 24.0f * 24.0f;

	float o = sigmoid(preact[m][h][w]);
	
	// float dv = d_output[m][h][w] * o * (1 - o);
	d_preact[m][h][w] = d_output[m][h][w] * o * (1 - o);
	__syncthreads();

	for (c = 0; c < C; c++) {
		for (p = 0; p < K; p++) {
			for (q = 0; q < K; q++) {
				if(h < H_out && w < W_out) {
					d_weight[m][p][q] = d_preact[m][h][w] * input[28][28]/d;
				}
			}
		}
	}
}




// bp
/*
// =========================

__global__ void bp_f_s1(
	float l_f_d_weight[10][6][6][6],
	float l_f_d_preact[10],
	float l_f_bias[10],
	float l_f_weight[10][6][6][6],
	float l_s1_output[6][6][6],
	float l_s1_d_output[6][6][6],
	float l_s1_d_preact[6][6][6],
	float l_s1_bias[1]
){
	int n, m, h, w, y, p, q, o;
	n = blockIdx.x;
	m = blockIdx.y;  // 10
	h = threadIdx.x;  // 6
	w = threadIdx.y;  // 6
	y = threadIdx.z;  // 6


	l_f_d_weight[m][h][w][y] = l_f_d_preact[m] * l_s1_output[h][w][y];
	l_s1_d_output[h][w][y] += l_f_weight[m][h][w][y] * l_f_d_preact[m];

	if(h==0 && w==0 && y==0 )
		l_f_bias[m] += dt * l_f_d_preact[m];


	// if(m == 1) {
	// 	l_s1_d_preact[h][w][y] = l_s1_d_output[x][w][y] * l_s1_output[x][w][y] * (1 - l_s1_output[x][w][y]);
	// 	__syncthreads();

	// 	bias[0] += dt * l_s1_d_preact[h][w][y]/(6*6*6);
	// }
}

__global__ void bp_s1(
	float l_s1_output[6][6][6],
	float l_s1_d_output[6][6][6],
	float l_s1_d_preact[6][6][6],
	float l_s1_d_weight[6][4][4],
	float l_c1_output[6][24][24],
	float l_c1_d_output[6][24][24]

){
	int n, m, h, w, y, p, q, o;
	n = blockIdx.x;
	m = blockIdx.y;  // 6
	h = threadIdx.x;  // 6
	w = threadIdx.y;  // 6
	y = threadIdx.z;  // -



	l_s1_d_preact[m][h][w] = l_s1_d_output[m][h][w] * l_s1_output[m][h][w] * (1 - l_s1_output[m][h][w]);
	__syncthreads();

	bias[0] += dt * l_s1_d_preact[m][h][w]/(6*6*6);

	for(int i=0; i<4; i++) {
		l_s1_d_weight[m][h][w] += l_s1_d_preact[m][h][w] * l_c1_output[m][h*4+i][w*4+i];
		l_c1_d_output[m][h*4+i][w*4+i], n_weight[m][h][w] * nd_preact[m][h][w];
	}
}


__global__ void bp_s1(
	float l_c1_preact[6][24][24],
	float l_c1_d_preact[6][24][24],
	float l_c1_d_output[6][24][24],
	float l_c1_d_weight[6][5][5],
	float l_input_output[28][28],
	float l_c1_bias[6]

){
	int n, m, h, w, y, p, q, o;
	n = blockIdx.x;
	m = blockIdx.y;  // 6
	h = threadIdx.x;  // 24
	w = threadIdx.y;  // 24
	y = threadIdx.z;  // -


	float o = sigmoid(l_c1_preact[m][h][w]);
	l_c1_d_preact[m][h][w] = l_c1_d_output[m][h][w] * o * (1 - o);

	for(int i=0; i<5; i++){
		for(int j=0; j<5; j++){
			l_c1_d_weight[m][i][j] += l_c1_d_preact[m][h][w] * l_input_output[h + i][w + j] / (24*24);
		}
	}

	l_c1_bias[i1], dt * l_c1_d_preact[m][h][w] / (6*24*24);
}


// =================================

*/


__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 5*5*6*24*24;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 5);
		const int i2 = ((idx /= 5	) % 5);
		const int i3 = ((idx /= 5	) % 6);
		const int i4 = ((idx /= 6	) % 24);
		const int i5 = ((idx /= 24	) % 24);

		atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
	}
}

__global__ void fp_bias_c1(float preact[6][24][24], float bias[6])
{
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

__global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4*4*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 4);
		const int i2 = ((idx /= 4	) % 4);
		const int i3 = ((idx /= 4	) % 6);
		const int i4 = ((idx /= 6	) % 6);
		const int i5 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
	}
}

__global__ void fp_bias_s1(float preact[6][6][6], float bias[1])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		preact[i1][i2][i3] += bias[0];
	}
}


__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
	}
}

__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		bias[idx] += dt * d_preact[idx];
	}
}

__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
	}
}

__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		const float o = sigmoid(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*4*4*6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 4);
		const int i3 = ((idx /= 4	) % 4);
		const int i4 = ((idx /= 4	) % 6);
		const int i5 = ((idx /= 6	) % 6);
		const int i6 = ((idx /= 6	) % 6);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
	}
}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d);
	}
}

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*4*4*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 4);
		const int i3 = ((idx /= 4	) % 4);
		const int i4 = ((idx /= 4	) % 6);
		const int i5 = ((idx /= 6	) % 6);
		const int i6 = ((idx /= 6	) % 6);

		atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
	}
}

__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		const float o = sigmoid(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*5*5*24*24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 5);
		const int i3 = ((idx /= 5	) % 5);
		const int i4 = ((idx /= 5	) % 24);
		const int i5 = ((idx /= 24	) % 24);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
	}
}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);
	}
}
