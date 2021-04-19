#include "fn.h"


// __global__ void ConvLayerForward_Kernel_1(float input[28][28], float output[6][24][24], float weight[6][5][5], int C, int H_in, int W_in, int W_out, int K, int M){

// __global__ void FullyConLayerForward_kernel(float* Md, float* Nd, float* Pd, float* B, int M_height_in, int M_width_N_height_in, int N_width_in , int height_out, int width_out) {
__global__ void FullyConLayerForward_kernel(float* Md, float* Nd, float output[6][24][24], float* B, int M_height_in, int M_width_N_height_in, int N_width_in , int height_out, int width_out) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    //width
    for(int m = 0 ; m < ceilf((float)M_width_N_height_in / TILE_WIDTH) ; ++m)
    {
        if(row < M_height_in && (m*TILE_WIDTH + tx) < M_width_N_height_in) // X
            Mds[ty][tx] = Md[row*M_width_N_height_in+(m*TILE_WIDTH + tx)];
        else
            Mds[ty][tx] = 0;
        if((m*TILE_WIDTH + ty) < M_width_N_height_in && col < N_width_in) // W
            Nds[ty][tx] = Nd[(m*TILE_WIDTH + ty)*N_width_in + col];
        else
            Nds[ty][tx] = 0;
        __syncthreads();

        for(int k = 0 ; k < TILE_WIDTH ; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    if(row < height_out && col < width_out)
        output[row*width_out + col] = Pvalue + B[col]; // Output
}
__global__ void FullyConLayerForward_bias();
__global__ void FullyConLayerBackward();
__global__ void FullyConLayerBackward_bias();