#include "layer.h"

#ifndef MAXPOOL
#define MAXPOOL


class MAXPOOL2D : public Layer{
    private:
        int kernel_h;
        int kernel_w;
        int pad_h;
        int pad_w;
        int stride;
    public:
        MAXPOOL2D (int kernel_h, int kernel_w, int pad_h, int pad_w, int stride)
      : kernel_h(kernel_h),
        kernel_w(kernel_w),
        pad_h(pad_h),
        pad_w(pad_w),
        stride_h(stride) {}

        void forward();
        void backward();
};

__global__ void MaxPool2dForward_Kernel_1(int stride, int poolSize, float input[6][24][24], float output[6][6][6]);
__global__ void MaxPool2dBackward_Kernel_1(int stride, int poolSize, float input[6][24][24], float output[6][6][6]);




#endif























#endif