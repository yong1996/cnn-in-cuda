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



























#endif