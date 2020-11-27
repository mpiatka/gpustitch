#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include "gpustitch_common.hpp"
#include "image.hpp"

void cuda_image_memset(gpustitch::Image_cuda *img,
		int x, int y, int w, int h,
		int r, int g, int b, int a,
		CUstream_st *stream);

#endif
