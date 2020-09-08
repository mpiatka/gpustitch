#ifndef GAUSS_LAPLACE_H
#define GAUSS_LAPLACE_H

#include "gpustitch_common.hpp"
#include "image.hpp"

void cuda_gaussian_blur(const gpustitch::Image_cuda *img, int start_x, int start_y,
		int w, int h,
		CUstream_st *stream);

void cuda_subtract_images(const gpustitch::Image_cuda *a,
		const gpustitch::Image_cuda *b,
		gpustitch::Image_cuda *result,
		CUstream_st *stream);

void cuda_add_images(const gpustitch::Image_cuda *a,
		const gpustitch::Image_cuda *b,
		gpustitch::Image_cuda *result,
		CUstream_st *stream);

void cuda_downsample(gpustitch::Image_cuda dst,
		const gpustitch::Image_cuda src,
		CUstream_st *stream);

#endif
