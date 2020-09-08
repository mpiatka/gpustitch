#include <cuda_runtime.h>
#include "pyramid.hpp"
#include "gauss_laplace.h"

namespace gpustitch{

Pyramid::Pyramid(size_t w, size_t h, unsigned levels) : 
	width((w/2)*2),
	height((h/2)*2),
	levels(levels),
	tmp(width, height),
	tmp_blurred(width, height)
{
	size_t curr_w = width;
	size_t curr_h = height;
	for(unsigned i = 0; i < levels; i++){
		laplace_imgs.emplace_back(curr_w, curr_h);
		curr_w /= 2;
		curr_h /= 2;
	}
}

void Pyramid::construct(const Image_cuda& src,
			size_t x, size_t y, size_t w, size_t h,
			CUstream_st *stream)
{
	cudaError_t res;

	res = cudaMemcpy2DAsync(tmp.data(), tmp.get_pitch(),
			src.data(), src.get_pitch(),
			w, h,
			cudaMemcpyDeviceToDevice,
			stream);

	for(unsigned i = 0; i < levels - 1; i++){
		auto& laplacian = laplace_imgs[i];

		res = cudaMemcpy2DAsync(tmp_blurred.data(), tmp_blurred.get_pitch(),
				tmp.data(), tmp.get_pitch(),
				laplacian.get_width(), laplacian.get_height(),
				cudaMemcpyDeviceToDevice,
				stream);

		cuda_gaussian_blur(&tmp_blurred,
				0, 0,
				laplacian.get_width(), laplacian.get_height(),
				stream);

		cuda_subtract_images(&tmp, &tmp_blurred, &laplacian, stream);

		//cuda_downsample(tmp_blurred, tmp);

	}

	auto& laplacian = laplace_imgs[levels-1];
	res = cudaMemcpy2DAsync(laplacian.data(), laplacian.get_pitch(),
			tmp.data(), tmp.get_pitch(),
			laplacian.get_width(), laplacian.get_height(),
			cudaMemcpyDeviceToDevice,
			stream);


}

}
