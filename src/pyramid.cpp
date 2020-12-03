#include <cuda_runtime.h>
#include "pyramid.hpp"
#include "gauss_laplace.h"

namespace gpustitch{

Pyramid::Pyramid(size_t w, size_t h, unsigned levels) : 
	width(w),
	height(h),
	levels(levels),
	tmp(width, height),
	tmp_blurred(width, height)
{
	if(levels == 0){
		throw "Level count needs to be non-zero";
	}
	size_t curr_w = width;
	size_t curr_h = height;
	laplace_imgs.emplace_back(curr_w, curr_h);

	/* Zeroes out lowest levels-1 bits to make sure that each level is exactly
	 * half the size of the previous one (without remainder in the division) */
	curr_w &= ~((1 << (levels - 1)) - 1);
	curr_h &= ~((1 << (levels - 1)) - 1);

	for(unsigned i = 0; i < levels - 1; i++){
		curr_w /= 2;
		curr_h /= 2;
		laplace_imgs.emplace_back(curr_w, curr_h);
	}

	tmp_blurred.init_to(128, 128, 128, 255);
}

Image_cuda *Pyramid::get_construct_in(){
	return &tmp;
}

void Pyramid::construct(size_t w, size_t h, const Cuda_stream& stream){
	cudaError_t res;

	for(unsigned i = 0; i < levels - 1; i++){
		auto& laplacian = laplace_imgs[i];
		auto& laplacian_next = laplace_imgs[i + 1];
		int curr_w = (laplacian_next.get_width())*2;
		int curr_h = (laplacian_next.get_height())*2;

		copy_image(&tmp_blurred, &tmp, 0, 0, 0, 0,
				curr_w, curr_h,
				stream);

		cuda_gaussian_blur(&tmp_blurred,
				0, 0,
				curr_w, curr_h,
				stream.get());

		cuda_downsample(&laplacian_next, &tmp_blurred,
				curr_w, curr_h,
				stream.get());

		cuda_upsample(&tmp_blurred, &laplacian_next,
				laplacian_next.get_width(), laplacian_next.get_height(),
				stream.get());

		cuda_gaussian_blur(&tmp_blurred,
				0, 0,
				curr_w, curr_h,
				stream.get());

		cuda_subtract_images(&tmp, &tmp_blurred,
				&laplacian, laplacian.get_width(), laplacian.get_height(),
				stream.get());

		copy_image(&tmp, &laplacian_next, 0, 0, 0, 0,
				laplacian_next.get_width(), laplacian_next.get_height(),
				stream);
	}
}

const Image_cuda *Pyramid::get_reconstructed(size_t w, size_t h,
		const Cuda_stream& stream)
{
	cudaError_t res;
	auto& base = laplace_imgs[levels-1];

	int curr_w = w >> (levels - 1);
	int curr_h = h >> (levels - 1);

	tmp_blurred.init_to(128, 128, 128, 255, stream);

	copy_image(&tmp, &base, 0, 0, 0, 0, curr_w, curr_h, stream);

	for(unsigned i = levels - 2; i > 0; i--){
		cuda_upsample(&tmp_blurred, &tmp, curr_w, curr_h, stream.get());
		curr_w *= 2;
		curr_h *= 2;
		cuda_gaussian_blur(&tmp_blurred, 0, 0, curr_w, curr_h, stream.get());
		cuda_add_images(&tmp_blurred, &laplace_imgs[i], &tmp, curr_w, curr_h, stream.get());
	}

	cuda_upsample(&tmp_blurred, &tmp, curr_w, curr_h, stream.get());
	curr_w *= 2;
	curr_h *= 2;
	cuda_gaussian_blur(&tmp_blurred, 0, 0, curr_w, curr_h, stream.get());
	cuda_add_images(&tmp_blurred, &laplace_imgs[0], &tmp, w, h, stream.get());

	return &tmp;
}

}
