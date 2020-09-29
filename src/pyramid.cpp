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
	size_t curr_w = width;
	size_t curr_h = height;
	laplace_imgs.emplace_back(curr_w, curr_h);
	for(unsigned i = 0; i < levels - 1; i++){
		curr_w /= 2;
		curr_h /= 2;
	}
	for(unsigned i = 0; i < levels - 1; i++){
		curr_w *= 2;
		curr_h *= 2;
	}
	for(unsigned i = 1; i < levels; i++){
		curr_w /= 2;
		curr_h /= 2;
		laplace_imgs.emplace_back(curr_w, curr_h);
	}

	tmp_blurred.init_to(128, 128, 128, 255);
}

void Pyramid::construct_from(const Image_cuda& src,
			size_t x, size_t y, size_t w, size_t h,
			const Cuda_stream& stream)
{
	cudaError_t res;

	copy_image(&tmp, &src, 0, 0, x, y, w, h, stream);

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

	auto& base = laplace_imgs[levels-1];
	//copy_image(&base, &tmp, 0, 0, 0, 0, base.get_width(), base.get_height(), stream);
}

void Pyramid::reconstruct_to(Image_cuda *dst,
		size_t x, size_t y, size_t w, size_t h,
		const Cuda_stream& stream)
{
	cudaError_t res;
#if 1
	auto& base = laplace_imgs[levels-1];

	int curr_w = w;//base.get_width();
	int curr_h = h;//base.get_height();

	for(int i = 0; i < levels - 1; i++){
		curr_w /= 2;
		curr_h /= 2;
	}

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

	copy_image(dst, &tmp, x, y, 0, 0, w, h, stream);

#else
	const int lvl_idx = 0;
	for(int i = 0; i < lvl_idx; i++){
		w /= 2;
		h /= 2;
	}

	auto& lvl = laplace_imgs[lvl_idx];
	//cuda_upsample(&lvl, &laplace_imgs[1], w / 2, h / 2, stream.get());
	//cuda_gaussian_blur(&lvl, 0, 0, w, h, stream.get());
	copy_image(dst, &lvl, x, y, 0, 0, w, h, stream);
#endif
}

}
