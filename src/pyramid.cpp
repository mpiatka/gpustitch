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

void Pyramid::construct_from(const Image_cuda& src,
			size_t x, size_t y, size_t w, size_t h,
			const Cuda_stream& stream)
{
	cudaError_t res;

	unsigned char *src_p = (unsigned char *)(src.data())
		+ y * src.get_pitch()
		+ x * src.get_bytes_per_px();


	copy_image(&tmp, &src, 0, 0, x, y, w, h, stream);

	for(unsigned i = 0; i < levels - 1; i++){
		auto& laplacian = laplace_imgs[i];
		auto& laplacian_next = laplace_imgs[i + 1];

		copy_image(&tmp_blurred, &tmp, 0, 0, 0, 0,
				laplacian.get_width(), laplacian.get_height(),
				stream);

		cuda_gaussian_blur(&tmp_blurred,
				0, 0,
				laplacian.get_width(), laplacian.get_height(),
				stream.get());

		cuda_downsample(&laplacian_next, &tmp_blurred,
				laplacian.get_width(), laplacian.get_height(),
				stream.get());

		cuda_upsample(&tmp_blurred, &laplacian_next,
				laplacian_next.get_width(), laplacian_next.get_width(),
				stream.get());

		cuda_gaussian_blur(&tmp_blurred,
				0, 0,
				laplacian.get_width(), laplacian.get_height(),
				stream.get());

		cuda_subtract_images(&tmp, &tmp_blurred,
				&laplacian, laplacian.get_width(), laplacian.get_height(),
				stream.get());

		copy_image(&tmp, &laplacian_next, 0, 0, 0, 0,
				laplacian_next.get_width(), laplacian_next.get_height(),
				stream);
	}

	auto& base = laplace_imgs[levels-1];
	copy_image(&base, &tmp, 0, 0, 0, 0, base.get_width(), base.get_height(), stream);
}

void Pyramid::reconstruct_to(Image_cuda *dst,
		size_t x, size_t y, size_t w, size_t h,
		const Cuda_stream& stream)
{
	cudaError_t res;
#if 1
	auto& base = laplace_imgs[levels-1];

	copy_image(&tmp, &base, x, y, 0, 0, base.get_width(), base.get_height(), stream);

	int curr_w = base.get_width();
	int curr_h = base.get_height();

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
	cuda_add_images(&tmp_blurred, &laplace_imgs[0], dst, w, h, stream.get());

#else
	res = cudaMemcpy2DAsync(dst->data(), dst->get_pitch(),
			laplace_imgs[0].data(), laplace_imgs[0].get_pitch(),
			laplace_imgs[0].get_width() * 4, laplace_imgs[0].get_height(),
			cudaMemcpyDeviceToDevice,
			stream);
	//cuda_upsample(dst, &laplace_imgs[1], 1024, 1024, stream);
#endif
}

}
