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
			CUstream_st *stream)
{
	cudaError_t res;

	unsigned char *src_p = (unsigned char *)(src.data())
		+ y * src.get_pitch()
		+ x * src.get_bytes_per_px();

	res = cudaMemcpy2DAsync(tmp.data(), tmp.get_pitch(),
			src_p, src.get_pitch(),
			w * 4, h,
			cudaMemcpyDeviceToDevice,
			stream);

	for(unsigned i = 0; i < levels - 1; i++){
		auto& laplacian = laplace_imgs[i];

		res = cudaMemcpy2DAsync(tmp_blurred.data(), tmp_blurred.get_pitch(),
				tmp.data(), tmp.get_pitch(),
				laplacian.get_width() * 4, laplacian.get_height(),
				cudaMemcpyDeviceToDevice,
				stream);

		cuda_gaussian_blur(&tmp_blurred,
				0, 0,
				laplacian.get_width(), laplacian.get_height(),
				stream);

		cuda_subtract_images(&tmp, &tmp_blurred,
				&laplacian, laplacian.get_width(), laplacian.get_height(),
				stream);

		cuda_downsample(&tmp, &tmp_blurred,
				laplacian.get_width(), laplacian.get_height(),
				stream);
	}

	auto& base = laplace_imgs[levels-1];
	res = cudaMemcpy2DAsync(base.data(), base.get_pitch(),
			tmp.data(), tmp.get_pitch(),
			base.get_width() * 4, base.get_height(),
			cudaMemcpyDeviceToDevice,
			stream);
}

void Pyramid::reconstruct_to(Image_cuda *dst,
		size_t x, size_t y,
		CUstream_st *stream)
{
	cudaError_t res;
#if 1
	auto& base = laplace_imgs[levels-1];

	res = cudaMemcpy2DAsync(tmp.data(), tmp.get_pitch(),
			base.data(), base.get_pitch(),
			base.get_width() * 4, base.get_height(),
			cudaMemcpyDeviceToDevice,
			stream);

	int w = base.get_width();
	int h = base.get_height();

	for(unsigned i = levels - 2; i > 0; i--){
		cuda_upsample(&tmp_blurred, &tmp, w, h, stream);
		w *= 2;
		h *= 2;
		cuda_gaussian_blur(&tmp_blurred, 0, 0, w, h, stream);
		cuda_add_images(&tmp_blurred, &laplace_imgs[i], &tmp, w, h, stream);
	}

	cuda_upsample(&tmp_blurred, &tmp, w, h, stream);
	w *= 2;
	h *= 2;
	cuda_gaussian_blur(&tmp_blurred, 0, 0, w, h, stream);
	cuda_add_images(&tmp_blurred, &laplace_imgs[0], dst, w, h, stream);

#else
	/*
	res = cudaMemcpy2DAsync(dst->data(), dst->get_pitch(),
			laplace_imgs[0].data(), laplace_imgs[0].get_pitch(),
			laplace_imgs[0].get_width() * 4, laplace_imgs[0].get_height(),
			cudaMemcpyDeviceToDevice,
			stream);
			*/
	cuda_upsample(dst, &laplace_imgs[1], 1024, 1024, stream);
#endif
}

}
