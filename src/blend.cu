#include "image.hpp"
#include "gaussian_kernel.hpp"

using namespace gpustitch;

__global__
void kern_blit_overlap(
		const unsigned char *left, int l_start_x, int l_start_y, int l_pitch,
		const unsigned char *right, int r_start_x, int r_start_y, int r_pitch,
		int seam_center, int seam_width, int seam_begin, int overlap_width,
		unsigned char *dst, int dst_x, int dst_y, int dst_pitch,
		int w, int h)
{

	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if(x >= w)
		return;

	if(y >= h)
		return;

	uchar4 *l = (uchar4 *)(left + (l_start_y + y) * l_pitch + (l_start_x + x) * 4);
	uchar4 *r = (uchar4 *)(right + (r_start_y + y) * r_pitch + (r_start_x + x) * 4);
	const float vert_angle = fabsf(y - h/2) * 3.14f/h;
	const int s_w = fminf(seam_width * (1 / cosf(vert_angle)), overlap_width);
	const int s_c = seam_center
		- min(0, seam_center - s_w / 2)
		+ min(0, overlap_width - (seam_center + s_w / 2));
	const int seam_start = s_c - s_w / 2;
	float seam_val = (float) ((x + seam_begin) - seam_start) / s_w;
	seam_val = fmaxf(0, fminf(1, seam_val));
	unsigned char blend_ratio = seam_val * 255;
	uchar4 *to = (uchar4 *)(dst + (dst_y + y) * dst_pitch + (dst_x+ x) * 4);

#if 1
	*to = make_uchar4(
			l->x - (l->x * blend_ratio)/255 + (r->x * blend_ratio)/255,
			l->y - (l->y * blend_ratio)/255 + (r->y * blend_ratio)/255,
			l->z - (l->z * blend_ratio)/255 + (r->z * blend_ratio)/255,
			255
			);
#else
	*to = make_uchar4(
			blend_ratio,
			blend_ratio,
			blend_ratio,
			255
			);
#endif
}

#define GAUSS_KERN_SIZE 7
__constant__ float gauss_kern[GAUSS_KERN_SIZE];

__global__
void kern_gauss_blur(const unsigned char *src, int src_pitch,
		int start_x, int start_y,
		int w, int h)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if(x >= w)
		return;

	if(y >= h)
		return;

	float4 val = make_float4(0, 0, 0, 0);
	for(int i = 0; i < GAUSS_KERN_SIZE; i++){
		int n = i - (GAUSS_KERN_SIZE / 2);
		int sample_x = start_x + max(0, min(x + n, w));
		uchar4 *src = (uchar4 *)(left + (l_start_y + y) * l_pitch + (l_start_x + x) * 4);

		val.x += 
	}
}

void cuda_blit_overlap(const Image_cuda *left, int l_start_x, int l_start_y,
		const Image_cuda *right, int r_start_x, int r_start_y,
		int seam_center, int seam_width, int seam_start, int overlap_width,
		Image_cuda *dst, int dst_x, int dst_y,
		int w, int h,
		CUstream_st *stream)
{
	
	dim3 blockSize(32,32);
	dim3 numBlocks((w + blockSize.x - 1) / blockSize.x,
			(h + blockSize.y - 1) / blockSize.y);

	kern_blit_overlap<<<numBlocks, blockSize, 0, stream>>>
		((const unsigned char *)left->data(), l_start_x, l_start_y, left->get_pitch(),
		(const unsigned char *)right->data(), r_start_x, r_start_y, right->get_pitch(),
		seam_center, seam_width, seam_start, overlap_width,
		(unsigned char *)dst->data(), dst_x, dst_y, dst->get_pitch(),
		w, h);
}

void cuda_gaussian_blur(const Image_cuda *img, int start_x, int start_y,
		int w, int h,
		CUstream_st *stream)
{
	const float sigma = 1.4f;

	static bool kern_initialized = false;
	if(!kern_initialized){
		Gaussian_kernel<GAUSS_KERN_SIZE> kern(sigma);

		cudaMemcpyToSymbol("gauss_kern", kern.get(), sizeof(float) * GAUSS_KERN_SIZE);

		kern_initialized = true;
	}
}
