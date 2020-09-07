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

#define GAUSS_KERN_RADIUS 3
#define GAUSS_KERN_SIZE ((GAUSS_KERN_RADIUS) * 2 + 1)
#define GAUSS_TILE_W 16
#define GAUSS_TILE_H 16
#define GAUSS_TILE_SIZE ((GAUSS_TILE_WIDTH) * (GAUSS_TILE_HEIGHT))
__constant__ float gauss_kern[GAUSS_KERN_SIZE];

__global__
void kern_gauss_blur_row(
		unsigned char *src, int src_w, int src_h, int src_pitch,
		int start_x, int start_y,
		int w, int h)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	const int shared_w = GAUSS_TILE_W + GAUSS_KERN_RADIUS * 2;

	__shared__ uchar4 data[shared_w * GAUSS_TILE_H];

	const int src_x = start_x + x;
	const int src_y = start_y + y;

	if(src_y > src_h || y > h)
		return;

	uchar4 *src_line = (uchar4 *) (src + src_y * src_pitch);
	const int src_x_left = max(0, src_x - GAUSS_KERN_RADIUS);
	const int src_x_right = min(src_w, src_x + GAUSS_KERN_RADIUS);

	data[threadIdx.x + shared_w * threadIdx.y] = src_line[src_x_left];
	data[threadIdx.x + GAUSS_KERN_RADIUS*2 + shared_w * threadIdx.y] = src_line[src_x_right];

	__syncthreads();

	if(x > w || src_x > src_w)
		return;

	float4 val = make_float4(0, 0, 0, 0);
	for(int i = 0; i < GAUSS_KERN_SIZE; i++){
		val.x += data[threadIdx.x + i + shared_w * threadIdx.y].x * gauss_kern[i];
		val.y += data[threadIdx.x + i + shared_w * threadIdx.y].y * gauss_kern[i];
		val.z += data[threadIdx.x + i + shared_w * threadIdx.y].z * gauss_kern[i];
	}

	src_line[src_x] = make_uchar4(val.x, val.y, val.z, 255);
}

__global__
void kern_gauss_blur_col(
		unsigned char *src, int src_w, int src_h, int src_pitch,
		int start_x, int start_y,
		int w, int h)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	__shared__ uchar4 data[GAUSS_TILE_W * (GAUSS_TILE_H + GAUSS_KERN_RADIUS * 2)];

	const int src_x = start_x + x;
	const int src_y = start_y + y;

	if(src_x > src_w || x > w)
		return;

	int src_y_top = max(0, src_y - GAUSS_KERN_RADIUS);
	int src_y_bot = min(src_h, src_y + GAUSS_KERN_RADIUS);
	uchar4 *src_line_top = (uchar4 *) (src + src_y_top * src_pitch);
	uchar4 *src_line_bot = (uchar4 *) (src + src_y_bot * src_pitch);

	data[threadIdx.x + GAUSS_TILE_W * threadIdx.y] = src_line_top[src_x];
	data[threadIdx.x + GAUSS_TILE_W * (threadIdx.y + GAUSS_KERN_RADIUS*2)] = src_line_bot[src_x];

	__syncthreads();

	if(y > h || src_y > src_h)
		return;

	float4 val = make_float4(0, 0, 0, 0);
	for(int i = 0; i < GAUSS_KERN_SIZE; i++){
		val.x += data[threadIdx.x + GAUSS_TILE_W * (threadIdx.y + i)].x * gauss_kern[i];
		val.y += data[threadIdx.x + GAUSS_TILE_W * (threadIdx.y + i)].y * gauss_kern[i];
		val.z += data[threadIdx.x + GAUSS_TILE_W * (threadIdx.y + i)].z * gauss_kern[i];
	}

	uchar4 *src_line = (uchar4 *) (src + src_y * src_pitch);
	src_line[src_x] = make_uchar4(val.x, val.y, val.z, 255);
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

		cudaMemcpyToSymbol(gauss_kern, kern.get(), sizeof(float) * GAUSS_KERN_SIZE);

		kern_initialized = true;
	}

	dim3 blockSize(GAUSS_TILE_W, GAUSS_TILE_H);
	dim3 numBlocks((w + blockSize.x - 1) / blockSize.x,
			(h + blockSize.y - 1) / blockSize.y);

	kern_gauss_blur_row<<<numBlocks, blockSize, 0, stream>>>(
			(unsigned char *) img->data(),
			img->get_width(), img->get_height(), img->get_pitch(),
			start_x, start_y,
			w, h);
	kern_gauss_blur_col<<<numBlocks, blockSize, 0, stream>>>(
			(unsigned char *) img->data(),
			img->get_width(), img->get_height(), img->get_pitch(),
			start_x, start_y,
			w, h);
}
