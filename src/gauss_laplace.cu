#include "image.hpp"
#include "gaussian_kernel.hpp"

using namespace gpustitch;

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

__global__
void kern_subtract_images(const unsigned char *a, int a_pitch,
		const unsigned char *b, int b_pitch,
		const unsigned char *res, int res_pitch,
		int w, int h)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	const uchar4 *a_row = (uchar4 *) (a + y * a_pitch);
	const uchar4 *b_row = (uchar4 *) (b + y * b_pitch);
	uchar4 *res_row = (uchar4 *) (res + y * res_pitch);

	if(x > w || y > h)
		return;

	res_row[x] = make_uchar4(
			max(0, min(255, 128 + a_row[x].x - b_row[x].x)),
			max(0, min(255, 128 + a_row[x].y - b_row[x].y)),
			max(0, min(255, 128 + a_row[x].z - b_row[x].z)),
			0
			);
}

void cuda_subtract_images(const gpustitch::Image_cuda *a,
		const gpustitch::Image_cuda *b,
		gpustitch::Image_cuda *result,
		int w, int h,
		CUstream_st *stream)
{

	dim3 blockSize(32, 32);
	dim3 numBlocks((w + blockSize.x - 1) / blockSize.x,
			(h + blockSize.y - 1) / blockSize.y);

	kern_subtract_images<<<numBlocks, blockSize, 0, stream>>>(
			(const unsigned char *) a->data(), a->get_pitch(),
			(const unsigned char *) b->data(), b->get_pitch(),
			(unsigned char *) result->data(), result->get_pitch(),
			w, h);
}

__global__
void kern_add_images(const unsigned char *low, int low_pitch,
		const unsigned char *high, int high_pitch,
		unsigned char *res, int res_pitch,
		int w, int h)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	const uchar4 *low_row = (uchar4 *) (low + y * low_pitch);
	const uchar4 *high_row = (uchar4 *) (high + y * high_pitch);
	uchar4 *res_row = (uchar4 *) (res + y * res_pitch);

	if(x > w || y > h)
		return;

	res_row[x] = make_uchar4(
			max(0, min(255, low_row[x].x + (high_row[x].x - 128))),
			max(0, min(255, low_row[x].y + (high_row[x].y - 128))),
			max(0, min(255, low_row[x].z + (high_row[x].z - 128))),
			255
			);
}

void cuda_add_images(const gpustitch::Image_cuda *a,
		const gpustitch::Image_cuda *b,
		gpustitch::Image_cuda *result,
		int w, int h,
		CUstream_st *stream)
{

	dim3 blockSize(32, 32);
	dim3 numBlocks((w + blockSize.x - 1) / blockSize.x,
			(h + blockSize.y - 1) / blockSize.y);

	kern_add_images<<<numBlocks, blockSize, 0, stream>>>(
			(const unsigned char *) a->data(), a->get_pitch(),
			(const unsigned char *) b->data(), b->get_pitch(),
			(unsigned char *) result->data(), result->get_pitch(),
			w, h);
}

__global__
void kern_downsample(unsigned char *dst, int dst_pitch,
		const unsigned char *src, int src_pitch,
		int w, int h)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	const uchar4 *src_row = (uchar4 *) (src + y * 2 * src_pitch);
	uchar4 *dst_row = (uchar4 *) (dst + y * dst_pitch);

	if(x > w || y > h)
		return;

	dst_row[x] = src_row[2*x];
}

void cuda_downsample(gpustitch::Image_cuda *dst,
		const gpustitch::Image_cuda *src,
		int w, int h,
		CUstream_st *stream)
{
	int downsampled_w = w / 2;
	int downsampled_h = h / 2;
	dim3 blockSize(32, 32);
	dim3 numBlocks((downsampled_w + blockSize.x - 1) / blockSize.x,
			(downsampled_h + blockSize.y - 1) / blockSize.y);

	kern_downsample<<<numBlocks, blockSize, 0, stream>>>(
			(unsigned char *) dst->data(), dst->get_pitch(),
			(const unsigned char *) src->data(), src->get_pitch(),
			downsampled_w, downsampled_h);
}

__global__
void kern_upsample(unsigned char *dst, int dst_pitch,
		const unsigned char *src, int src_pitch,
		int w, int h)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	const uchar4 *src_row = (uchar4 *) (src + (y/2) * src_pitch);
	uchar4 *dst_row = (uchar4 *) (dst + y * dst_pitch);

	if(x > w || y > h)
		return;

	dst_row[x] = src_row[x / 2];
}

void cuda_upsample(gpustitch::Image_cuda *dst,
		const gpustitch::Image_cuda *src,
		int w, int h,
		CUstream_st *stream)
{

	int upsampled_w = 2 * w;
	int upsampled_h = 2 * h;
	dim3 blockSize(32, 32);
	dim3 numBlocks((upsampled_w + blockSize.x - 1) / blockSize.x,
			(upsampled_h + blockSize.y - 1) / blockSize.y);

	kern_upsample<<<numBlocks, blockSize, 0, stream>>>(
			(unsigned char *) dst->data(), dst->get_pitch(),
			(const unsigned char *) src->data(), src->get_pitch(),
			upsampled_w, upsampled_h);
}
