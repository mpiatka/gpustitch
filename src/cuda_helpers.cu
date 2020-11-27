#include "cuda_helpers.h"

__global__
void kern_image_memset(unsigned char *img, int img_pitch,
		int start_x, int start_y, int w, int h,
		uchar4 val)
{

	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	uchar4 *res_row = (uchar4 *) (img + (y + start_y) * img_pitch);

	if(x >= w || y >= h)
		return;

	res_row[x + start_x] = val;
}

void cuda_image_memset(gpustitch::Image_cuda *img,
		int x, int y, int w, int h,
		int r, int g, int b, int a,
		CUstream_st *stream)
{

	dim3 blockSize(32, 32);
	dim3 numBlocks((w + blockSize.x - 1) / blockSize.x,
			(h + blockSize.y - 1) / blockSize.y);

	kern_image_memset<<<numBlocks, blockSize, 0, stream>>>(
			(unsigned char *) img->data(), img->get_pitch(),
			x, y, w, h,
			make_uchar4(r, g, b, a));
}
