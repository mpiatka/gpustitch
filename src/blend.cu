#include "image.hpp"

__global__
void kern_blit_overlap(
		const unsigned char *left, int l_start_x, int l_start_y, int l_pitch,
		const unsigned char *right, int r_start_x, int r_start_y, int r_pitch,
		const unsigned char *mask, int m_start_x, int m_start_y, int m_pitch,
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
	uchar4 *m = (uchar4 *)(mask + (m_start_y + y) * m_pitch + (m_start_x + x) * 4);
	uchar4 *to = (uchar4 *)(dst + (dst_y + y) * dst_pitch + (dst_x+ x) * 4);

	*to = make_uchar4(
			l->x - (l->x * m->x)/255 + (r->x * m->x)/255,
			l->y - (l->y * m->x)/255 + (r->y * m->x)/255,
			l->z - (l->z * m->x)/255 + (r->z * m->x)/255,
			255
			);
}

using namespace gpustitch;

void cuda_blit_overlap(const Image_cuda *left, int l_start_x, int l_start_y,
		const Image_cuda *right, int r_start_x, int r_start_y,
		const Image_cuda *mask, int m_start_x, int m_start_y,
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
		(const unsigned char *)mask->data(), m_start_x, m_start_y, mask->get_pitch(),
		(unsigned char *)dst->data(), dst_x, dst_y, dst->get_pitch(),
		w, h);
}
