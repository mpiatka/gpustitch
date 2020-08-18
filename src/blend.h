#ifndef BLEND_H
#define BLEND_H

void cuda_blit_overlap(const gpustitch::Image_cuda *left, int l_start_x, int l_start_y,
		const gpustitch::Image_cuda *right, int r_start_x, int r_start_y,
		const gpustitch::Image_cuda *mask, int m_start_x, int m_start_y,
		gpustitch::Image_cuda *dst, int dst_x, int dst_y,
		int w, int h);

#endif
