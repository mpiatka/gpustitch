#ifndef BLEND_H
#define BLEND_H

#include "gpustitch_common.hpp"

void cuda_blit_overlap(const gpustitch::Image_cuda *left, int l_start_x, int l_start_y,
		const gpustitch::Image_cuda *right, int r_start_x, int r_start_y,
		int seam_center, int seam_width, int seam_start, int overlap_width,
		gpustitch::Image_cuda *dst, int dst_x, int dst_y,
		int w, int h,
		CUstream_st *stream);

#endif
