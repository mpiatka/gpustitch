#ifndef OVERLAP_HPP
#define OVERLAP_HPP

#include "image.hpp"

namespace gpustitch{

struct Overlap{
	size_t left_idx;
	size_t right_idx;

	double proj_angle_start;
	double proj_angle_end;

	int start_x;
	int end_x;

	Image_cuda mask;
};


}

#endif
