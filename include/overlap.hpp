#ifndef OVERLAP_HPP
#define OVERLAP_HPP

#include "image.hpp"

namespace gpustitch{

/**
 * Structure containing information about camera overlaps
 */
struct Overlap{
	size_t left_idx; /**< Index of the camera on the left side of overlap */
	size_t right_idx; /**< Index of the camera on the right side of overlap */

	double proj_angle_start; /**< Yaw of the left edge of the projected image */
	double proj_angle_end; /**< Yaw of the right edge of the projected image */

	int start_x; /**< X coord of the left edge of the projected image */
	int end_x; /**< X coord of the right edge of the projected image */
	size_t width; /**< Width in pixels of the projected image */

};


}

#endif
