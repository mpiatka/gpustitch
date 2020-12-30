#ifndef STITCHER_PARAMS_HPP
#define STITCHER_PARAMS_HPP

namespace gpustitch{

/**
 * Enum representig blending algorithm
 */
enum Blend_algorithm{
	Multiband = 0, 
	Feather
};

/**
 * Structure containing stitching parameters
 */
struct Stitcher_params{
	int width; /**< Width of the resulting image */
	int height; /**< Height of the resulting image */

	Blend_algorithm blend_algorithm; /**< Blend algorithm to use */
	int feather_width; /**< Width to be used with the featger blending */
	int multiband_levels; /**< Number of pyramid levels for multiband blending */
};

}

#endif
