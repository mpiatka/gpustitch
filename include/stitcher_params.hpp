#ifndef STITCHER_PARAMS_HPP
#define STITCHER_PARAMS_HPP

namespace gpustitch{

enum Blend_algorithm{
	Multiband = 0,
	Feather
};

struct Stitcher_params{
	int width;
	int height;

	Blend_algorithm blend_algorithm;
};

}

#endif
