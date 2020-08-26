#ifndef CONFIG_UTILS
#define CONFIG_UTILS

#include <vector>
#include <string>
#include "cam_params.hpp"
#include "stitcher_params.hpp"

namespace gpustitch{

bool read_params(const std::string& path,
		Stitcher_params& stitch_param,
		std::vector<Cam_params>& cam_params);

}

#endif
