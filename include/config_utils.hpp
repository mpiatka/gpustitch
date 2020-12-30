#ifndef CONFIG_UTILS
#define CONFIG_UTILS

#include <vector>
#include <string>
#include "cam_params.hpp"
#include "stitcher_params.hpp"

namespace gpustitch{

/** Helper function to read parameters from a config file. It expects a TOML
 * formatted file containing key value pairs using the same names as the
 * Cam_params struct
 *
 * @param path filepath of config file
 * @param stitch_param reference to struct where the result will be written
 * @param cam_params reference to vector of structs where the result will be written
 */
void read_params(const std::string& path,
		Stitcher_params& stitch_param,
		std::vector<Cam_params>& cam_params);

}

#endif
