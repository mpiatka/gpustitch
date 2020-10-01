#include <toml11/toml.hpp>
#include "config_utils.hpp"
#include "math_util.hpp"

namespace gpustitch{

	namespace{
		double get_floating(const toml::value& val){
			return val.is_floating() ? val.as_floating(std::nothrow) :
				static_cast<double>(val.as_integer());
		}
	}

void read_params(const std::string& path,
		Stitcher_params& stitch_param,
		std::vector<Cam_params>& cam_params)
{
	const auto conf = toml::parse(path);

	const auto& cameras = toml::find(conf, "cameras");

	for(const auto& camera : cameras.as_array()){
		Cam_params param = {};
		param.width = toml::get<int>(camera.at("width"));
		param.height = toml::get<int>(camera.at("height"));
		param.focal_len = get_floating(camera.at("focal_len"));
		param.yaw = toRadian(get_floating(camera.at("yaw")));
		param.pitch = toRadian(get_floating(camera.at("pitch")));
		param.roll = toRadian(get_floating(camera.at("roll")));
		param.x_offset = get_floating(camera.at("x_offset"));
		param.y_offset = get_floating(camera.at("y_offset"));
		const auto distortion = toml::find<std::vector<float>>(camera, "distortion");
		for(size_t i = 0; i < distortion.size(); i++){
			param.distortion[i] = distortion[i];
		}

		cam_params.push_back(param);
	}
}

}
