#ifndef CAM_PARAMS
#define CAM_PARAMS

namespace gpustitch{

struct Cam_params{
	size_t width;
	size_t height;

	double focal_len;

	double yaw;
	double pitch;
	double roll;
};

}

#endif
