#ifndef CAM_PARAMS
#define CAM_PARAMS

namespace gpustitch{

struct Cam_params{
	size_t width;
	size_t height;

	double focal_len;
	float distortion[4];
	float x_offset;
	float y_offset;

	double yaw;
	double pitch;
	double roll;

	float rot_mat[9];
};

}

#endif
