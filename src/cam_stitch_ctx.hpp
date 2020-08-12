#ifndef CAM_STITCH_CTX
#define CAM_STITCH_CTX

#include "cam_params.hpp"
#include "stitcher_params.hpp"
#include "image.hpp"
#include "math_util.hpp"
#include "profile_timer.hpp"

namespace gpustitch{

class Cam_stitch_ctx{
public:
	Cam_stitch_ctx(Stitcher_params stitch_params, Cam_params cam_params) :
		cam_params(cam_params),
		stitch_params(stitch_params),
		in(cam_params.width, cam_params.height),
		projected(stitch_params.width, stitch_params.height)
	{
		PROFILE_FUNC;
		getRotationMat(cam_params.yaw,
				cam_params.pitch,
				cam_params.roll,
				rotation_mat);

		double yaw_center = cam_params.yaw;
		double half_fov = (double) cam_params.height / 2 / cam_params.focal_len; //TODO
		proj_angle_start = yaw_center - half_fov;
		proj_angle_end = yaw_center + half_fov;

   	}

	Image *get_input_image() { return &in; }
	Image *get_projected_image() { return &projected; }
	const Cam_params& get_cam_params() const { return cam_params; }
	const double *get_rot_mat() const { return rotation_mat; }

	Cam_params cam_params;
	Stitcher_params stitch_params;
	Image in;
	Image projected;

	double proj_angle_start;
	double proj_angle_end;

	double rotation_mat[9];
};

}

#endif
