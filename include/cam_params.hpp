#ifndef CAM_PARAMS
#define CAM_PARAMS

namespace gpustitch{

/**
 * Structure containing camera rotation and lens information
 */
struct Cam_params{
	size_t width; /**< Width of input frames from camera */
	size_t height; /**< Height of input frames from camera */

	double focal_len; /**< Focal length expressed in pixels */

	/**
	 * Distortion coefficients of used lens. Uses the panotools distortion
	 * model (https://wiki.panotools.org/Lens_correction_model)
	 */
	float distortion[4];

	float x_offset; /**< Lens optical center offset from the image center */
	float y_offset; /**< Lens optical center offset from the image center */

	double yaw; /**< Yaw of the camera */
	double pitch; /**< Pitch of the camera */
	double roll; /**< Roll of the camera */

	//Variables after this line are used internally and are not user settable
	float rot_mat[9];
};

}

#endif
