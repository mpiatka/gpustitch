#ifndef PROJECT_CAM_H
#define PROJECT_CAM_H

#include "cam_stitch_ctx.hpp"

void cuda_project_cam(gpustitch::Cam_stitch_ctx& cam_ctx,
		size_t start_x, size_t end_x,
		size_t start_y, size_t end_y);

#endif
