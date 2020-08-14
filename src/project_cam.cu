#include "project_cam.h"
#include "image.hpp"

__constant__ float rot_mat[9];

	__global__
void kern_proj_cam(unsigned char *dst, int out_w, int out_h, int out_pitch,
		unsigned char *src, int in_w, int in_h, int in_pitch,
		int start_x, int start_y,
		float focal_len
		)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x + start_x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y + start_y;

	if(x + 1>= out_w)
		return;

	if(y >= out_h)
		return;

	const int cx = out_w / 2;
	const int cy = out_h / 2;

	const float pi = 3.14159265f;

	float lat = (((float)cy - y) / (out_h / 2)) * (pi / 2); 
	float lon = (((float)x - cx) / (out_w / 2)) * pi; 
	float r = cosf(lat);

	float3 dir = make_float3(sinf(lon) * r, sinf(lat), cosf(lon) * r);

	float3 rot_dir;

	rot_dir.x = dir.x * rot_mat[0] + dir.y * rot_mat[1] + dir.z * rot_mat[2];
	rot_dir.y = dir.x * rot_mat[3] + dir.y * rot_mat[4] + dir.z * rot_mat[5];
	rot_dir.z = dir.x * rot_mat[6] + dir.y * rot_mat[7] + dir.z * rot_mat[8];

	float angle = acosf(rot_dir.z);

	//double angle2 = atan2(ty, tx);

	float norm = rhypotf(rot_dir.x, rot_dir.y);
	rot_dir.x = rot_dir.x * norm;
	rot_dir.y = rot_dir.y * norm;

	float sampleR = focal_len * angle;
	int sampleX = /*cos(angle2)*/ rot_dir.x * sampleR + in_w / 2;
	int sampleY = /*-sin(angle2)*/ -rot_dir.y * sampleR + in_h / 2;

	if(sampleY >= 0 && sampleY < in_h
			&& sampleX >= 0 && sampleX < in_w)
	{
		uchar4 *from = (uchar4 *)(src + sampleY * in_pitch + sampleX * 4);
		uchar4 *to = (uchar4 *)(dst + y * out_pitch + x * 4);

		*to = *from;

	}
}

void cuda_project_cam(gpustitch::Cam_stitch_ctx& cam_ctx,
		size_t start_x, size_t end_x,
		size_t start_y, size_t end_y)
{
	size_t w = end_x - start_x;
	size_t h = end_y - start_y;

	gpustitch::Image_cuda *out = cam_ctx.get_projected_image();
	gpustitch::Image_cuda *in = cam_ctx.get_input_image();

	size_t out_w = out->get_width();
	size_t out_h = out->get_height();

	const auto& cam_params = cam_ctx.get_cam_params();

	const double *rot_mat_d = cam_ctx.get_rot_mat();
	float rot_mat_f[9];

	for(int i = 0; i < 9; i++){
		rot_mat_f[i] = rot_mat_d[i];
	}

	/*
	rot_mat_f[0] = 1;
	rot_mat_f[4] = 1;
	rot_mat_f[8] = 1;
	*/

	cudaMemcpyToSymbol(rot_mat, &rot_mat_f[0], 9 * sizeof(float), 0);


	dim3 blockSize(32,32);
	dim3 numBlocks((w + blockSize.x - 1) / blockSize.x,
			(h + blockSize.y - 1) / blockSize.y);

	kern_proj_cam<<<numBlocks, blockSize, 0, 0>>>
		((unsigned char *)out->data(), out_w, out_h, out->get_pitch(),
		 (unsigned char *)in->data(), in->get_width(), in->get_height(), in->get_pitch(),
		 start_x, start_y,
		 cam_params.focal_len
		 );
}
