#include <iostream>
#include <cmath>
#include <vector>
#include <png++/png.hpp>
#include <cuda_runtime.h>

#include "stitcher.hpp"
#include "profile_timer.hpp"

using namespace gpustitch;

static double toRadian(double deg){
	return (deg / 180) * 3.14159265;
}

void submit_cam_image(Stitcher& stitcher,
		const png::image<png::rgba_pixel>& in,
		size_t idx)
{
	PROFILE_FUNC;
	size_t tmp_pitch = in.get_width() * 4;

	unsigned char *buf;
	cudaHostAlloc(&buf, in.get_width() * in.get_height() * 4, 0);

	for(size_t y = 0; y < in.get_height(); y++){
		for(size_t x = 0; x < in.get_width(); x++){
			unsigned char *dst = buf + y * tmp_pitch + x * 4;
			dst[0] = in[y][x].red;
			dst[1] = in[y][x].green;
			dst[2] = in[y][x].blue;
			dst[3] = in[y][x].alpha;
		}
	}

	stitcher.submit_input_image(idx, buf, in.get_width(), in.get_height(), tmp_pitch);
}

int main(){
	Stitcher_params stitch_params;
	stitch_params.width = 7680;
	stitch_params.height = 3840;

	std::vector<Cam_params> cam_params;

	Cam_params params{};
	params.width = 3840;
	params.height = 2160;
	params.distortion[3] = 1;

	params.focal_len = 1173.753;
	params.yaw = toRadian(0.02325995);
	params.pitch = toRadian(-0.832604);
	params.roll = toRadian(90.0565);
	cam_params.push_back(params);

	params.focal_len = 1198.055;
	params.yaw = toRadian(90.89778);
	params.pitch = toRadian(-0.379786);
	params.roll = toRadian(90.3411);
	cam_params.push_back(params);

	params.focal_len = 1181.822;
	params.yaw = toRadian(-177.1611);
	params.pitch = toRadian(-0.679113);
	params.roll = toRadian(89.41799);
	cam_params.push_back(params);

	params.focal_len = 1164.125;
	params.yaw = toRadian(-88.45461);
	params.pitch = toRadian(0.9491351);
	params.roll = toRadian(89.92228);
	cam_params.push_back(params);

	Stitcher stitcher(stitch_params, cam_params);

	png::image<png::rgba_pixel> output(stitch_params.width, stitch_params.height);
	std::vector<unsigned char> out;
	size_t w = stitch_params.width;
	size_t h = stitch_params.height;
	size_t pitch = w * 4;
	out.resize(h * pitch);

	png::image<png::rgba_pixel> in("0_0.png");
	submit_cam_image(stitcher, in, 0);
	in = png::image<png::rgba_pixel>("1_0.png");
	submit_cam_image(stitcher, in, 1);
	in = png::image<png::rgba_pixel>("2_0.png");
	submit_cam_image(stitcher, in, 2);
	in = png::image<png::rgba_pixel>("3_0.png");
	submit_cam_image(stitcher, in, 3);


	stitcher.stitch();

	stitcher.download_stitched(out.data(), pitch);

	for(size_t y = 0; y < h; y++){
		for(size_t x = 0; x < w; x++){
			unsigned char *buf = out.data() + y * pitch + x * 4;
			output[y][x].red = buf[0];
			output[y][x].green = buf[1];
			output[y][x].blue = buf[2];
			output[y][x].alpha = buf[3];
		}
	}

	output.write("out.png");

	return 0;
}
