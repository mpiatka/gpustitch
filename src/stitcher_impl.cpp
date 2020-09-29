#include <cmath>
#include <cstring>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include "stitcher_impl.hpp"
#include "math_util.hpp"
#include "project_cam.h"
#include "blend.h"
#include "profile_timer.hpp"
#include "gauss_laplace.h"

namespace gpustitch{

Stitcher_impl::Stitcher_impl(Stitcher_params stitch_params,
		const std::vector<Cam_params>& cam_params) :
	stitcher_params(stitch_params),
	output(stitch_params.width, stitch_params.height)
{
	PROFILE_FUNC;
	for(const auto& cam_param : cam_params){
		cam_ctxs.emplace_back(stitch_params, cam_param);
	}

	PROFILE_DETAIL("init image");
	Image_cpu tmp(stitcher_params.width, stitcher_params.height);
	for(size_t y = 0; y < tmp.get_height(); y++){
		for(size_t x = 0; x < tmp.get_width(); x++){
			unsigned char *buf = static_cast<unsigned char*>(tmp.data())
				+ y * tmp.get_pitch()
				+ x * tmp.get_bytes_per_px();

			buf[0] = 0;
			buf[1] = 255;
			buf[2] = 0;
			buf[3] = 255;
		}
	}

	tmp.upload(output);

	find_overlaps();

	blender = std::unique_ptr<Blender>(new Multiband_blender(stitcher_params, overlaps));
}

void Stitcher_impl::submit_input_image_async(size_t cam_idx){
	Cam_stitch_ctx& cam_ctx = cam_ctxs[cam_idx];
	project_cam(cam_ctx);
	blender->submit_image(*cam_ctx.get_projected_image(), cam_idx, cam_ctx.in_stream);
}

void Stitcher_impl::submit_input_image(size_t cam_idx, const void *data,
		size_t w, size_t h, size_t pitch,
		Src_mem_kind mem_kind)
{
	Cam_stitch_ctx& cam_ctx = cam_ctxs[cam_idx];
	submit_input_image_async(cam_idx, data, w, h, pitch, mem_kind);
	cam_ctx.in_stream.synchronize();
}

void Stitcher_impl::submit_input_image_async(size_t cam_idx, const void *data,
		size_t w, size_t h, size_t pitch,
		Src_mem_kind mem_kind)
{
	Image_cuda_array *i = cam_ctxs[cam_idx].get_input_image();

	Cam_stitch_ctx& cam_ctx = cam_ctxs[cam_idx];

	cudaMemcpyKind memcpy_kind =
		(mem_kind == Src_mem_kind::Host) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;

	cudaMemcpy2DToArrayAsync(i->get_array(), 0, 0,
			data, pitch,
			w * 4, h,
			memcpy_kind,
			cam_ctx.in_stream.get());

	submit_input_image_async(cam_idx);
}

void Stitcher_impl::get_input_stream(size_t cam_idx, CUstream_st **stream) const{
	*stream = cam_ctxs[cam_idx].in_stream.get();
}

void Stitcher_impl::get_output_stream(CUstream_st **stream) const{
	*stream = out_stream.get();
}

void Stitcher_impl::download_stitched(void *dst, size_t pitch){
	Image_cuda *i = get_output_image();
	//Image_cuda *i = cam_ctxs[0].get_projected_image();

	out_stream.synchronize();

	cudaMemcpy2D(dst, pitch,
			i->data(), i->get_pitch(),
			i->get_width() * i->get_bytes_per_px(), i->get_height(),
			cudaMemcpyDeviceToHost);
}

Image_cuda *Stitcher_impl::get_output_image(){
	return &output;
}

void Stitcher_impl::project_cam(Cam_stitch_ctx& cam_ctx){
	int start_x = stitcher_params.width / 2 * (1 + cam_ctx.proj_angle_start / 3.14);
	int end_x = stitcher_params.width / 2 * (1 + cam_ctx.proj_angle_end / 3.14);

	if(start_x < 0) start_x += stitcher_params.width;
	if(start_x > stitcher_params.width) start_x -= stitcher_params.width;

	if(end_x < 0) end_x += stitcher_params.width;
	if(end_x > stitcher_params.width) end_x -= stitcher_params.width;

	int start_y = 0;
	int end_y = stitcher_params.height;

	if(start_x < end_x){
		cuda_project_cam(cam_ctx, start_x, end_x, start_y, end_y);
	} else {
		cuda_project_cam(cam_ctx, 0, end_x, start_y, end_y);
		cuda_project_cam(cam_ctx, start_x, stitcher_params.width, start_y, end_y);
	}
}

static double normalize_angle(double angle){
	while(angle > 3.14) angle -= 2*3.14;
	while(angle <= -3.14) angle += 2*3.14;
	return angle;
}

static bool angle_is_between(double left, double angle, double right){
	if(left > right)
		return (-3.14 <= angle && angle < right) || (left < angle && angle < 3.14);

	return left < angle && angle < right;
}

void Stitcher_impl::find_overlaps(){
	for(size_t left_idx = 0; left_idx < cam_ctxs.size(); left_idx++){
		const auto& left_cam = cam_ctxs[left_idx];
		double left_end = normalize_angle(left_cam.proj_angle_end);
		double left_center = normalize_angle(left_cam.cam_params.yaw);

		for(size_t right_idx = 0; right_idx < cam_ctxs.size(); right_idx++){
			if(left_idx == right_idx)
				continue;

			const auto& right_cam = cam_ctxs[right_idx];
			double right_start = normalize_angle(right_cam.proj_angle_start);

			if(angle_is_between(left_center, right_start, left_end)){
				std::cout << "Found overlap: " << left_idx << ", " << right_idx;
				std::cout << std::endl;
				int start_x = stitcher_params.width / 2 * (1 + right_start / 3.14);
				int end_x = stitcher_params.width / 2 * (1 + left_end / 3.14);
				std::cout << "Pixel loc: " << start_x << ", " << end_x;
				std::cout << std::endl;

				Overlap o;
				o.left_idx = left_idx;
				o.right_idx = right_idx;
				o.proj_angle_start = right_start;
				o.proj_angle_end = left_end;
				o.start_x = start_x;
				o.end_x = end_x;

				if(end_x > start_x)
					o.width = end_x - start_x;
				else
					o.width = end_x + (stitcher_params.width - start_x);

				overlaps.push_back(std::move(o));
			}
		}
	}

	cam_overlaps.resize(cam_ctxs.size());
	for(auto& overlap : overlaps){
		cam_overlaps[overlap.left_idx].right = &overlap;
		cam_overlaps[overlap.right_idx].left = &overlap;
	}
}

static void cuda_blit(const Image_cuda *src, int src_x, int src_y,
		Image_cuda *dst, int dst_x, int dst_y,
		int w, int h,
		const Cuda_stream& stream)
{
	copy_image(dst, src, dst_x, dst_y, src_x, src_y, w, h, stream);

	//not needed stream is output stream
	//stream.synchronize();
	cudaError_t err = cudaGetLastError();

	if(err != cudaSuccess){
		std::cerr << cudaGetErrorString(err) << std::endl;
	}
}

int angle_to_px(size_t width, double angle){
	return width / 2 * (1 + angle / 3.14);
}

void Stitcher_impl::copy_non_overlaping(){
	PROFILE_FUNC;
	for(size_t i = 0; i < cam_ctxs.size(); i++){
		Overlap *left_overlap = cam_overlaps[i].left;
		Overlap *right_overlap = cam_overlaps[i].right;

		int start_x;
		int end_x;
		if(left_overlap){
			start_x = left_overlap->end_x;
		} else {
			start_x = angle_to_px(stitcher_params.width, cam_ctxs[i].proj_angle_start);
		}
		if(right_overlap){
			end_x = right_overlap->start_x;
		} else {
			end_x = angle_to_px(stitcher_params.width, cam_ctxs[i].proj_angle_end);
		}
		cam_ctxs[i].in_stream.synchronize();

		if(end_x < start_x){
			cuda_blit(cam_ctxs[i].get_projected_image(), start_x, 0,
					&output, start_x, 0,
					output.get_width() - start_x, output.get_height(),
					out_stream);
			cuda_blit(cam_ctxs[i].get_projected_image(), 0, 0,
					&output, 0, 0,
					end_x, output.get_height(),
					out_stream);
		} else {
			cuda_blit(cam_ctxs[i].get_projected_image(), start_x, 0,
					&output, start_x, 0,
					end_x - start_x, output.get_height(),
					out_stream);
		}
	}
}

void Stitcher_impl::stitch(){
	copy_non_overlaping();
	blender->blend_overlaps(&output, cam_ctxs, out_stream);
}


}
