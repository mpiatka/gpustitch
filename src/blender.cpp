#include <cmath>
#include "blender.hpp"
#include "profile_timer.hpp"
#include "blend.h"

namespace gpustitch{

Blender::Blender(const Stitcher_params& params,
		const std::vector<Overlap>& overlaps) :
	params(params), overlaps(overlaps) {  }

Feather_blender::Feather_blender(const Stitcher_params& params,
		const std::vector<Overlap>& overlaps) :
	Blender(params, overlaps)
{ 

}

void Feather_blender::blend_overlaps(Image_cuda *output,
			const std::vector<Cam_stitch_ctx>& cam_ctxs,
			const Cuda_stream& stream)
{
	PROFILE_FUNC;
	for(const auto& o : overlaps){
		const Image_cuda *left = cam_ctxs[o.left_idx].get_projected_image();
		const Image_cuda *right = cam_ctxs[o.right_idx].get_projected_image();

		const int seam_width = 30;

		int overlap_width = (o.start_x < o.end_x) ? o.end_x - o.start_x : output->get_width() - o.start_x + o.end_x;

		const int seam_center = overlap_width / 2 + sin(clock() / 1000000.0) * 150;

		if(o.start_x < o.end_x){
			const int w = o.end_x - o.start_x;
			cuda_blit_overlap(left, o.start_x, 0,
					right, o.start_x, 0,
					seam_center, seam_width, 0, overlap_width,
					output, o.start_x, 0,
					w, output->get_height(),
					stream.get()
					);
		} else {
			int w = output->get_width() - o.start_x;
			cuda_blit_overlap(left, o.start_x, 0,
					right, o.start_x, 0,
					seam_center, seam_width, 0, overlap_width,
					output, o.start_x, 0,
					w, output->get_height(),
					stream.get()
					);
			w = o.end_x;
			cuda_blit_overlap(left, 0, 0,
					right, 0, 0,
					seam_center, seam_width, output->get_width() - o.start_x, overlap_width,
					output, 0, 0,
					w, output->get_height(),
					stream.get()
					);
		}
	}
}

static size_t get_max_width(const std::vector<Overlap>& overlaps){
	size_t max_w = 0;

	for(const auto& o : overlaps){
		if(o.width > max_w)
			max_w = o.width;
	}

	return max_w;
}

Multiband_blender::Multiband_blender(const Stitcher_params& params,
		const std::vector<Overlap>& overlaps) :
	Blender(params, overlaps),
	tmp(get_max_width(overlaps), params.height, 4)
{
	for(const auto& o : overlaps){
		Overlap_pyramid in_pyramid{
			Pyramid(o.width, params.height, 4),
			Pyramid(o.width, params.height, 4)
		};

		overlap_pyramids.emplace_back(std::move(in_pyramid));
	}

}

void Multiband_blender::submit_image(const Image_cuda& in,
		size_t idx,
		const Cuda_stream& stream)
{
	for(size_t i = 0; i < overlaps.size(); i++){
		const auto& o = overlaps[i];
		auto& pyramids = overlap_pyramids[i];
		//TODO handle wrap-around case
		if(idx == o.left_idx){
			pyramids.left.construct_from(in,
					o.start_x, 0, o.width, params.height,
					stream);
		}
		if(idx == o.right_idx){
			pyramids.right.construct_from(in,
					o.start_x, 0, o.width, params.height,
					stream);
		}
	}
}

void Multiband_blender::blend_overlaps(Image_cuda *output,
			const std::vector<Cam_stitch_ctx>& cam_ctxs,
			const Cuda_stream& stream)
{
	for(size_t i = 0; i < overlaps.size(); i++){
		const auto& o = overlaps[i];
		const auto& pyramid = overlap_pyramids[i];

		for(size_t i = 0; i < tmp.get_levels(); i++){
			const auto& src_l = pyramid.left.get_level(i);
			const auto& src_r = pyramid.right.get_level(i);
			int w = src_l->get_width();
			int h = src_l->get_height();
			cuda_feather_simple(src_l,
					src_r,
					w / 2, 24,
					tmp.get_level(i),
					w, h,
					stream.get());
		}

		tmp.reconstruct_to(output, o.start_x, 0, o.width, params.height, stream);
	}

}


}
