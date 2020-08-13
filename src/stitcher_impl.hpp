#ifndef STITCHER_IMPL_HPP
#define STITCHER_IMPL_HPP

#include <vector>
#include "cam_stitch_ctx.hpp"
#include "stitcher_params.hpp"
#include "image.hpp"
#include "overlap.hpp"

namespace gpustitch{

struct Cam_overlap_info{
	Overlap *left;
	Overlap *right;
};


class Stitcher_impl{
public:
	Stitcher_impl(Stitcher_params stitch_params,
			const std::vector<Cam_params>& cam_params);

	Image_cuda *get_input_image(size_t cam_idx);
	void submit_input_image(size_t cam_idx, const void *data,
			size_t w, size_t h, size_t pitch);

	void download_stitched(void *dst, size_t pitch);
	Image_cuda *get_output_image();

	void stitch();

private:
	Stitcher_params stitcher_params;
	std::vector<Cam_stitch_ctx> cam_ctxs;

	std::vector<Overlap> overlaps;
	std::vector<Cam_overlap_info> cam_overlaps;

	Image_cuda output;

	void project_cam(Cam_stitch_ctx& cam_ctx);
	void project_cam(Cam_stitch_ctx& cam_ctx,
		size_t start_x, size_t end_x,
		size_t start_y, size_t end_y);

	void blend();

	void find_overlaps();
	void generate_masks();
};

}

#endif
