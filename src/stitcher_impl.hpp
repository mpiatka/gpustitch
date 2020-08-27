#ifndef STITCHER_IMPL_HPP
#define STITCHER_IMPL_HPP

#include <vector>
#include "cam_stitch_ctx.hpp"
#include "stitcher_params.hpp"
#include "image.hpp"
#include "overlap.hpp"
#include "stitcher.hpp"

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

	void submit_input_image_async(size_t cam_idx);

	void submit_input_image(size_t cam_idx, const void *data,
			size_t w, size_t h, size_t pitch,
			Src_mem_kind mem_kind = Src_mem_kind::Host);

	void submit_input_image_async(size_t cam_idx, const void *data,
			size_t w, size_t h, size_t pitch,
			Src_mem_kind mem_kind = Src_mem_kind::Host);

	void get_input_stream(size_t cam_idx, CUstream_st **stream) const;
	void get_output_stream(CUstream_st **stream) const;

	void download_stitched(void *dst, size_t pitch);
	Image_cuda *get_output_image();

	void stitch();

private:
	Stitcher_params stitcher_params;
	std::vector<Cam_stitch_ctx> cam_ctxs;

	std::vector<Overlap> overlaps;
	std::vector<Cam_overlap_info> cam_overlaps;

	Image_cuda output;

	//TODO RAII stream wrapper
	CUstream_st *out_stream;

	void project_cam(Cam_stitch_ctx& cam_ctx);

	void blend();

	void find_overlaps();
	void generate_masks();
};

}

#endif
