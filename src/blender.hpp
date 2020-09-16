#ifndef BLENDER_HPP
#define BLENDER_HPP

#include <vector>
#include "overlap.hpp"
#include "image.hpp"
#include "cam_stitch_ctx.hpp"
#include "gpustitch_common.hpp"
#include "cuda_stream.hpp"
#include "pyramid.hpp"
#include "stitcher_params.hpp"

namespace gpustitch{

class Blender{
public:
	Blender(const Stitcher_params& params,
			const std::vector<Overlap>& overlaps);

	virtual void blend_overlaps(Image_cuda *output,
			const std::vector<Cam_stitch_ctx>& cam_ctxs,
			const Cuda_stream& stream) = 0;

	virtual void submit_image(const Image_cuda& in,
			size_t idx,
			const Cuda_stream& stream) {  }

protected:
	const std::vector<Overlap>& overlaps;
	const Stitcher_params& params;

};

class Feather_blender : public Blender{
public:
	Feather_blender(const Stitcher_params& params,
			const std::vector<Overlap>& overlaps);

	virtual void blend_overlaps(Image_cuda *output,
			const std::vector<Cam_stitch_ctx>& cam_ctxs,
			const Cuda_stream& stream) override;

private:

};

struct Overlap_pyramid{
	Pyramid left;
	Pyramid right;
};

class Multiband_blender : public Blender{
public:
	Multiband_blender(const Stitcher_params& params,
			const std::vector<Overlap>& overlaps);

	virtual void blend_overlaps(Image_cuda *output,
			const std::vector<Cam_stitch_ctx>& cam_ctxs,
			const Cuda_stream& stream) override;

	virtual void submit_image(const Image_cuda& in,
			size_t idx,
			const Cuda_stream& stream) override;

private:
	std::vector<Overlap_pyramid> overlap_pyramids; 
	Pyramid tmp;
};

}

#endif
