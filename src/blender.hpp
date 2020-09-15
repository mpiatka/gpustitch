#ifndef BLENDER_HPP
#define BLENDER_HPP

#include <vector>
#include "overlap.hpp"
#include "image.hpp"
#include "cam_stitch_ctx.hpp"
#include "gpustitch_common.hpp"
#include "cuda_stream.hpp"

namespace gpustitch{

class Blender{
public:
	Blender(const std::vector<Overlap>& overlaps);

	virtual void blend_overlaps(Image_cuda *output,
			const std::vector<Cam_stitch_ctx>& cam_ctxs,
			const Cuda_stream& stream) = 0;

protected:
	const std::vector<Overlap>& overlaps;

};

class Feather_blender : public Blender{
public:
	Feather_blender(const std::vector<Overlap>& overlaps);

	virtual void blend_overlaps(Image_cuda *output,
			const std::vector<Cam_stitch_ctx>& cam_ctxs,
			const Cuda_stream& stream) override;

private:

};

}

#endif
