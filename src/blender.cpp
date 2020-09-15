#include <cmath>
#include "blender.hpp"
#include "profile_timer.hpp"
#include "blend.h"

namespace gpustitch{

Blender::Blender(const std::vector<Overlap>& overlaps) : overlaps(overlaps) {  }

Feather_blender::Feather_blender(const std::vector<Overlap>& overlaps) :
	Blender(overlaps)
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


}
