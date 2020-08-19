#include "stitcher.hpp"
#include "stitcher_impl.hpp"

namespace gpustitch{


Stitcher::Stitcher(Stitcher_params stitch_params, const std::vector<Cam_params>& cam_params) :
	impl(new Stitcher_impl(stitch_params, cam_params)) {  }


Image *Stitcher::get_input_image(size_t cam_idx){
	return impl->get_input_image(cam_idx);
}
Image *Stitcher::get_output_image(){
	return impl->get_output_image();
}

Stitcher::~Stitcher() = default;

void Stitcher::stitch(){
	impl->stitch();
}

void Stitcher::submit_input_image(size_t cam_idx, const void *data,
		size_t w, size_t h, size_t pitch)
{
	impl->submit_input_image(cam_idx, data, w, h, pitch);
}

void Stitcher::get_input_stream(size_t cam_idx, CUstream_st **stream) const{
	impl->get_input_stream(cam_idx, stream);
}

void Stitcher::download_stitched(void *dst, size_t pitch){
	impl->download_stitched(dst, pitch);
}



}
