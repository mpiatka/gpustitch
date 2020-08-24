#ifndef STITCHER_HPP
#define STITCHER_HPP

#include <vector>
#include <memory>
#include "cam_params.hpp"
#include "stitcher_params.hpp"
#include "image.hpp"
#include "gpustitch_common.hpp"

namespace gpustitch{

class Stitcher_impl;

class Stitcher{
public:
	Stitcher(Stitcher_params stitch_params, const std::vector<Cam_params>& cam_params);
	Stitcher();
	~Stitcher();

	Stitcher& operator=(Stitcher&&);

	Image *get_input_image(size_t cam_idx);
	Image *get_output_image();

	void get_input_stream(size_t cam_idx, CUstream_st **stream) const;

	void submit_input_image(size_t cam_idx, const void *data,
			size_t w, size_t h, size_t pitch);

	void download_stitched(void *dst, size_t pitch);

	void stitch();
private:
	std::unique_ptr<Stitcher_impl> impl;
};

}

#endif
