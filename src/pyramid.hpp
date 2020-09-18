#ifndef PYRAMID_HPP
#define PYRAMID_HPP

#include <vector>
#include "gpustitch_common.hpp"
#include "image.hpp"
#include "cuda_stream.hpp"

namespace gpustitch{

class Pyramid{
public:
	Pyramid(size_t w, size_t h, unsigned levels);

	Pyramid(const Pyramid&) = delete;
	Pyramid(Pyramid&&) = default;
	Pyramid& operator=(const Pyramid&) = delete;
	Pyramid& operator=(Pyramid&&) = default;

	void construct_from(const Image_cuda& src,
			size_t x, size_t y, size_t w, size_t h,
			const Cuda_stream& stream);

	void reconstruct_to(Image_cuda *dst,
			size_t x, size_t y, size_t w, size_t h,
			const Cuda_stream& stream);

	const Image_cuda *get_level(int lvl) const{ return &laplace_imgs[lvl]; }
	Image_cuda *get_level(int lvl){ return &laplace_imgs[lvl]; };

	size_t get_levels(){ return levels; }

private:
	size_t width;
	size_t height;
	size_t levels;
	Image_cuda tmp;
	Image_cuda tmp_blurred;
	std::vector<Image_cuda> laplace_imgs;
};

}


#endif
