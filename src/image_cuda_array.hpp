#ifndef IMAGE_CUDA_ARRAY_HPP
#define IMAGE_CUDA_ARRAY_HPP

#include "image.hpp"

namespace gpustitch{

class Image_cuda_array : public Image{
public:
	Image_cuda_array();
	~Image_cuda_array();
	Image_cuda_array(size_t width, size_t height, size_t pitch = 0);
	Image_cuda_array(const Image_cuda_array&) = delete;
	Image_cuda_array(Image_cuda_array&& o);
	Image_cuda_array& operator=(const Image_cuda_array&) = delete;
	Image_cuda_array& operator=(Image_cuda_array&&);

	cudaArray_t get_array(){ return cuda_array; }
	cudaTextureObject_t get_tex_obj(){ return tex_obj; }

private:
	cudaArray_t cuda_array;
	cudaTextureObject_t tex_obj;
	
};

}


#endif
