#include <cuda_runtime.h>
#include "image_cuda_array.hpp"

namespace gpustitch{

Image_cuda_array::Image_cuda_array(): Image_cuda_array(0, 0, 0){ }

Image_cuda_array::Image_cuda_array(size_t width, size_t height, size_t pitch):
	Image(width, height, pitch)
{
	cudaChannelFormatDesc fmt_desc;

	fmt_desc.f = cudaChannelFormatKindUnsigned;
	fmt_desc.x = 8;
	fmt_desc.y = 8;
	fmt_desc.z = 8;
	fmt_desc.w = 8;
	
	cudaMallocArray(&cuda_array,
			&fmt_desc,
			width,
			height,
			cudaArrayTextureGather);
}

Image_cuda_array::~Image_cuda_array(){
	cudaFreeArray(cuda_array);
}

Image_cuda_array::Image_cuda_array(Image_cuda_array&& o): Image(std::move(o)){
	cuda_array = o.cuda_array;
	o.cuda_array = nullptr;
}

Image_cuda_array& Image_cuda_array::operator=(Image_cuda_array&& o){
	Image::operator=(std::move(o));
	cuda_array = o.cuda_array;
	o.cuda_array = nullptr;

	return *this;
}

}
