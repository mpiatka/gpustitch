#include <cuda_runtime.h>
#include "image_cuda_array.hpp"

namespace gpustitch{

Image_cuda_array::Image_cuda_array(): Image_cuda_array(0, 0, 0){ }

Image_cuda_array::Image_cuda_array(size_t width, size_t height, size_t pitch):
	Image(width, height, pitch)
{
	cudaChannelFormatDesc fmt_desc = cudaCreateChannelDesc<uchar4>();
	
	cudaError_t res = cudaMallocArray(&cuda_array,
			&fmt_desc,
			width,
			height,
			0);

	if(res != cudaSuccess){
		throw "Failed to allocate array";
	}

	cudaResourceDesc res_desc;
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = cuda_array;

	cudaTextureDesc tex_desc = {};
	tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.filterMode = cudaFilterModeLinear;
	tex_desc.readMode = cudaReadModeNormalizedFloat;

	res = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);
	if(res != cudaSuccess){
		throw "Failed to create tex obj";
	}

}

Image_cuda_array::~Image_cuda_array(){
	if(cuda_array){
		cudaFreeArray(cuda_array);
		cudaDestroyTextureObject(tex_obj);
	}
}

Image_cuda_array::Image_cuda_array(Image_cuda_array&& o): Image(std::move(o)){
	cuda_array = o.cuda_array;
	tex_obj = o.tex_obj;
	o.cuda_array = nullptr;
	o.tex_obj = 0;
}

Image_cuda_array& Image_cuda_array::operator=(Image_cuda_array&& o){
	Image::operator=(std::move(o));
	cuda_array = o.cuda_array;
	tex_obj = o.tex_obj;
	o.cuda_array = nullptr;
	o.tex_obj = 0;

	return *this;
}

}
