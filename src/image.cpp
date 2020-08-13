#include <cuda_runtime.h>
#include "image.hpp"
#include "profile_timer.hpp"

namespace gpustitch{

Image::Image() : Image(0, 0, 0) {  }

Image::Image(size_t width, size_t height, size_t pitch) :
	width(width),
	height(height),
	pitch(pitch)
{
	PROFILE_FUNC;
	bytes_per_px = 4; //RGBA
	if(pitch == 0){
		this->pitch = width * bytes_per_px;
	}
}

Image::Image(Image&& o){
	//device_data = o.device_data;
	width = o.width;
	height = o.height;
	pitch = o.pitch;
	bytes_per_px = o.bytes_per_px;
	o.width = 0;
	o.height = 0;
	o.pitch = 0;
	//o.device_data = nullptr;
}

Image& Image::operator=(Image&& o){
	//device_data = o.device_data;
	width = o.width;
	height = o.height;
	pitch = o.pitch;
	bytes_per_px = o.bytes_per_px;
	o.width = 0;
	o.height = 0;
	o.pitch = 0;
	//o.device_data = nullptr;

	return *this;
}

Image_cuda::Image_cuda(): Image_cuda(0, 0, 0){ }

Image_cuda::Image_cuda(size_t width, size_t height, size_t pitch):
	Image(width, height, pitch)
{
	cudaMalloc(&device_data, height * this->pitch * bytes_per_px);
}

Image_cuda::~Image_cuda(){
	cudaFree(device_data);
}

Image_cuda::Image_cuda(Image_cuda&& o): Image(std::move(o)){
	device_data = o.device_data;
	o.device_data = nullptr;
}

Image_cuda& Image_cuda::operator=(Image_cuda&& o){
	Image::operator=(std::move(o));
	device_data = o.device_data;
	o.device_data = nullptr;

	return *this;
}

Image_cpu::Image_cpu(): Image_cpu(0, 0, 0){ }

Image_cpu::Image_cpu(size_t width, size_t height, size_t pitch):
	Image(width, height, pitch)
{
	buf.resize(height * this->pitch * bytes_per_px);
}

Image_cpu::Image_cpu(Image_cpu&& o): Image(std::move(o)), buf(std::move(o.buf))
{

}

Image_cpu& Image_cpu::operator=(Image_cpu&& o){
	Image::operator=(std::move(o));
	buf = std::move(o.buf);

	return *this;
}

void Image_cpu::upload(Image_cuda& dst){
	cudaMemcpy2D(dst.data(), dst.get_pitch(),
			data(), pitch,
			width * bytes_per_px, height,
			cudaMemcpyHostToDevice); //TODO specify stream

	cudaStreamSynchronize(0);
}

}
