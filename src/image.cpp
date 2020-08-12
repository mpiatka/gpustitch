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

	cudaMalloc(&device_data, height * this->pitch * bytes_per_px);
}

Image::Image(Image&& o){
	device_data = o.device_data;
	width = o.width;
	height = o.height;
	pitch = o.pitch;
	bytes_per_px = o.bytes_per_px;
	o.device_data = nullptr;
}

Image& Image::operator=(Image&& o){
	device_data = o.device_data;
	width = o.width;
	height = o.height;
	pitch = o.pitch;
	bytes_per_px = o.bytes_per_px;
	o.device_data = nullptr;

	return *this;
}

Image::~Image(){
	cudaFree(device_data);
}

}
