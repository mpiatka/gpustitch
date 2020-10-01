#include <cassert>
#include <cuda_runtime.h>
#include "image.hpp"
#include "profile_timer.hpp"
#include "cuda_helpers.h"

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

void Image_cuda::init_to(int r, int g, int b, int a,
		const Cuda_stream& stream)
{
	//cudaMemset2DAsync(device_data, pitch, val, width * bytes_per_px, height, stream.get());
	cuda_image_memset(this, 0, 0, width, height, r, g, b, a, stream.get());
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

void copy_image(Image_cuda *dst, const Image_cuda *src,
		unsigned dst_x, unsigned dst_y, unsigned src_x, unsigned src_y,
		unsigned w, unsigned h,
		const Cuda_stream& stream)
{
	assert(dst_x + w <= dst->get_width());
	assert(dst_y + h <= dst->get_height());
	assert(src_x + w <= src->get_width());
	assert(src_y + h <= src->get_height());
	assert(dst->get_bytes_per_px() == src->get_bytes_per_px());

	const unsigned char *src_p = static_cast<const unsigned char *>(src->data())
		+ src_y * src->get_pitch()
		+ src_x * src->get_bytes_per_px();

	unsigned char *dst_p = static_cast<unsigned char *>(dst->data())
		+ dst_y * dst->get_pitch()
		+ dst_x * dst->get_bytes_per_px();

	cudaError_t res;
	res = cudaMemcpy2DAsync(dst_p, dst->get_pitch(),
			src_p, src->get_pitch(),
			w * dst->get_bytes_per_px(), h,
			cudaMemcpyDeviceToDevice,
			stream.get());
}

}
