#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <vector>
#include "gpustitch_common.hpp"
#include "cuda_stream.hpp"

namespace gpustitch{

class Image{
public:
	size_t get_width() const { return width; }
	size_t get_height() const { return height; }
	size_t get_pitch() const { return pitch; }
	size_t get_bytes_per_px() const { return bytes_per_px; }
	size_t get_row_bytes() const { return bytes_per_px * width; }

protected:
	Image();
	Image(size_t width, size_t height, size_t pitch = 0);
	Image(const Image&) = delete;
	Image(Image&& o);
	Image& operator=(const Image&) = delete;
	Image& operator=(Image&&);

	size_t width;
	size_t height;
	size_t pitch;
	size_t bytes_per_px;
};

class Image_cuda;

class Image_cpu : public Image{
public:
	Image_cpu();
	Image_cpu(size_t width, size_t height, size_t pitch = 0);
	Image_cpu(const Image_cpu&) = delete;
	Image_cpu(Image_cpu&& o);
	Image_cpu& operator=(const Image_cpu&) = delete;
	Image_cpu& operator=(Image_cpu&&);
	void *data() { return buf.data(); }
	const void *data() const { return buf.data(); }

	void upload(Image_cuda& dst);

private:
	std::vector<unsigned char> buf;
	
};

class Image_cuda : public Image{
public:
	Image_cuda();
	~Image_cuda();
	Image_cuda(size_t width, size_t height, size_t pitch = 0);
	Image_cuda(const Image_cuda&) = delete;
	Image_cuda(Image_cuda&& o);
	Image_cuda& operator=(const Image_cuda&) = delete;
	Image_cuda& operator=(Image_cuda&&);
	void *data() { return device_data; }
	const void *data() const { return device_data; }

private:
	void *device_data = nullptr;
	
};

void copy_image(Image_cuda *dst, const Image_cuda *src,
		int dst_x, int dst_y, int src_x, int src_y, int w, int h,
		const Cuda_stream& stream);

}


#endif
