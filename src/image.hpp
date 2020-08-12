#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <vector>

namespace gpustitch{

class Image{
public:
	Image();
	Image(size_t width, size_t height, size_t pitch = 0);
	~Image();

	Image(const Image&) = delete;
	Image(Image&& o);
	Image& operator=(const Image&) = delete;
	Image& operator=(Image&&);

	size_t get_width() const { return width; }
	size_t get_height() const { return height; }
	size_t get_pitch() const { return pitch; }
	size_t get_bytes_per_px() const { return bytes_per_px; }

	void *data() { return device_data; }
	const void *data() const { return device_data; }

private:
	size_t width;
	size_t height;
	size_t pitch;
	size_t bytes_per_px;

	void *device_data = nullptr;
};

}


#endif
