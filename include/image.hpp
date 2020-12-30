#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <vector>
#include "gpustitch_common.hpp"
#include "cuda_stream.hpp"

namespace gpustitch{

/**
 * Abstract base class representing an image.
 * @see Image_cuda
 * @see Image_cpu
 */
class Image{
public:
	/**
	 * Returns width of contained image 
	 */
	size_t get_width() const { return width; }

	/**
	 * Returns height of contained image 
	 */
	size_t get_height() const { return height; }

	/**
	 * Returns pitch of contained image 
	 */
	size_t get_pitch() const { return pitch; }

	/**
	 * Returns bytes per pixel of contained image 
	 */
	size_t get_bytes_per_px() const { return bytes_per_px; }

	/**
	 * Returns bytes per row of contained image 
	 */
	size_t get_row_bytes() const { return bytes_per_px * width; }

protected:
	Image();
	virtual ~Image() = default;
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

/**
 * Class reprezenting an image stored in memory addressable by cpu
 */
class Image_cpu : public Image{
public:
	Image_cpu();
	Image_cpu(size_t width, size_t height, size_t pitch = 0);
	Image_cpu(const Image_cpu&) = delete;
	Image_cpu(Image_cpu&& o);
	Image_cpu& operator=(const Image_cpu&) = delete;
	Image_cpu& operator=(Image_cpu&&);

	/**
	 * Returns a pointer to beginning of the buffer used to store image data
	 */
	void *data() { return buf.data(); }
	const void *data() const { return buf.data(); }

	/**
	 * Copies contained image data to target Image_cuda image
	 * @param dst destination image
	 * @see Image_cuda
	 */
	void upload(Image_cuda& dst);

private:
	std::vector<unsigned char> buf;
	
};

/**
 * Class reprezenting an image stored in GPU memory.
 */
class Image_cuda : public Image{
public:
	Image_cuda();
	~Image_cuda();
	Image_cuda(size_t width, size_t height, size_t pitch = 0);
	Image_cuda(const Image_cuda&) = delete;
	Image_cuda(Image_cuda&& o);
	Image_cuda& operator=(const Image_cuda&) = delete;
	Image_cuda& operator=(Image_cuda&&);

	/**
	 * Returns a pointer to beginning of the buffer used to store image data
	 */
	void *data() { return device_data; }
	const void *data() const { return device_data; }

	/**
	 * Sets every pixel of contained image to provided values
	 */
	void init_to(int r, int g, int b, int a,
			const Cuda_stream& stream = Cuda_stream::get_default());

private:
	void *device_data = nullptr;
	
};

/**
 * Helper function to copy the contents of Image_cuda images
 *
 * @param dst destination image
 * @param src source image
 * @param dst_x starting x coord of destination image
 * @param dst_y starting y coord of destination image
 * @param src_x starting x coord of source image
 * @param src_y starting y coord of source image
 * @param stream CUDA stream to use
 */
void copy_image(Image_cuda *dst, const Image_cuda *src,
		unsigned dst_x, unsigned dst_y, unsigned src_x, unsigned src_y,
		unsigned w, unsigned h,
		const Cuda_stream& stream);

}


#endif
