#ifndef STITCHER_HPP
#define STITCHER_HPP

#include <vector>
#include <memory>
#include "cam_params.hpp"
#include "stitcher_params.hpp"
#include "image.hpp"
#include "gpustitch_common.hpp"

namespace gpustitch{

class Stitcher_impl;

/**
 * Enum representing memory location
 */
enum class Src_mem_kind{
	Host = 0, /**< Memory addressable from CPU */
	Device /**< Memory addressable from GPU */
};

/**
 * PImpl Stitcher interface
 *
 * This class is used to construct and use the stitcher.
 */
class Stitcher{
public:
	/**
	 * Construct and initialize stitcher instance
	 *
	 * @param stitch_params structure containing stitching properties
	 * @param cam_params parameters of used cameras and lenses
	 */
	Stitcher(Stitcher_params stitch_params, const std::vector<Cam_params>& cam_params);
	Stitcher();
	~Stitcher();

	Stitcher& operator=(Stitcher&&);

	/**
	 * Returns an Image pointer that can be used to read the stitching result
	 * 
	 * Reading the result should be performed or synchronized using the output
	 * stream.
	 *
	 * @see get_output_stream()
	 */
	Image_cuda *get_output_image();

	/**
	 * Returns CUDA stream for writing the input image
	 *
	 * @param cam_idx Index of target camera
	 * @param stream The result is written here
	 */
	void get_input_stream(size_t cam_idx, CUstream_st **stream) const;

	/**
	 * Returns CUDA stream for reading the output image
	 *
	 * @param stream The result is written here
	 */
	void get_output_stream(CUstream_st **stream) const;

	/**
	 * Copies specified image to input buffer and submits it
	 *
	 * Submitting an image means that the computations involving this
	 * image can be started.
	 *
	 * @param cam_idx Index of target camera
	 * @param data pointer to buffer containing the image data
	 * @param w width of input image
	 * @param h height of input image
	 * @param pitch pitch of input image
	 * @param mem_kind kind of memory where input image is stored
	 */
	void submit_input_image(size_t cam_idx, const void *data,
			size_t w, size_t h, size_t pitch,
			Src_mem_kind mem_kind = Src_mem_kind::Host);

	/**
	 * Asynchronous version of submit_input_image()
	 *
	 * Returns before the operation is completed. The data pointed to by the parameter
	 * data should not be modified until a manual synchronization using the specified
	 * stream is performed
	 */
	void submit_input_image_async(size_t cam_idx, const void *data,
			size_t w, size_t h, size_t pitch,
			Src_mem_kind mem_kind = Src_mem_kind::Host);

	void submit_input_image_async(size_t cam_idx);

	/**
	 * Writes the resulting image to a CPU addressable buffer
	 *
	 * Should be called after stitch(). Synchronization is performed
	 * automatically. The pixel format of the resulting image is RGBA
	 *
	 * @param dst Pointer to destination buffer
	 * @param pitch pitch of the destination image buffer
	 */
	void download_stitched(void *dst, size_t pitch);

	/**
	 * Initiaites the stitching process
	 *
	 * Should be called after all input images are submitted. Synchronization
	 * with asynchronous input image submissions is automatic.
	 */
	void stitch();
private:
	std::unique_ptr<Stitcher_impl> impl;
};

}

#endif
