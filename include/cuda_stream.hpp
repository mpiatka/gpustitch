#ifndef CUDA_STREAM_HPP
#define CUDA_STREAM_HPP

#include "gpustitch_common.hpp"

namespace gpustitch{

class Cuda_event;

/**
 * RAII wrapper for CUDA streams
 */
class Cuda_stream{
public:
	/**
	 * Constructs and initializes a new CUDA stream
	 */
	Cuda_stream();
	~Cuda_stream();
	Cuda_stream(const Cuda_stream&) = delete;
	Cuda_stream(Cuda_stream&& o);
	Cuda_stream& operator=(const Cuda_stream&) = delete;
	Cuda_stream& operator=(Cuda_stream&& o);

	/**
	 * Get the default CUDA stream
	 */
	static const Cuda_stream& get_default();

	/**
	 * Returns the raw CUDA stream reference
	 */
	CUstream_st *get() const { return stream; }

	/**
	 * Calls cudaStreamSynchronize() on the contained stream
	 */
	void synchronize() const;
	/**
	 * Waits for CUDA event using cudaStreamWaitEvent()
	 * 
	 * @param event CUDA event to wait for
	 */
	void wait_event(const Cuda_event& event) const;

private:
	Cuda_stream(CUstream_st *s) : stream(s) {  }
	CUstream_st *stream = 0;

};

}


#endif
