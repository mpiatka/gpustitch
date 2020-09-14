#ifndef CUDA_STREAM_HPP
#define CUDA_STREAM_HPP

#include "gpustitch_common.hpp"

namespace gpustitch{

class Cuda_stream{
public:
	Cuda_stream();
	~Cuda_stream();
	Cuda_stream(const Cuda_stream&) = delete;
	Cuda_stream(Cuda_stream&& o);
	Cuda_stream& operator=(const Cuda_stream&) = delete;
	Cuda_stream& operator=(Cuda_stream&& o);

	CUstream_st *get() const { return stream; }
	void synchronize() const;

private:
	CUstream_st *stream;

};

}


#endif
