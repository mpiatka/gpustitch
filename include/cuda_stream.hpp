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

	static const Cuda_stream& get_default();

	CUstream_st *get() const { return stream; }
	void synchronize() const;

private:
	Cuda_stream(CUstream_st *s) : stream(s) {  }
	CUstream_st *stream = 0;

};

}


#endif
