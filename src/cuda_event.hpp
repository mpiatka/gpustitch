#ifndef CUDA_EVENT_HPP
#define CUDA_EVENT_HPP

#include <cuda_runtime.h>
#include "gpustitch_common.hpp"

namespace gpustitch{

class Cuda_stream;

class Cuda_event{
public:
	Cuda_event();
	~Cuda_event();
	Cuda_event(const Cuda_event&) = delete;
	Cuda_event(Cuda_event&& o);
	Cuda_event& operator=(const Cuda_event&) = delete;
	Cuda_event& operator=(Cuda_event&& o);

	cudaEvent_t get() const { return event; }
	void record(const Cuda_stream& stream);

private:
	cudaEvent_t event = nullptr;
};

}

#endif
