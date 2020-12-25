#include <utility>
#include "cuda_event.hpp"
#include "cuda_stream.hpp"

namespace gpustitch{

Cuda_event::Cuda_event(){
	cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
}

Cuda_event::~Cuda_event(){
	if(event){
		cudaEventDestroy(event);
	}
}

Cuda_event::Cuda_event(Cuda_event&& o) : event(o.event){
	o.event = nullptr;
}

Cuda_event& Cuda_event::operator=(Cuda_event&& o){
	std::swap(event, o.event);
	return *this;
}

void Cuda_event::record(const Cuda_stream& stream){
	cudaEventRecord(event, stream.get());
}


}
