#include <utility>
#include <cuda_runtime.h>
#include "cuda_stream.hpp"

namespace gpustitch{

Cuda_stream::Cuda_stream(){
	cudaStreamCreate(&stream);
}

Cuda_stream::~Cuda_stream(){
	if(stream != 0)
		cudaStreamDestroy(stream);
}

Cuda_stream::Cuda_stream(Cuda_stream&& o) : stream(o.stream){
	o.stream = 0;
}

Cuda_stream& Cuda_stream::operator=(Cuda_stream&& o){
	std::swap(stream, o.stream);
}

void Cuda_stream::synchronize() const{
	cudaStreamSynchronize(stream);
}

const Cuda_stream& Cuda_stream::get_default(){
	static Cuda_stream stream(0);

	return stream;
}

}
