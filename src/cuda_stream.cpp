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

Cuda_stream::Cuda_stream(Cuda_stream&& o){
	stream = o.stream;
	o.stream = 0;
}

Cuda_stream& Cuda_stream::operator=(Cuda_stream&& o){
	stream = o.stream;
	o.stream = 0;
}

void Cuda_stream::synchronize() const{
	cudaStreamSynchronize(stream);
}

}
