#ifndef GAUSSIAN_KERNEL_HPP
#define GAUSSIAN_KERNEL_HPP

#include <cmath>

namespace gpustitch{

template<size_t TSize>
class Gaussian_kernel{
public:
	Gaussian_kernel(float sigma){
		static_assert(TSize % 2 == 1, "Kernel size must be odd");

		float sum = 0;
		for(size_t i = 0; i < TSize; i++){
			int x = i - (TSize / 2);

			kernel[i] = std::exp(static_cast<float>(-x * x) / (2 * sigma * sigma));
			sum += kernel[i];
		}

		for(size_t i = 0; i < TSize; i++){
			kernel[i] /= sum;
		}
	}

	const float *get() const { return kernel; }
	constexpr size_t size() const { return TSize; }

private:
	float kernel[TSize];
};


}


#endif
