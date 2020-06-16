#ifndef BUCKETING_CUDA_LAYOUT_TRAITS_HPP
#define BUCKETING_CUDA_LAYOUT_TRAITS_HPP

#include <vector>
#include <random>
#include <limits>

template<int ALIGN = 32>
class SoALayoutTrait
{
public:
	template<typename F>
	static F& at(std::vector<F>& data, std::size_t idx, std::size_t dim, std::size_t precomputed)
	{
		return data[dim * precomputed + idx];
	}

	static std::size_t size(std::size_t n, std::size_t dims)
	{
		return ((n + ALIGN - 1) / ALIGN) * ALIGN * dims;
	}

	static std::size_t precomputeConstants(std::size_t n, std::size_t dims)
	{
		return ((n + ALIGN - 1) / ALIGN) * ALIGN;
	}
};





#endif
