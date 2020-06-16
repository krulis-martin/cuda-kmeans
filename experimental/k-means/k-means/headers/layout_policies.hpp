#ifndef K_MEANS_LAYOUT_POLICIES_HPP
#define K_MEANS_LAYOUT_POLICIES_HPP

#include <vector>
#include <random>
#include <limits>


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

template<int ALIGN = 32>
class SoALayoutPolicy
{
public:
	using precomputed_t = std::size_t;

	template<typename F>
	CUDA_CALLABLE_MEMBER static F& at(F* data, std::size_t idx, std::size_t dim, precomputed_t precomputed)
	{
		return data[dim * precomputed + idx];
	}

	CUDA_CALLABLE_MEMBER static std::size_t size(std::size_t n, std::size_t dims)
	{
		return ((n + ALIGN - 1) / ALIGN) * ALIGN * dims;
	}

	CUDA_CALLABLE_MEMBER static precomputed_t precomputeConstants(std::size_t n, std::size_t dims)
	{
		return ((n + ALIGN - 1) / ALIGN) * ALIGN;
	}
};


template<int ALIGN = 1>
class AoSLayoutPolicy
{
public:
	using precomputed_t = std::size_t;

	template<typename F>
	CUDA_CALLABLE_MEMBER static F& at(F* data, std::size_t idx, std::size_t dim, precomputed_t precomputed)
	{
		return data[idx * precomputed + dim];
	}

	CUDA_CALLABLE_MEMBER static std::size_t size(std::size_t n, std::size_t dims)
	{
		return ((dims + ALIGN - 1) / ALIGN) * ALIGN * n;
	}

	CUDA_CALLABLE_MEMBER static precomputed_t precomputeConstants(std::size_t n, std::size_t dims)
	{
		return ((dims + ALIGN - 1) / ALIGN) * ALIGN;
	}
};



#endif
