#ifndef K_MEANS_METRIC_POLICIES_HPP
#define K_MEANS_METRIC_POLICIES_HPP

#include "layout_policies.hpp"
#include <vector>
#include <limits>
#include <cmath>


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

template<typename F = float, class LAYOUT1 = AoSLayoutPolicy<>, class LAYOUT2 = LAYOUT1>
class EuclideanSquaredMetricPolicy
{
public:
	CUDA_CALLABLE_MEMBER static F distance(const F* data1, std::size_t idx1, typename LAYOUT1::precomputed_t layoutPrecomputedConst1,
		const F* data2, std::size_t idx2, typename LAYOUT2::precomputed_t layoutPrecomputedConst2, std::size_t dim)
	{
		F sum = (F)0;
		for (std::size_t d = 0; d < dim; ++d) {
			F dist = LAYOUT1::at(data1, idx1, d, layoutPrecomputedConst1) - LAYOUT2::at(data2, idx2, d, layoutPrecomputedConst2);
			sum += dist * dist;
		}
		return sum;
	}
};


template<typename F = float, class LAYOUT1 = AoSLayoutPolicy<>, class LAYOUT2 = LAYOUT1>
class EuclideanMetricPolicy
{
public:
	CUDA_CALLABLE_MEMBER static F distance(const F* data1, std::size_t idx1, typename LAYOUT1::precomputed_t layoutPrecomputedConst1,
		const F* data2, std::size_t idx2, typename LAYOUT2::precomputed_t layoutPrecomputedConst2, std::size_t dim)
	{
		F sum = EuclideanSquaredMetricPolicy<F, LAYOUT1, LAYOUT2>::distance(data1, idx1, layoutPrecomputedConst1, data2, idx2, layoutPrecomputedConst2, dim);
#ifdef __CUDACC__
		return sqrtf(sum);
#else
		return std::sqrt(sum);
#endif
	}
};


template<class LAYOUT1, class LAYOUT2>
class EuclideanMetricPolicy<double, LAYOUT1, LAYOUT2>
{
public:
	CUDA_CALLABLE_MEMBER static double distance(const double* data1, std::size_t idx1, std::size_t layoutPrecomputedConst1,
		const double* data2, std::size_t idx2, std::size_t layoutPrecomputedConst2, std::size_t dim)
	{
		double sum = EuclideanSquaredMetricPolicy<double, LAYOUT1, LAYOUT2>::distance(data1, idx1, layoutPrecomputedConst1, data2, idx2, layoutPrecomputedConst2, dim);
#ifdef __CUDACC__
		return sqrt(sum);
#else
		return std::sqrt(sum);
#endif
	}
};


#endif
