#include "kernels.cuh"
#include "metric_policies.hpp"
#include "layout_policies.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>
#include <stdexcept>

// For some reasons we had some troubles with cudaMemset() when applied on floats, so we do this zeroing manually in kernel.
template<typename F>
__global__ void fillZeros(F* data, std::uint32_t n)
{
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) data[idx] = (F)0.0;
}

template<typename F>
void runFillZeros(F* data, std::size_t n)
{
	constexpr unsigned int blockSize = 1024;
	fillZeros<F><<<((unsigned int)n + blockSize - 1) / blockSize, blockSize>>>(data, (std::uint32_t)n);
}

template void runFillZeros<float>(float* data, std::size_t n);
template void runFillZeros<double>(double* data, std::size_t n);
template void runFillZeros<std::uint32_t>(std::uint32_t* data, std::size_t n);


/*
 * Atomic add for doubles (emulation)
 */
__device__ double atomicAddEmul(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +	__longlong_as_double(assumed)));
	} while (assumed != old);

	return __longlong_as_double(old);
}

__device__ float atomicAddEmul(float* address, float val)
{
	unsigned int* address_as_u = (unsigned int*)address;
	unsigned int old = *address_as_u, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_u, assumed, __float_as_uint(val + __uint_as_float(assumed)));
	} while (assumed != old);

	return __uint_as_float(old);
}


/*
 * Class policies (wrappers) for atomic operations.
 */
template<typename F>
class AtomicPolicy
{
public:
	__device__ static F add(F* data, F value)
	{
		return atomicAdd(data, value);
	}
};

#if __CUDA_ARCH__ < 600
// Partial specialization for doubles, if CC < 6.0 (double version of atomicAdd has to be emulated)
template<> class AtomicPolicy<double>
{
public:
	__device__ static double add(double* data, double value)
	{
		return atomicAddEmul(data, value);
	}
};
#endif


template<typename F>
class AtomicEmulPolicy
{
public:
	__device__ static F add(F* data, F value)
	{
		return atomicAddEmul(data, value);
	}
};


/*
 * --------------------------------------------------------------------------------
 * Kernels and their runners
 * --------------------------------------------------------------------------------
 */

template<typename F, typename IDX_T, class LAYOUT>
__device__ __inline__ F* __restrict__ privatizeResultPointer(F* __restrict__ result, IDX_T dim, IDX_T k, IDX_T privCopies)
{
	if (privCopies > 1) {
		result += LAYOUT::size(k, dim) * (((blockDim.x * blockIdx.x + threadIdx.x) / warpSize) % privCopies);
	}
	return result;
}


// Common reduction kernel that aggregates all privatized copies into one.
template<typename F, typename IDX_T, class LAYOUT>
__global__ void reducePrivCopiesKernel(F* __restrict__ result, IDX_T dim, IDX_T k, IDX_T privCopies)
{
	auto copySize = LAYOUT::size(k, dim);
	auto preK = LAYOUT::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	auto n = k * dim;
	for (IDX_T i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += threads) {
		IDX_T d = i % dim;
		IDX_T idx = i / dim;

		F acc = LAYOUT::at(result, idx, d, preK);
		F* copies = result;
		for (auto c = 1; c < privCopies; ++c) {
			copies += copySize;
			acc += LAYOUT::at(copies, idx, d, preK);
		}

		LAYOUT::at(result, idx, d, preK) = acc;
	}
}

template<typename F, typename IDX_T, class LAYOUT>
void runReducePrivCopiesKernel(F* __restrict__ means, IDX_T dim, IDX_T k, CudaExecParameters& exec)
{
	if (exec.privatizedCopies > 1) {
		unsigned int n = (unsigned int)(k * dim);
		unsigned int blockSize = 1024;
		reducePrivCopiesKernel<F, IDX_T, LAYOUT><<<(n + blockSize - 1) / blockSize, blockSize>>>(means, dim, k, exec.privatizedCopies);
	}
}


// Common reduction kernel that aggregates all privatized copies into one.
template<typename F, typename IDX_T, class LAYOUT_MEANS>
__global__ void divideMeansKernel(F* __restrict__ means, const IDX_T* __restrict__ clusterSizes, IDX_T dim, IDX_T k)
{
	auto size = k * dim;
	auto preK = LAYOUT_MEANS::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	for (IDX_T i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += threads) {
		IDX_T d = i % dim;
		IDX_T idx = i / dim;
		auto divisor = clusterSizes[idx];
		if (divisor > 0) {
			LAYOUT_MEANS::at(means, idx, d, preK) /= (F)divisor;
		}
	}
}

template<typename F, typename IDX_T, class LAYOUT_MEANS>
void runDivideMeansKernel(F* __restrict__ means, const IDX_T* __restrict__ clusterSizes, IDX_T dim, IDX_T k)
{
	unsigned int n = (unsigned int)(k * dim);
	unsigned int blockSize = 1024;
	divideMeansKernel<F, IDX_T, LAYOUT_MEANS><<<(n + blockSize - 1) / blockSize, blockSize>>>(means, clusterSizes, dim, k);
}

// --------------------------------------------------------------------------------


/**
 * Baseline kernel. Runs one thread for each point. Thread computes distances to all means and
 * determines the nearest one.
 */
template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = AoSLayoutPolicy<>,
	class LAYOUT_MEANS = AoSLayoutPolicy<>, class METRIC = EuclideanMetricPolicy<F, LAYOUT, LAYOUT_MEANS>>
__global__ void baseAssignmentKernel(const F* __restrict__ data, const F* __restrict__ means, IDX_T* __restrict__ assignment,
	IDX_T dim, IDX_T k, IDX_T n)
{
	IDX_T idx = threadIdx.x + blockIdx.x * blockDim.x;
	auto precomputedData = LAYOUT::precomputeConstants(n, dim);
	auto precomputedMeans = LAYOUT_MEANS::precomputeConstants(k, dim);

	F minDist = METRIC::distance(data, idx, precomputedData, means, 0, precomputedMeans, dim);
	IDX_T nearestIdx = 0;
	for (IDX_T m = 1; m < k; ++m) {
		F dist = METRIC::distance(data, idx, precomputedData, means, m, precomputedMeans, dim);
		if (minDist > dist) {
			minDist = dist;
			nearestIdx = m;
		}
	}
	assignment[idx] = nearestIdx;
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>
void BaseAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(const AssignmentProblemInstance<F, IDX_T> &in, CudaExecParameters &exec)
{
	unsigned int threads = (unsigned int)(in.n);
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	baseAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC><<<exec.blockCount, exec.blockSize>>>(in.data, in.means, in.assignment, in.dim, in.k, in.n);
}


// --------------------------------------------------------------------------------


/**
 * Kernel with fixed-size caching window. Only means are cached in shm. Temporal sums are kept in registers.
 */
template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = AoSLayoutPolicy<>,
	class LAYOUT_MEANS = AoSLayoutPolicy<>, int shmK, int shmDim>
__global__ void cachedFixedAssignmentKernel(const F* __restrict__ data, const F* __restrict__ means, IDX_T* __restrict__ assignment,
	IDX_T dim, IDX_T k, IDX_T n)
{
	volatile __shared__ F shmMeans[shmDim][shmK];

	IDX_T idx = threadIdx.x + blockIdx.x * blockDim.x;
	auto precomputedData = LAYOUT::precomputeConstants(n, dim);
	auto precomputedMeans = LAYOUT_MEANS::precomputeConstants(k, dim);

	F dists[shmK];
	F minDist = (F)100000000;
	IDX_T nearestIdx = 0;

	for (IDX_T mOffset = 0; mOffset < k; mOffset += shmK) {
		// Reset variables for new set of means.
		#pragma unroll
		for (IDX_T i = 0; i < shmK; ++i) {
			dists[i] = (F)0;
		}

		for (IDX_T dimOffset = 0; dimOffset < dim; dimOffset += shmDim) {
			// Load means to shm
			for (IDX_T i = threadIdx.x; i < shmK * shmDim; i += blockDim.x) {
				IDX_T d = i / shmK;
				IDX_T m = i % shmK;
				shmMeans[d][m] = LAYOUT_MEANS::at(means, m + mOffset, d + dimOffset, precomputedMeans);
			}

			__syncthreads();

			// Accumulate distance values
			#pragma unroll
			for (IDX_T d = 0; d < shmDim; ++d) {
				F x = LAYOUT::at(data, idx, d + dimOffset, precomputedData);
				#pragma unroll
				for (IDX_T m = 0; m < shmK; ++m) {
					F dx = x - shmMeans[d][m];
					dists[m] += dx * dx;
				}
			}

			__syncthreads();
		}

		#pragma unroll
		for (IDX_T i = 0; i < shmK; ++i) {
			F dist = sqrtf(dists[i]);
			if (minDist > dist) {
				minDist = dist;
				nearestIdx = i + mOffset;
			}
		}
	}

	assignment[idx] = nearestIdx;
}


template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC, int shmDim, int shmK>
void cachedFixedAssignmentRunnerHelper(const AssignmentProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	unsigned int threads = (unsigned int)(in.n);
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	if (in.k % shmK != 0) throw std::runtime_error("K is not multiple of shmK constant! This is just a prototype (requires aligned parameters).");
	if (in.dim % shmDim != 0) throw std::runtime_error("Dim is not multiple of shmDim constant! This is just a prototype (requires aligned parameters).");
	cachedFixedAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, shmK, shmDim><<<exec.blockCount, exec.blockSize>>>(in.data, in.means, in.assignment, in.dim, in.k, in.n);
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC, int shmDim>
void cachedFixedAssignmentRunnerHelper2(const AssignmentProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	if (in.k >= 32) {
		cachedFixedAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, shmDim, 32>(in, exec);
	}
	else {
		cachedFixedAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, shmDim, 16>(in, exec);
	}
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>
void CachedFixedAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(const AssignmentProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	if (in.dim >= 16) {
		cachedFixedAssignmentRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 16>(in, exec);
	}
	else {
		switch (in.dim) {
		case 4:
			cachedFixedAssignmentRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 4>(in, exec); break;
		case 6:
			cachedFixedAssignmentRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 6>(in, exec); break;
		case 8:
			cachedFixedAssignmentRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 8>(in, exec); break;
		case 10:
			cachedFixedAssignmentRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 10>(in, exec); break;
		case 12:
			cachedFixedAssignmentRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 12>(in, exec); break;
		default:
			throw std::runtime_error("Unsupported data dimensions. This is just a prototype.");
		}
	}
}


// --------------------------------------------------------------------------------


/**
 * All means are cached in the shared memory and current dim value if kPerThread means is cached in registers.
 */
template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = AoSLayoutPolicy<>,
	class LAYOUT_MEANS = AoSLayoutPolicy<>, int kPerThread>
__global__ void cachedAllMeansAssignmentKernel(const F* __restrict__ data, const F* __restrict__ means, IDX_T* __restrict__ assignment,
	IDX_T dim, IDX_T k, IDX_T n)
{
	__shared__ extern float shm[];
	F* shmMeans = (F*)shm;
	IDX_T shmMeansSize = k * dim;

	auto precomputedData = LAYOUT::precomputeConstants(n, dim);
	auto precomputedMeans = LAYOUT_MEANS::precomputeConstants(k, dim);

	for (IDX_T i = threadIdx.x; i < shmMeansSize; i += blockDim.x) {
		shmMeans[i] = LAYOUT_MEANS::at(means, i % k, i / k, precomputedMeans);
	}

	__syncthreads();

	IDX_T threads = blockDim.x * gridDim.x;
	for (IDX_T idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += threads) {
		F minDist = (F)100000000;
		IDX_T nearestIdx = 0;

		for (IDX_T m = 0; m < k; m += kPerThread) {
			F dists[kPerThread];
			for (IDX_T i = 0; i < kPerThread; ++i) dists[i] = (F)0.0;

			for (IDX_T d = 0; d < dim; ++d) {
				F x = LAYOUT::at(data, idx, d, precomputedData);
				for (IDX_T i = 0; i < kPerThread; ++i) {
					F diff = x - shmMeans[d * k + m + i];
					dists[i] += diff * diff;
				}
			}

			for (IDX_T i = 0; i < kPerThread; ++i) {
				F dist = sqrtf(dists[i]);
				if (minDist > dist) {
					minDist = dist;
					nearestIdx = m + i;
				}
			}
		}

		assignment[idx] = nearestIdx;
	}
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC, int regs>
void cachedAllMeansAssignmentRunnerHelper(const AssignmentProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	if (in.k % regs != 0) throw std::runtime_error("Parameter k must be divisible by parameter regsCache. This is just a prototype implementation.");
	unsigned int threads = ((unsigned int)in.n + exec.itemsPerThread - 1) / exec.itemsPerThread;
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	unsigned int desiredShm = in.k * in.dim * sizeof(F);
	if (exec.sharedMemorySize < desiredShm) throw std::runtime_error("Insifficient shared memory allowed. At least dim x k items needs to be cached.");
	exec.sharedMemorySize = desiredShm;
	cachedAllMeansAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, regs><<<exec.blockCount, exec.blockSize, desiredShm>>>(
		in.data, in.means, in.assignment, in.dim, in.k, in.n);
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>
void CachedAllMeansAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(const AssignmentProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	switch (exec.regsCache) {
	case 1:
		cachedAllMeansAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 1>(in, exec); break;
	case 2:
		cachedAllMeansAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 2>(in, exec); break;
	case 4:
		cachedAllMeansAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 4>(in, exec); break;
	case 8:
		cachedAllMeansAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 8>(in, exec); break;
	case 16:
		cachedAllMeansAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 16>(in, exec); break;
	case 24:
		cachedAllMeansAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 24>(in, exec); break;
	case 32:
		cachedAllMeansAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 32>(in, exec); break;
	default:
		throw std::runtime_error("Unsupported value of regsCache parameter.");
	}
}


// --------------------------------------------------------------------------------


/**
 * Testing kernel of no concequence ...
 */
template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = AoSLayoutPolicy<>,
	class LAYOUT_MEANS = AoSLayoutPolicy<>, int shmK, int shmDim>
__global__ void
//__launch_bounds__(1024, 2)
cached2AssignmentKernel(const F * __restrict__ data, const F * __restrict__ means, IDX_T * __restrict__ assignment,
	IDX_T dim, IDX_T k, IDX_T n)
{
	volatile __shared__ F shmMeans[shmDim][shmK];
	__shared__ extern float shm[];
	F *shmData = ((F*)shm) + threadIdx.x;

	IDX_T idx = threadIdx.x + blockIdx.x * blockDim.x;
	auto precomputedData = LAYOUT::precomputeConstants(n, dim);
	auto precomputedMeans = LAYOUT_MEANS::precomputeConstants(k, dim);

	F *s = shmData;
	for (IDX_T d = 0; d < dim; ++d) {
		*s = LAYOUT::at(data, idx, d, precomputedData);
		s += blockDim.x;
	}
		
	F dists[shmK];
	F minDist = (F)100000000;
	IDX_T nearestIdx = 0;

	for (IDX_T mOffset = 0; mOffset < k; mOffset += shmK) {
		// Reset variables for new set of means.
		#pragma unroll
		for (IDX_T i = 0; i < shmK; ++i) {
			dists[i] = (F)0;
		}

		for (IDX_T dimOffset = 0; dimOffset < dim; dimOffset += shmDim) {
			// Load means to shm
			for (IDX_T i = threadIdx.x; i < shmK * shmDim; i += blockDim.x) {
				IDX_T d = i / shmK;
				IDX_T m = i % shmK;
				shmMeans[d][m] = LAYOUT_MEANS::at(means, m + mOffset, d + dimOffset, precomputedMeans);
			}

			__syncthreads();

			// Accumulate distance values
			#pragma unroll
			for (IDX_T d = 0; d < shmDim; ++d) {
				F x = shmData[(d + dimOffset) * blockDim.x]; //LAYOUT::at(data, idx, d + dimOffset, precomputedData);
				#pragma unroll
				for (IDX_T m = 0; m < shmK; ++m) {
					F dx = x - shmMeans[d][m];
					dists[m] += dx * dx;
				}
			}

			__syncthreads();
		}

		#pragma unroll
		for (IDX_T i = 0; i < shmK; ++i) {
			F dist = sqrtf(dists[i]);
			if (minDist > dist) {
				minDist = dist;
				nearestIdx = i + mOffset;
			}
		}
	}

	assignment[idx] = nearestIdx;
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC, int shmDim, int shmK>
void cached2AssignmentRunnerHelper(const AssignmentProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	unsigned int threads = (unsigned int)(in.n);
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	if (in.k % shmK != 0) throw std::runtime_error("K is not multiple of shmK constant! This is just a prototype (requires aligned parameters).");
	if (in.dim % shmDim != 0) throw std::runtime_error("Dim is not multiple of shmDim constant! This is just a prototype (requires aligned parameters).");
	unsigned int desiredShm = in.dim * exec.blockSize * sizeof(F);
	if (exec.sharedMemorySize < desiredShm + shmDim*shmK*sizeof(F)) throw std::runtime_error("Insufficient shared memory given.");
	exec.sharedMemorySize = desiredShm;
	cachedFixedAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, shmK, shmDim><<<exec.blockCount, exec.blockSize, exec.sharedMemorySize>>>(
		in.data, in.means, in.assignment, in.dim, in.k, in.n);
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC, int shmDim>
void cached2AssignmentRunnerHelper2(const AssignmentProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	if (in.k >= 32) {
		cached2AssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, shmDim, 32>(in, exec);
	}
	else {
		cached2AssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, shmDim, 16>(in, exec);
	}
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>
void Cached2AssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(const AssignmentProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	if (in.dim >= 16) {
		cachedFixedAssignmentRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 16>(in, exec);
	}
	else {
		switch (in.dim) {
		case 4:
			cached2AssignmentRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 4>(in, exec); break;
		case 6:
			cached2AssignmentRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 6>(in, exec); break;
		case 8:
			cached2AssignmentRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 8>(in, exec); break;
		case 10:
			cached2AssignmentRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 10>(in, exec); break;
		case 12:
			cached2AssignmentRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 12>(in, exec); break;
		default:
			throw std::runtime_error("Unsupported data dimensions. This is just a prototype.");
		}
	}
}


// --------------------------------------------------------------------------------


/**
 * Assignment kernel that caches all related points in shm and iteratively caches
 * means in shm as well. Furthermore, it uses smart caching in registry, so each loaded value
 * is immediately used in more than one computation.
 */
template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = AoSLayoutPolicy<>,
	class LAYOUT_MEANS = AoSLayoutPolicy<>, int regN = 1, int regK = 1, int dimBlock = 1>
__global__ void cachedRegsAssignmentKernel(const F* __restrict__ data, const F* __restrict__ means, IDX_T* __restrict__ assignment,
	IDX_T dim, IDX_T k, IDX_T n)
{
	extern __shared__ float shm[];
	F *shmData = (F*)shm;
	F *shmMeans = shmData + (dim * blockDim.x * regN); // sizeof shmData (blockDim.x = points being cached)
	F *shmDists = shmMeans + (dim * blockDim.y * regK); // sizeof shmMeans (blockDim.y = means being cached)
	IDX_T* shmNearests = (IDX_T*)(shmDists + blockDim.x * blockDim.y); // shmDists - each thread stores on one value

	auto precomputedData = LAYOUT::precomputeConstants(n, dim);
	auto precomputedMeans = LAYOUT_MEANS::precomputeConstants(k, dim);

	IDX_T blockSize = blockDim.x * blockDim.y;
	IDX_T tIdx = threadIdx.y * blockDim.x + threadIdx.x;
	IDX_T idx = threadIdx.x + blockIdx.x * blockDim.x * regN;

	// Cache points data in shm
	for (IDX_T d = threadIdx.y; d < dim; d += blockDim.y) {
		#pragma unroll
		for (IDX_T rn = 0; rn < regN; ++rn) {
			shmData[d * blockDim.x * regN + rn * blockDim.x + threadIdx.x] = LAYOUT::at(data, idx + rn*blockDim.x, d, precomputedData);
		}
	}

	F dist[regN];
	IDX_T nearest[regN];
	#pragma unroll
	for (IDX_T rn = 0; rn < regN; ++rn) {
		dist[rn] = (F)100000000;
		nearest[rn] = k;
	}

	for (IDX_T kOffset = 0; kOffset < k; kOffset += blockDim.y*regK) {
		// Cache means in shm
		IDX_T totalMeansSize = (dim * blockDim.y * regK);
		for (IDX_T m = tIdx; m < totalMeansSize; m += blockSize) {
			shmMeans[m] = LAYOUT_MEANS::at(means, kOffset + m % (blockDim.y*regK), m / (blockDim.y * regK), precomputedMeans);
		}

		__syncthreads();

		// Compute distance between x point and y mean
			
		F sum[regN][regK];
		#pragma unroll
		for (IDX_T rn = 0; rn < regN; ++rn)
			#pragma unroll
			for (IDX_T rk = 0; rk < regK; ++rk)
				sum[rn][rk] = (F)0.0;

		const F* sd = shmData + threadIdx.x;
		const F* sm = shmMeans + threadIdx.y;
		for (IDX_T d = 0; d < dim; d += dimBlock) {
			#pragma unroll
			for (IDX_T dd = 0; dd < dimBlock; ++dd) {
				// Load data to registers
				F regData[regN];
				F regMeans[regK];

				#pragma unroll
				for (IDX_T rn = 0; rn < regN; ++rn) {
					regData[rn] = *sd;
					sd += blockDim.x;
				}
				#pragma unroll
				for (IDX_T rk = 0; rk < regK; ++rk) {
					regMeans[rk] = *sm;
					sm += blockDim.y;
				}

				#pragma unroll
				for (IDX_T rn = 0; rn < regN; ++rn)
					#pragma unroll
					for (IDX_T rk = 0; rk < regK; ++rk) {
						F diff = regData[rn] - regMeans[rk];
						sum[rn][rk] += diff * diff;
					}
			}
		}

		#pragma unroll
		for (IDX_T rn = 0; rn < regN; ++rn)
			#pragma unroll
			for (IDX_T rk = 0; rk < regK; ++rk) {
				sum[rn][rk] = sqrtf(sum[rn][rk]);
				if (sum[rn][rk] < dist[rn]) {
					dist[rn] = sum[rn][rk];
					nearest[rn] = kOffset + threadIdx.y + rk * blockDim.y;
				}
			}

		__syncthreads();
	}

	for (IDX_T rn = 0; rn < regN; ++rn) {
		// 
		shmDists[tIdx] = dist[rn];
		shmNearests[tIdx] = nearest[rn];

		__syncthreads();

		if (threadIdx.y == 0) {
			for (IDX_T i = threadIdx.x + blockDim.x; i < blockSize; i += blockDim.x) {
				if (shmDists[i] < dist[rn]) {
					dist[rn] = shmDists[i];
					nearest[rn] = shmNearests[i];
				}
			}
			assignment[idx + rn * blockDim.x] = nearest[rn];
		}

		__syncthreads();
	}
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC, int regN, int regK>
void cachedRegsAssignmentRunnerHelper(const AssignmentProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	if (in.n % exec.blockSize) {
		throw std::runtime_error("Number of items must be divisible by block size (this is just a prototype).");
	}

	unsigned int dimX = exec.itemsPerThread;
	unsigned int dimY = exec.blockSize / dimX;
	unsigned int shmN = dimX * regN;
	unsigned int shmK = dimY * regK;
	unsigned desiredShm = in.dim * (shmN + shmK) * sizeof(F) + dimX * dimY * (sizeof(F) + sizeof(IDX_T));
	if (exec.sharedMemorySize < desiredShm) throw std::runtime_error("Insifficient shared memory allowed.");
	exec.sharedMemorySize = desiredShm;
	exec.blockCount = ((unsigned int)(in.n) + shmN - 1) / shmN;

	if (in.dim % 8 == 0)
		cachedRegsAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, regN, regK, 8><<<exec.blockCount, dim3(dimX, dimY), exec.sharedMemorySize>>>(
			in.data, in.means, in.assignment, in.dim, in.k, in.n);
	else
		cachedRegsAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, regN, regK, 1><<<exec.blockCount, dim3(dimX, dimY), exec.sharedMemorySize>>>(
			in.data, in.means, in.assignment, in.dim, in.k, in.n);
}


template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>
void CachedRegsAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(const AssignmentProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	switch (exec.regsCache) {
	case 1:
		cachedRegsAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 1, 1>(in, exec); break;
	case 2:
		cachedRegsAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 2, 2>(in, exec); break;
	case 4:
		cachedRegsAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 4, 4>(in, exec); break;
	case 8:
		cachedRegsAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 8, 8>(in, exec); break;
	default:
		throw std::runtime_error("Unsupported value of regsCache parameter.");
	}
}


template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>
void BestCachedAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(const AssignmentProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	if (in.k < 64 || (in.dim < 8 && in.k < 1024) || in.dim > 64) {
		CachedFixedAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(in, exec);
	}
	else {
		exec.itemsPerThread = 64;
		if (in.dim >= 64) exec.itemsPerThread = 32;
		cachedRegsAssignmentRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 4, 4>(in, exec);
	}
}


// --------------------------------------------------------------------------------


/**
 * Simple kernel that launches one thread per source point (n) and performs the additions of the whole vector using atomic instructions.
 */
template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class ATOMIC = AtomicPolicy<F>>
__global__ void updateAtomicPointKernel(const F* __restrict__ data, const std::uint32_t* __restrict__ indices, F* __restrict__ means, IDX_T * __restrict__ clusterSizes,
	std::uint32_t dim, std::uint32_t k, std::uint32_t n, std::uint32_t privCopies)
{
	means = privatizeResultPointer<F, IDX_T, LAYOUT_MEANS>(means, dim, k, privCopies);
	auto preN = LAYOUT::precomputeConstants(n, dim);
	auto preK = LAYOUT_MEANS::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	for (std::uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += threads) {
		std::uint32_t target = indices[idx];
		for (std::uint32_t d = 0; d < dim; ++d) {
			F x = LAYOUT::at(data, idx, d, preN);
			F* res = &LAYOUT_MEANS::at(means, target, d, preK);
			ATOMIC::add(res, x);
		}
		atomicInc(&clusterSizes[target], ~(IDX_T)0);
	}
}


template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, int EMUL>
void UpdateAtomicPointKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, EMUL>::run(const UpdateProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	unsigned int threads = ((unsigned int)in.n + exec.itemsPerThread - 1) / exec.itemsPerThread;
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	if (EMUL) {
		updateAtomicPointKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, AtomicEmulPolicy<F>><<<exec.blockCount, exec.blockSize>>>(
			in.data, in.assignment, in.means, in.clusterSizes, in.dim, in.k, in.n, exec.privatizedCopies);
	}
	else {
		updateAtomicPointKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, AtomicPolicy<F>><<<exec.blockCount, exec.blockSize>>>(
			in.data, in.assignment, in.means, in.clusterSizes, in.dim, in.k, in.n, exec.privatizedCopies);
	}
	runReducePrivCopiesKernel<F, IDX_T, LAYOUT_MEANS>(in.means, in.dim, in.k, exec);
	runDivideMeansKernel<F, IDX_T, LAYOUT_MEANS>(in.means, in.clusterSizes, in.dim, in.k);
}


// --------------------------------------------------------------------------------


/**
 * Simple kernel that launches one thread per source value (n x dim) and performs the additions atomically.
 */
template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class ATOMIC = AtomicPolicy<F>>
__global__ void updateAtomicFineKernel(const F * __restrict__ data, const std::uint32_t * __restrict__ indices, F * __restrict__ means, IDX_T * __restrict__ clusterSizes,
	std::uint32_t dim, std::uint32_t k, std::uint32_t n, std::uint32_t privCopies)
{
	means = privatizeResultPointer<F, IDX_T, LAYOUT_MEANS>(means, dim, k, privCopies);
	auto preN = LAYOUT::precomputeConstants(n, dim);
	auto preK = LAYOUT_MEANS::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	auto size = n * dim;
	for (std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += threads) {
		std::uint32_t d = i % dim;
		std::uint32_t idx = i / dim;

		F x = LAYOUT::at(data, idx, d, preN);
		std::uint32_t target = indices[idx];
		F* res = &LAYOUT_MEANS::at(means, target, d, preK);
		ATOMIC::add(res, x);
		if (d == 0) {
			atomicInc(&clusterSizes[target], ~(IDX_T)0);
		}
	}
}


template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, int EMUL>
void UpdateAtomicFineKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, EMUL>::run(const UpdateProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	unsigned int threads = ((unsigned int)(in.n * in.dim) + exec.itemsPerThread - 1) / exec.itemsPerThread;
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	if (EMUL) {
		updateAtomicFineKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, AtomicEmulPolicy<F>><<<exec.blockCount, exec.blockSize>>>(
			in.data, in.assignment, in.means, in.clusterSizes, in.dim, in.k, in.n, exec.privatizedCopies);
	}
	else {
		updateAtomicFineKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, AtomicPolicy<F>><<<exec.blockCount, exec.blockSize>>>(
			in.data, in.assignment, in.means, in.clusterSizes, in.dim, in.k, in.n, exec.privatizedCopies);
	}
	runReducePrivCopiesKernel<F, IDX_T, LAYOUT_MEANS>(in.means, in.dim, in.k, exec);
	runDivideMeansKernel<F, IDX_T, LAYOUT_MEANS>(in.means, in.clusterSizes, in.dim, in.k);
}


// --------------------------------------------------------------------------------

/**
 * Kernel that launches one thread per source value (n x dim) and uses shared memory as a cache for the result.
 */
template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class ATOMIC = AtomicPolicy<F>>
__global__ void updateAtomicShmKernel(const F * __restrict__ data, const IDX_T * __restrict__ indices, F * __restrict__ means, IDX_T* __restrict__ clusterSizes,
	IDX_T dim, IDX_T k, IDX_T n, IDX_T privCopies, IDX_T cachedVectors)
{
	extern __shared__ int sharedMemory[];
	F* resultCache = (F*)sharedMemory;
	auto cacheSize = cachedVectors * dim;
	for (IDX_T i = threadIdx.x; i < cacheSize; i += blockDim.x) {
		resultCache[i] = (F)0;
	}

	__syncthreads();

	means = privatizeResultPointer<F, IDX_T, LAYOUT_MEANS>(means, dim, k, privCopies);
	auto preN = LAYOUT::precomputeConstants(n, dim);
	auto preK = LAYOUT_MEANS::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	auto size = n * dim;
	for (IDX_T i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += threads) {
		IDX_T d = i % dim;
		IDX_T idx = i / dim;

		F x = LAYOUT::at(data, idx, d, preN);
		IDX_T target = indices[idx];
		F* res = (target < cachedVectors)
			? resultCache + target * dim + d
			: &LAYOUT_MEANS::at(means, target, d, preK);
		ATOMIC::add(res, x);

		if (d == 0) {
			atomicInc(&clusterSizes[target], ~(IDX_T)0);
		}
	}

	__syncthreads();

	// Write data from shm cache to result...
	for (IDX_T i = threadIdx.x; i < cacheSize; i += blockDim.x) {
		F* res = &LAYOUT_MEANS::at(means, i / dim, i % dim, preK);
		ATOMIC::add(res, resultCache[i]);
	}
}


template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, int EMUL>
void UpdateAtomicShmKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, EMUL>::run(const UpdateProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	unsigned itemsPerThread = 64;
	unsigned int threads = ((unsigned int)(in.n * in.dim) + itemsPerThread - 1) / itemsPerThread;
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	IDX_T cachedVectors = std::min<IDX_T>(exec.sharedMemorySize / (in.dim * sizeof(F)), in.k);
	if (EMUL) {
		updateAtomicShmKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, AtomicEmulPolicy<F>><<<exec.blockCount, exec.blockSize, cachedVectors * in.dim * sizeof(F)>>>(
			in.data, in.assignment, in.means, in.clusterSizes, in.dim, in.k, in.n, exec.privatizedCopies, cachedVectors);
	}
	else {
		updateAtomicShmKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, AtomicPolicy<F>><<<exec.blockCount, exec.blockSize, cachedVectors * in.dim * sizeof(F)>>>(
			in.data, in.assignment, in.means, in.clusterSizes, in.dim, in.k, in.n, exec.privatizedCopies, cachedVectors);
	}
	runReducePrivCopiesKernel<F, IDX_T, LAYOUT_MEANS>(in.means, in.dim, in.k, exec);
	runDivideMeansKernel<F, IDX_T, LAYOUT_MEANS>(in.means, in.clusterSizes, in.dim, in.k);
}



/**
 * Fused kernel (cached fixed assignment kernel fused with atomic point kernel), 1 thread ~ point
 */
template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = AoSLayoutPolicy<>, class LAYOUT_MEANS, int shmK, int shmDim>
__global__ void fusedCachedFixedKernel(const F* __restrict__ data, const F* __restrict__ meansIn, F* __restrict__ meansOut, IDX_T *clusterSizes,
	IDX_T dim, IDX_T k, IDX_T n, std::uint32_t privCopies)
{
	volatile __shared__ F shmMeans[shmDim][shmK];

	IDX_T idx = threadIdx.x + blockIdx.x * blockDim.x;
	auto precomputedData = LAYOUT::precomputeConstants(n, dim);
	auto precomputedMeans = LAYOUT_MEANS::precomputeConstants(k, dim);

	F dists[shmK];
	F minDist = (F)100000000;
	IDX_T nearestIdx = 0;

	for (IDX_T mOffset = 0; mOffset < k; mOffset += shmK) {
		// Reset variables for new set of means.
		#pragma unroll
		for (IDX_T i = 0; i < shmK; ++i) {
			dists[i] = (F)0;
		}

		for (IDX_T dimOffset = 0; dimOffset < dim; dimOffset += shmDim) {
			// Load means to shm
			for (IDX_T i = threadIdx.x; i < shmK * shmDim; i += blockDim.x) {
				IDX_T d = i / shmK;
				IDX_T m = i % shmK;
				shmMeans[d][m] = LAYOUT_MEANS::at(meansIn, m + mOffset, d + dimOffset, precomputedMeans);
			}

			__syncthreads();

			// Accumulate distance values
			#pragma unroll
			for (IDX_T d = 0; d < shmDim; ++d) {
				F x = LAYOUT::at(data, idx, d + dimOffset, precomputedData);
				#pragma unroll
				for (IDX_T m = 0; m < shmK; ++m) {
					F dx = x - shmMeans[d][m];
					dists[m] += dx * dx;
				}
			}

			__syncthreads();
		}

		#pragma unroll
		for (IDX_T i = 0; i < shmK; ++i) {
			F dist = sqrtf(dists[i]);
			if (minDist > dist) {
				minDist = dist;
				nearestIdx = i + mOffset;
			}
		}
	}

	atomicInc(&clusterSizes[nearestIdx], ~(IDX_T)0);

	meansOut = privatizeResultPointer<F, IDX_T, LAYOUT_MEANS>(meansOut, dim, k, privCopies);
	for (std::uint32_t d = 0; d < dim; ++d) {
		F x = LAYOUT::at(data, idx, d, precomputedData);
		F* res = &LAYOUT_MEANS::at(meansOut, nearestIdx, d, precomputedMeans);
		AtomicPolicy<F>::add(res, x);
	}
}


template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC, int shmDim, int shmK>
void fusedCachedFixedKernelRunnerHelper(const FusedProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	unsigned int threads = (unsigned int)(in.n);
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	if (in.k % shmK != 0) throw std::runtime_error("K is not multiple of shmK constant! This is just a prototype (requires aligned parameters).");
	if (in.dim % shmDim != 0) throw std::runtime_error("Dim is not multiple of shmDim constant! This is just a prototype (requires aligned parameters).");
	fusedCachedFixedKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, shmK, shmDim><<<exec.blockCount, exec.blockSize>>>(
		in.data, in.meansIn, in.meansOut, in.clusterSizes, in.dim, in.k, in.n, exec.privatizedCopies);

	runReducePrivCopiesKernel<F, IDX_T, LAYOUT_MEANS>(in.meansOut, in.dim, in.k, exec);
	runDivideMeansKernel<F, IDX_T, LAYOUT_MEANS>(in.meansOut, in.clusterSizes, in.dim, in.k);
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC, int shmDim>
void fusedCachedFixedKernelRunnerHelper2(const FusedProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	if (in.k >= 32) {
		fusedCachedFixedKernelRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, shmDim, 32>(in, exec);
	}
	else {
		fusedCachedFixedKernelRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, shmDim, 16>(in, exec);
	}
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>
void FusedCachedFixedKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(const FusedProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	if (in.dim >= 16) {
		fusedCachedFixedKernelRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 16>(in, exec);
	}
	else {
		switch (in.dim) {
		case 4:
			fusedCachedFixedKernelRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 4>(in, exec); break;
		case 6:
			fusedCachedFixedKernelRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 6>(in, exec); break;
		case 8:
			fusedCachedFixedKernelRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 8>(in, exec); break;
		case 10:
			fusedCachedFixedKernelRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 10>(in, exec); break;
		case 12:
			fusedCachedFixedKernelRunnerHelper2<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 12>(in, exec); break;
		default:
			throw std::runtime_error("Unsupported data dimensions. This is just a prototype.");
		}
	}
}


// --------------------------------------------------------------------------------


/**
 * Fused kernel (cached regs assignment kernel fused with specific atomic update that is based on atomic_warp,
 * but the size of the thread group cooperating is configurable),
 * (itemsPerThread*regsCount) threads cooperate on one point in update
 */
template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = AoSLayoutPolicy<>,
	class LAYOUT_MEANS = AoSLayoutPolicy<>, int regN = 1, int regK = 1, int dimBlock = 1>
__global__ void fusedCachedRegsKernel(const F* __restrict__ data, const F* __restrict__ meansIn, F * __restrict__ meansOut, IDX_T* __restrict__ clusterSizes,
	IDX_T dim, IDX_T k, IDX_T n, std::uint32_t privCopies)
{
	extern __shared__ float shm[];
	F *shmData = (F*)shm;
	F *shmMeans = shmData + (dim * blockDim.x * regN); // sizeof shmData (blockDim.x = points being cached)
	F *shmDists = shmMeans + (dim * blockDim.y * regK); // sizeof shmMeans (blockDim.y = means being cached)
	IDX_T* shmNearests = (IDX_T*)(shmDists + blockDim.x * blockDim.y); // shmDists - each thread stores on one value

	auto precomputedData = LAYOUT::precomputeConstants(n, dim);
	auto precomputedMeans = LAYOUT_MEANS::precomputeConstants(k, dim);

	IDX_T blockSize = blockDim.x * blockDim.y;
	IDX_T tIdx = threadIdx.y * blockDim.x + threadIdx.x;
	IDX_T idx = threadIdx.x + blockIdx.x * blockDim.x * regN;

	// Cache points data in shm
	for (IDX_T d = threadIdx.y; d < dim; d += blockDim.y) {
		#pragma unroll
		for (IDX_T rn = 0; rn < regN; ++rn) {
			shmData[d * blockDim.x * regN + rn * blockDim.x + threadIdx.x] = LAYOUT::at(data, idx + rn*blockDim.x, d, precomputedData);
		}
	}

	F dist[regN];
	IDX_T nearest[regN];
	#pragma unroll
	for (IDX_T rn = 0; rn < regN; ++rn) {
		dist[rn] = (F)100000000;
		nearest[rn] = k;
	}

	for (IDX_T kOffset = 0; kOffset < k; kOffset += blockDim.y*regK) {
		// Cache means in shm
		IDX_T totalMeansSize = (dim * blockDim.y * regK);
		for (IDX_T m = tIdx; m < totalMeansSize; m += blockSize) {
			shmMeans[m] = LAYOUT_MEANS::at(meansIn, kOffset + m % (blockDim.y*regK), m / (blockDim.y * regK), precomputedMeans);
		}

		__syncthreads();

		// Compute distance between x point and y mean
			
		F sum[regN][regK];
		#pragma unroll
		for (IDX_T rn = 0; rn < regN; ++rn)
			#pragma unroll
			for (IDX_T rk = 0; rk < regK; ++rk)
				sum[rn][rk] = (F)0.0;

		const F* sd = shmData + threadIdx.x;
		const F* sm = shmMeans + threadIdx.y;
		for (IDX_T d = 0; d < dim; d += dimBlock) {
			#pragma unroll
			for (IDX_T dd = 0; dd < dimBlock; ++dd) {
				// Load data to registers
				F regData[regN];
				F regMeans[regK];

				#pragma unroll
				for (IDX_T rn = 0; rn < regN; ++rn) {
					regData[rn] = *sd;
					sd += blockDim.x;
				}
				#pragma unroll
				for (IDX_T rk = 0; rk < regK; ++rk) {
					regMeans[rk] = *sm;
					sm += blockDim.y;
				}

				#pragma unroll
				for (IDX_T rn = 0; rn < regN; ++rn)
					#pragma unroll
					for (IDX_T rk = 0; rk < regK; ++rk) {
						F diff = regData[rn] - regMeans[rk];
						sum[rn][rk] += diff * diff;
					}
			}
		}

		#pragma unroll
		for (IDX_T rn = 0; rn < regN; ++rn)
			#pragma unroll
			for (IDX_T rk = 0; rk < regK; ++rk) {
				sum[rn][rk] = sqrtf(sum[rn][rk]);
				if (sum[rn][rk] < dist[rn]) {
					dist[rn] = sum[rn][rk];
					nearest[rn] = kOffset + threadIdx.y + rk * blockDim.y;
				}
			}

		__syncthreads();
	}

	meansOut = privatizeResultPointer<F, IDX_T, LAYOUT_MEANS>(meansOut, dim, k, privCopies);

	#pragma unroll
	for (IDX_T rn = 0; rn < regN; ++rn) {
		// 
		shmDists[tIdx] = dist[rn];
		shmNearests[tIdx] = nearest[rn];

		__syncthreads();

		if (threadIdx.y == 0) {
			for (IDX_T i = threadIdx.x + blockDim.x; i < blockSize; i += blockDim.x) {
				if (shmDists[i] < dist[rn]) {
					dist[rn] = shmDists[i];
					nearest[rn] = shmNearests[i];
				}
			}
			//				assignment[idx + rn * blockDim.x] = nearest[rn];
			shmNearests[tIdx] = nearest[rn];
		}

		__syncthreads();

		if (threadIdx.y == 0) {
			atomicInc(&clusterSizes[nearest[rn]], ~(IDX_T)0);
		}
		else {
			nearest[rn] = shmNearests[threadIdx.x]; // broadcast index to cooperating threads
		}

		// All threads cooperate to atomically update output means
		for (std::uint32_t d = threadIdx.y; d < dim; d += blockDim.y) {
			F x = shmData[d * blockDim.x * regN + rn * blockDim.x + threadIdx.x];
			F* res = &LAYOUT_MEANS::at(meansOut, nearest[rn], d, precomputedMeans);
			AtomicPolicy<F>::add(res, x);
		}

		__syncthreads();
	}
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC, int regN, int regK>
void fusedCachedRegsRunnerHelper(const FusedProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	if (in.n % exec.blockSize) {
		throw std::runtime_error("Number of items must be divisible by block size (this is just a prototype).");
	}

	unsigned int dimX = exec.itemsPerThread;
	unsigned int dimY = exec.blockSize / dimX;
	unsigned int shmN = dimX * regN;
	unsigned int shmK = dimY * regK;
	unsigned desiredShm = in.dim * (shmN + shmK) * sizeof(F) + dimX * dimY * (sizeof(F) + sizeof(IDX_T));
	if (exec.sharedMemorySize < desiredShm) throw std::runtime_error("Insifficient shared memory allowed.");
	exec.sharedMemorySize = desiredShm;
	exec.blockCount = ((unsigned int)(in.n) + shmN - 1) / shmN;

	if (in.dim % 8 == 0)
		fusedCachedRegsKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, regN, regK, 8><<<exec.blockCount, dim3(dimX, dimY), exec.sharedMemorySize>>>(
			in.data, in.meansIn, in.meansOut, in.clusterSizes, in.dim, in.k, in.n, exec.privatizedCopies);
	else
		fusedCachedRegsKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, regN, regK, 1><<<exec.blockCount, dim3(dimX, dimY), exec.sharedMemorySize>>>(
			in.data, in.meansIn, in.meansOut, in.clusterSizes, in.dim, in.k, in.n, exec.privatizedCopies);

	runReducePrivCopiesKernel<F, IDX_T, LAYOUT_MEANS>(in.meansOut, in.dim, in.k, exec);
	runDivideMeansKernel<F, IDX_T, LAYOUT_MEANS>(in.meansOut, in.clusterSizes, in.dim, in.k);
}



template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>
void FusedCachedRegsKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(const FusedProblemInstance<F, IDX_T>& in, CudaExecParameters& exec)
{
	switch (exec.regsCache) {
	case 1:
		fusedCachedRegsRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 1, 1>(in, exec); break;
	case 2:
		fusedCachedRegsRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 2, 2>(in, exec); break;
	case 4:
		fusedCachedRegsRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 4, 4>(in, exec); break;
	case 8:
		fusedCachedRegsRunnerHelper<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, 8, 8>(in, exec); break;
	default:
		throw std::runtime_error("Unsupported value of regsCache parameter.");
	}
}


// --------------------------------------------------------------------------------


/*
 * All kernels should be registered here...
 */

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>
void instantiateAssignmentKernelRunnerTemplates(AssignmentProblemInstance<F,IDX_T> &instance, CudaExecParameters &exec)
{
	BaseAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(instance, exec);
	CachedFixedAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(instance, exec);
	CachedAllMeansAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(instance, exec);
	Cached2AssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(instance, exec);
	CachedRegsAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(instance, exec);
	BestCachedAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(instance, exec);
}

template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>
void instantiateKernelRunnerTemplates()
{
	// This is fake routine (just to enforce explicit template instantiations).
	// No code will be actually executed.
	AssignmentProblemInstance<F, IDX_T> assignmentInstance(nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
	CudaExecParameters exec;

	instantiateAssignmentKernelRunnerTemplates<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>(assignmentInstance, exec);

	FusedProblemInstance<F, IDX_T> fusedInstance(nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
	FusedCachedFixedKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(fusedInstance, exec);
	FusedCachedRegsKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>::run(fusedInstance, exec);
}


// Machinery that instantiates all templated versions of kernel runners...
template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS>
void instantiateKernelRunnerTemplatesHelper1()
{
	UpdateProblemInstance<F, IDX_T> updateInstance(nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
	CudaExecParameters exec;

	UpdateAtomicPointKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS>::run(updateInstance, exec);
	UpdateAtomicFineKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS>::run(updateInstance, exec);
	UpdateAtomicShmKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS>::run(updateInstance, exec);

	instantiateKernelRunnerTemplates<F, IDX_T, LAYOUT, LAYOUT_MEANS, EuclideanMetricPolicy<F, LAYOUT, LAYOUT_MEANS>>();
	instantiateKernelRunnerTemplates<F, IDX_T, LAYOUT, LAYOUT_MEANS, EuclideanSquaredMetricPolicy<F, LAYOUT, LAYOUT_MEANS>>();
}

template<typename F, int ALIGN>
void instantiateKernelRunnerTemplatesHelper2()
{
	instantiateKernelRunnerTemplatesHelper1<F, std::uint32_t, SoALayoutPolicy<ALIGN>, SoALayoutPolicy<ALIGN>>();
	instantiateKernelRunnerTemplatesHelper1<F, std::uint32_t, AoSLayoutPolicy<ALIGN>, AoSLayoutPolicy<ALIGN>>();
	instantiateKernelRunnerTemplatesHelper1<F, std::uint32_t, SoALayoutPolicy<ALIGN>, AoSLayoutPolicy<ALIGN>>();
	instantiateKernelRunnerTemplatesHelper1<F, std::uint32_t, AoSLayoutPolicy<ALIGN>, SoALayoutPolicy<ALIGN>>();
}

template<typename F>
void instantiateKernelRunnerTemplatesHelper3()
{
	instantiateKernelRunnerTemplatesHelper2<F, 1>();
//	instantiateKernelRunnerTemplatesHelper2<F, 32>();
}

template void instantiateKernelRunnerTemplatesHelper3<float>();
#ifndef NO_DOUBLES
template void instantiateKernelRunnerTemplatesHelper3<double>();
#endif
