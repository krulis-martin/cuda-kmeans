#include "kernels.cuh"
#include "layout_traits.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdexcept>
#include <cstdint>

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
 * Class traits (wrappers) for atomic operations.
 */
template<typename F>
class AtomicTrait
{
public:
	__device__ static F add(F* data, F value)
	{
		return atomicAdd(data, value);
	}
};

#if __CUDA_ARCH__ < 600
// Partial specialization for doubles, if CC < 6.0 (double version of atomicAdd has to be emulated)
template<> class AtomicTrait<double>
{
public:
	__device__ static double add(double* data, double value)
	{
		return atomicAddEmul(data, value);
	}
};
#endif


template<typename F>
class AtomicEmulTrait
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

// Common part of all kernel runners have been separated into this macro.
#define KERNEL_RUNNER_HELPER(KERNEL_FNC, TEMPLATE_F)\
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;\
	KERNEL_FNC<TEMPLATE_F, LAYOUT><<<exec.blockCount, exec.blockSize>>>(in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies);\
	if (exec.privatizedCopies > 1) {\
		unsigned int n = (unsigned int)(in.k * in.dim);\
		unsigned int blockSize = 1024;\
		reducePrivCopiesKernel<TEMPLATE_F, LAYOUT><<<(n + blockSize - 1) / blockSize, blockSize>>>(in.result, in.dim, in.k, exec.privatizedCopies);\
	}\

template<typename F, class LAYOUT>
__device__ __inline__ F* __restrict__ privatizeResultPointer(F* __restrict__ result, std::uint32_t dim, std::uint32_t k, std::uint32_t privCopies)
{
	if (privCopies > 1) {
		result += LAYOUT::size(k, dim) * (((blockDim.x * blockIdx.x + threadIdx.x) / warpSize) % privCopies);
	}
	return result;
}


// Common reduction kernel that aggregates all privatized copies into one.
template<typename F, class LAYOUT>
__global__ void reducePrivCopiesKernel(F* __restrict__ result, std::uint32_t dim, std::uint32_t k, std::uint32_t privCopies)
{
	auto copySize = LAYOUT::size(k, dim);
	auto preK = LAYOUT::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	auto n = k * dim;
	for (std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += threads) {
		std::uint32_t d = i % dim;
		std::uint32_t idx = i / dim;

		F acc = LAYOUT::at(result, idx, d, preK);
		F* copies = result;
		for (auto c = 1; c < privCopies; ++c) {
			copies += copySize;
			acc += LAYOUT::at(copies, idx, d, preK);
		}

		LAYOUT::at(result, idx, d, preK) = acc;
	}
}

template<typename F, class LAYOUT>
void runReducePrivCopiesKernel(const ProblemInstance<F>& in, CudaExecParameters& exec)
{
	if (exec.privatizedCopies > 1) {
		unsigned int n = (unsigned int)(in.k * in.dim);
		unsigned int blockSize = 1024;
		reducePrivCopiesKernel<F, LAYOUT><<<(n + blockSize - 1) / blockSize, blockSize>>>(in.result, in.dim, in.k, exec.privatizedCopies);
	}
}


// --------------------------------------------------------------------------------


/**
 * Baseline kernel. Runs one thread for each result float value that iterates over n values in its dimension,
 * filters out irrelevant values, and sums the relevant ones.
 */
template<typename F, class LAYOUT>
__global__ void baseKernel(const F* __restrict__ data, const std::uint32_t* __restrict__ indices, F* __restrict__ result, std::uint32_t dim, std::uint32_t k, std::uint32_t n)
{
	std::uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	std::uint32_t d = idx % dim;
	idx /= dim;
	if (idx >= k) return;

	auto precomputed = LAYOUT::precomputeConstants(n, dim);
	F res = (F)0.0;
	for (std::uint32_t i = 0; i < n; ++i) {
		if (indices[i] != idx) continue;
		res += LAYOUT::at(data, i, d, precomputed);
	}
	LAYOUT::at(result, idx, d, LAYOUT::precomputeConstants(k, dim)) = res;
}

template<typename F, class LAYOUT>
void BaseKernel<F, LAYOUT>::run(const ProblemInstance<F> &in, CudaExecParameters &exec)
{
	unsigned int threads = (unsigned int)(in.k * in.dim);
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	baseKernel<F, LAYOUT><<<exec.blockCount, exec.blockSize>>>(in.data, in.indices, in.result, in.dim, in.k, in.n);
}


// --------------------------------------------------------------------------------


/**
 * Simple kernel that launches one thread per source point (n) and performs the additions of the whole vector using atomic instructions.
 */
template<typename F, class LAYOUT, class ATOMIC = AtomicTrait<F>>
__global__ void atomicDimKernel(const F* __restrict__ data, const std::uint32_t* __restrict__ indices, F* __restrict__ result,
	std::uint32_t dim, std::uint32_t k, std::uint32_t n, std::uint32_t privCopies)
{
	result = privatizeResultPointer<F, LAYOUT>(result, dim, k, privCopies);
	auto preN = LAYOUT::precomputeConstants(n, dim);
	auto preK = LAYOUT::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	for (std::uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += threads) {
		for (std::uint32_t d = 0; d < dim; ++d) {
			F x = LAYOUT::at(data, idx, d, preN);
			std::uint32_t target = indices[idx];
			F* res = &LAYOUT::at(result, target, d, preK);
			ATOMIC::add(res, x);
		}
	}
}


template<typename F, class LAYOUT, int EMUL>
void AtomicDimKernel<F, LAYOUT, EMUL>::run(const ProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int threads = ((unsigned int)in.n + exec.itemsPerThread - 1) / exec.itemsPerThread;
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	if (EMUL) {
		atomicDimKernel<F, LAYOUT, AtomicEmulTrait<F>><<<exec.blockCount, exec.blockSize>>>(in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies);
	}
	else {
		atomicDimKernel<F, LAYOUT, AtomicTrait<F>><<<exec.blockCount, exec.blockSize>>>(in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies);
	}
	runReducePrivCopiesKernel<F, LAYOUT>(in, exec);
}


// --------------------------------------------------------------------------------

/**
 * Simple kernel that launches one warp of threads per source point (n) and performs the additions using atomic instructions.
 */
template<typename F, class LAYOUT, class ATOMIC = AtomicTrait<F>>
__global__ void atomicWarpDimKernel(const F* __restrict__ data, const std::uint32_t* __restrict__ indices, F* __restrict__ result,
	std::uint32_t dim, std::uint32_t k, std::uint32_t n, std::uint32_t privCopies)
{
	result = privatizeResultPointer<F, LAYOUT>(result, dim, k, privCopies);
	auto preN = LAYOUT::precomputeConstants(n, dim);
	auto preK = LAYOUT::precomputeConstants(k, dim);
	auto warps = blockDim.x * gridDim.x / warpSize;
	auto laneIdx = threadIdx.x % warpSize;
	for (std::uint32_t idx = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize; idx < n; idx += warps) {
		for (std::uint32_t d = laneIdx; d < dim; d += warpSize) {
			F x = LAYOUT::at(data, idx, d, preN);
			std::uint32_t target = indices[idx];
			F* res = &LAYOUT::at(result, target, d, preK);
			ATOMIC::add(res, x);
		}
	}
}


template<typename F, class LAYOUT, int EMUL>
void AtomicWarpDimKernel<F, LAYOUT, EMUL>::run(const ProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int threads = ((unsigned int)(in.n * 32) + exec.itemsPerThread - 1) / exec.itemsPerThread;
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	if (EMUL) {
		atomicWarpDimKernel<F, LAYOUT, AtomicEmulTrait<F>><<<exec.blockCount, exec.blockSize>>>(in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies);
	}
	else {
		atomicWarpDimKernel<F, LAYOUT, AtomicTrait<F>><<<exec.blockCount, exec.blockSize>>>(in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies);
	}
	runReducePrivCopiesKernel<F, LAYOUT>(in, exec);
}


// --------------------------------------------------------------------------------


/**
 * Simple kernel that launches one thread per source value (n x dim) and performs the additions atomically.
 */
template<typename F, class LAYOUT, class ATOMIC = AtomicTrait<F>>
__global__ void atomicKernel(const F* __restrict__ data, const std::uint32_t* __restrict__ indices, F* __restrict__ result,
	std::uint32_t dim, std::uint32_t k, std::uint32_t n, std::uint32_t privCopies)
{
	result = privatizeResultPointer<F, LAYOUT>(result, dim, k, privCopies);
	auto preN = LAYOUT::precomputeConstants(n, dim);
	auto preK = LAYOUT::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	auto size = n * dim;
	for (std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += threads) {
		std::uint32_t d = i % dim;
		std::uint32_t idx = i / dim;

		F x = LAYOUT::at(data, idx, d, preN);
		std::uint32_t target = indices[idx];
		F* res = &LAYOUT::at(result, target, d, preK);
		ATOMIC::add(res, x);
	}
}


template<typename F, class LAYOUT, int EMUL>
void AtomicKernel<F, LAYOUT, EMUL>::run(const ProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int threads = ((unsigned int)(in.n * in.dim) + exec.itemsPerThread - 1) / exec.itemsPerThread;
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	if (EMUL) {
		atomicKernel<F, LAYOUT, AtomicEmulTrait<F>><<<exec.blockCount, exec.blockSize>>>(in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies);
	}
	else {
		atomicKernel<F, LAYOUT, AtomicTrait<F>><<<exec.blockCount, exec.blockSize>>>(in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies);
	}
	runReducePrivCopiesKernel<F, LAYOUT>(in, exec);
}


// --------------------------------------------------------------------------------


/**
 * Kernel that launches one thread per source value (n x dim) and uses shared memory as a cache for the result.
 */
template<typename F, class LAYOUT, class ATOMIC = AtomicTrait<F>>
__global__ void atomicShmKernel(const F* __restrict__ data, const std::uint32_t* __restrict__ indices, F* __restrict__ result,
	std::uint32_t dim, std::uint32_t k, std::uint32_t n, std::uint32_t privCopies, std::uint32_t cachedVectors)
{
	extern __shared__ int sharedMemory[];
	F* resultCache = (F*)sharedMemory;
	auto cacheSize = cachedVectors * dim;
	for (std::uint32_t i = threadIdx.x; i < cacheSize; i += blockDim.x) {
		resultCache[i] = (F)0;
	}

	__syncthreads();

	result = privatizeResultPointer<F, LAYOUT>(result, dim, k, privCopies);
	auto preN = LAYOUT::precomputeConstants(n, dim);
	auto preK = LAYOUT::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	auto size = n * dim;
	for (std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += threads) {
		std::uint32_t d = i % dim;
		std::uint32_t idx = i / dim;

		F x = LAYOUT::at(data, idx, d, preN);
		std::uint32_t target = indices[idx];
		F* res = (target < cachedVectors)
			? resultCache + target * dim + d
			: &LAYOUT::at(result, target, d, preK);
		ATOMIC::add(res, x);
	}

	__syncthreads();

	// Write data from shm cache to result...
	for (std::uint32_t i = threadIdx.x; i < cacheSize; i += blockDim.x) {
		F* res = &LAYOUT::at(result, i / dim, i % dim, preK);
		ATOMIC::add(res, resultCache[i]);
	}
}


template<typename F, class LAYOUT, int EMUL>
void AtomicShmKernel<F, LAYOUT, EMUL>::run(const ProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int threads = ((unsigned int)(in.n * in.dim) + exec.itemsPerThread - 1) / exec.itemsPerThread;
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	std::uint32_t cachedVectors = std::min<std::uint32_t>(exec.sharedMemorySize / (in.dim*sizeof(F)), in.k);
	if (EMUL) {
		atomicShmKernel<F, LAYOUT, AtomicEmulTrait<F>><<<exec.blockCount, exec.blockSize, cachedVectors* in.dim * sizeof(F)>>>(
			in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies, cachedVectors);
	}
	else {
		atomicShmKernel<F, LAYOUT, AtomicTrait<F>><<<exec.blockCount, exec.blockSize, cachedVectors* in.dim * sizeof(F)>>>(
			in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies, cachedVectors);
	}
	runReducePrivCopiesKernel<F, LAYOUT>(in, exec);
}


// --------------------------------------------------------------------------------


/**
 * Kernel that launches one thread per source value (n x dim) and uses shared memory as a cache for the result.
 */
template<typename F, class LAYOUT, class ATOMIC = AtomicTrait<F>>
__global__ void atomicShm2Kernel(const F * __restrict__ data, const std::uint32_t * __restrict__ indices, F * __restrict__ result,
	std::uint32_t dim, std::uint32_t k, std::uint32_t n, std::uint32_t privCopies, std::uint32_t cachePrivCopies)
{
	extern __shared__ int sharedMemory[];
	F* resultCache = (F*)sharedMemory;
	auto cacheSize = k * dim * cachePrivCopies;
	for (std::uint32_t i = threadIdx.x; i < cacheSize; i += blockDim.x) {
		resultCache[i] = (F)0;
	}

	__syncthreads();

	result = privatizeResultPointer<F, LAYOUT>(result, dim, k, privCopies);
	auto preN = LAYOUT::precomputeConstants(n, dim);
	auto preK = LAYOUT::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	auto size = n * dim;
	auto cachePrivCopy = threadIdx.x % cachePrivCopies;
	for (std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += threads) {
		std::uint32_t d = i % dim;
		std::uint32_t idx = i / dim;

		F x = LAYOUT::at(data, idx, d, preN);
		std::uint32_t target = indices[idx];
		F* res = resultCache + (target * cachePrivCopies + cachePrivCopy) * dim + d;
		ATOMIC::add(res, x);
	}

	__syncthreads();

	// Write data from shm cache to result...
	cacheSize = k * dim;
	for (std::uint32_t i = threadIdx.x; i < cacheSize; i += blockDim.x) {
		F* res = &LAYOUT::at(result, i / dim, i % dim, preK);
		F sum = (F)0.0;
		for (std::uint32_t copy = 0; copy < cachePrivCopies; ++copy) {
			sum += resultCache[i + copy*dim];
		}
		ATOMIC::add(res, sum);
	}
}


template<typename F, class LAYOUT, int EMUL>
void AtomicShm2Kernel<F, LAYOUT, EMUL>::run(const ProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int threads = ((unsigned int)(in.n * in.dim) + exec.itemsPerThread - 1) / exec.itemsPerThread;
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	std::uint32_t cachePrivCopies = exec.sharedMemorySize / (in.dim * in.k * sizeof(F));
	if (cachePrivCopies < 1) {
		throw std::runtime_error("Insufficient shared memory size for given parameters.");
	}
	exec.sharedMemorySize = cachePrivCopies * in.dim * in.k * sizeof(F);
	if (EMUL) {
		atomicShm2Kernel<F, LAYOUT, AtomicEmulTrait<F>><<<exec.blockCount, exec.blockSize, exec.sharedMemorySize>>>(
			in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies, cachePrivCopies);
	}
	else {
		atomicShm2Kernel<F, LAYOUT, AtomicTrait<F>><<<exec.blockCount, exec.blockSize, exec.sharedMemorySize>>>(
			in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies, cachePrivCopies);
	}
	runReducePrivCopiesKernel<F, LAYOUT>(in, exec);
}


// --------------------------------------------------------------------------------


/**
 * Kernel that launches one thread per source value (n x dim) but ignores the inputs and write fake data atomically to buckets.
 * This kernel should measure the raw overhead of the atomic writes.
 */
template<typename F, class LAYOUT, class ATOMIC = AtomicTrait<F>>
__global__ void atomicFakeKernel(const F * __restrict__ data, const std::uint32_t * __restrict__ indices, F * __restrict__ result,
	std::uint32_t dim, std::uint32_t k, std::uint32_t n, std::uint32_t privCopies)
{
	result = privatizeResultPointer<F, LAYOUT>(result, dim, k, privCopies);
	auto preK = LAYOUT::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	auto size = n * dim;
	F x = 0.1;
	for (std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += threads) {
		std::uint32_t d = i % dim;
		std::uint32_t target = (i / dim) % k;
		F* res = &LAYOUT::at(result, target, d, preK);
		ATOMIC::add(res, x);
	}
}


template<typename F, class LAYOUT, int EMUL>
void AtomicFakeKernel<F, LAYOUT, EMUL>::run(const ProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int threads = ((unsigned int)(in.n * in.dim) + exec.itemsPerThread - 1) / exec.itemsPerThread;
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	if (EMUL) {
		atomicFakeKernel<F, LAYOUT, AtomicEmulTrait<F>><<<exec.blockCount, exec.blockSize>>>(
			in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies);
	}
	else {
		atomicFakeKernel<F, LAYOUT, AtomicTrait<F>><<<exec.blockCount, exec.blockSize>>>(
			in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies);
	}
	runReducePrivCopiesKernel<F, LAYOUT>(in, exec);
}


// --------------------------------------------------------------------------------


/**
 * Kernel that launches one thread per source value (n x dim) and uses shared memory as a cache for the result.
 */
template<typename F, class LAYOUT, class ATOMIC = AtomicTrait<F>>
__global__ void atomicShmFakeKernel(const F * __restrict__ data, const std::uint32_t * __restrict__ indices, F * __restrict__ result,
	std::uint32_t dim, std::uint32_t k, std::uint32_t n, std::uint32_t privCopies, std::uint32_t cachedVectors)
{
	extern __shared__ int sharedMemory[];
	F* resultCache = (F*)sharedMemory;
	auto cacheSize = cachedVectors * dim;
	for (std::uint32_t i = threadIdx.x; i < cacheSize; i += blockDim.x) {
		resultCache[i] = (F)0;
	}

	__syncthreads();

	result = privatizeResultPointer<F, LAYOUT>(result, dim, k, privCopies);
	auto preN = LAYOUT::precomputeConstants(n, dim);
	auto preK = LAYOUT::precomputeConstants(k, dim);
	auto threads = blockDim.x * gridDim.x;
	auto size = n * dim;
	for (std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += threads) {
		std::uint32_t d = i % dim;

		F x = 0.1;
		std::uint32_t target = (i / dim) % k;
		F* res = (target < cachedVectors)
			? resultCache + target * dim + d
			: &LAYOUT::at(result, target, d, preK);
		ATOMIC::add(res, x);
	}

	__syncthreads();

	// Write data from shm cache to result...
	for (std::uint32_t i = threadIdx.x; i < cacheSize; i += blockDim.x) {
		F* res = &LAYOUT::at(result, i / dim, i % dim, preK);
		ATOMIC::add(res, resultCache[i]);
	}
}


template<typename F, class LAYOUT, int EMUL>
void AtomicShmFakeKernel<F, LAYOUT, EMUL>::run(const ProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int threads = ((unsigned int)(in.n * in.dim) + exec.itemsPerThread - 1) / exec.itemsPerThread;
	exec.blockCount = (threads + exec.blockSize - 1) / exec.blockSize;
	std::uint32_t cachedVectors = std::min<std::uint32_t>(exec.sharedMemorySize / (in.dim * sizeof(F)), in.k);
	if (EMUL) {
		atomicShmFakeKernel<F, LAYOUT, AtomicEmulTrait<F>><<<exec.blockCount, exec.blockSize, cachedVectors* in.dim * sizeof(F)>>>(
			in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies, cachedVectors);
	}
	else {
		atomicShmFakeKernel<F, LAYOUT, AtomicTrait<F>><<<exec.blockCount, exec.blockSize, cachedVectors* in.dim * sizeof(F)>>>(
			in.data, in.indices, in.result, in.dim, in.k, in.n, exec.privatizedCopies, cachedVectors);
	}
	runReducePrivCopiesKernel<F, LAYOUT>(in, exec);
}


// --------------------------------------------------------------------------------


/*
 * All kernels should be registered here...
 */

template<typename F, class LAYOUT, int EMUL>
void instantiateAtomicKernelRunnerTemplates(ProblemInstance<F> &instance, CudaExecParameters &exec)
{
	AtomicDimKernel<F, LAYOUT, EMUL>::run(instance, exec);
	AtomicWarpDimKernel<F, LAYOUT, EMUL>::run(instance, exec);
	AtomicKernel<F, LAYOUT, EMUL>::run(instance, exec);
	AtomicShmKernel<F, LAYOUT, EMUL>::run(instance, exec);
	AtomicShm2Kernel<F, LAYOUT, EMUL>::run(instance, exec);
	AtomicFakeKernel<F, LAYOUT, EMUL>::run(instance, exec);
	AtomicShmFakeKernel<F, LAYOUT, EMUL>::run(instance, exec);
}

template<typename F, class LAYOUT>
void instantiateKernelRunnerTemplates()
{
	// This is fake routine (just to enforce explicit template instantiations).
	// No code will be actually executed.
	ProblemInstance<F> instance(nullptr, nullptr, nullptr, 0, 0, 0);
	CudaExecParameters exec;
	BaseKernel<F, LAYOUT>::run(instance, exec);

	instantiateAtomicKernelRunnerTemplates<F, LAYOUT, true>(instance, exec);
	instantiateAtomicKernelRunnerTemplates<F, LAYOUT, false>(instance, exec);
}


// Machinery that instantiates all templated versions of kernel runners...
template<typename F, int ALIGN>
void instantiateKernelRunnerTemplatesHelper1()
{
	instantiateKernelRunnerTemplates<F, SoALayoutTrait<ALIGN>>();
	instantiateKernelRunnerTemplates<F, AoSLayoutTrait<ALIGN>>();
}

template<typename F>
void instantiateKernelRunnerTemplatesHelper2()
{
	instantiateKernelRunnerTemplatesHelper1<F, 1>();
	instantiateKernelRunnerTemplatesHelper1<F, 32>();
}

template void instantiateKernelRunnerTemplatesHelper2<float>();
#ifndef NO_DOUBLES
template void instantiateKernelRunnerTemplatesHelper2<double>();
#endif
