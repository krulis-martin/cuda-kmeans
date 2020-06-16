#ifndef BUCKETING_CUDA_KEKRNELS_CUH
#define BUCKETING_CUDA_KEKRNELS_CUH

#include <cstddef>
#include <cstdint>

template<typename F>
struct ProblemInstance {
public:
	const F* data;
	const std::uint32_t* indices;
	F* result;
	std::uint32_t dim;
	std::uint32_t k;
	std::uint32_t n;

	ProblemInstance(const F* _data, const std::uint32_t* _indices, F* _result, std::size_t _dim, std::size_t _k, std::size_t _n)
		: data(_data), indices(_indices), result(_result), dim((std::uint32_t)_dim), k((std::uint32_t)_k), n((std::uint32_t)_n) {}
};


struct CudaExecParameters {
public:
	unsigned int blockSize;
	unsigned int blockCount;
	unsigned int sharedMemorySize;
	std::uint32_t privatizedCopies;
	std::uint32_t itemsPerThread;


	CudaExecParameters(unsigned int _blockSize = 256, unsigned int _blockCount = 0, unsigned int _sharedMemorySize = 0,
		std::uint32_t _privatizedCopies = 1, std::uint32_t _itemsPerThread = 1)
		: blockSize(_blockSize), blockCount(_blockCount), sharedMemorySize(_sharedMemorySize),
		privatizedCopies(_privatizedCopies), itemsPerThread(_itemsPerThread){}
};



template<typename F>
void runFillZeros(F* data, std::size_t n);


template<typename F, class LAYOUT>
class BaseKernel {
public:
	static void run(const ProblemInstance<F>& in, CudaExecParameters& exec);
};


#define DECLARE_KERNEL(NAME)\
template<typename F, class LAYOUT, int EMUL>\
class NAME {\
public:\
	static void run(const ProblemInstance<F>& in, CudaExecParameters& exec);\
};

DECLARE_KERNEL(AtomicDimKernel)
DECLARE_KERNEL(AtomicWarpDimKernel)
DECLARE_KERNEL(AtomicKernel)
DECLARE_KERNEL(AtomicShmKernel)
DECLARE_KERNEL(AtomicShm2Kernel)
DECLARE_KERNEL(AtomicFakeKernel)
DECLARE_KERNEL(AtomicShmFakeKernel)

#endif
