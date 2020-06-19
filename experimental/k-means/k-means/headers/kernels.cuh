#ifndef K_MEANS_KEKRNELS_CUH
#define K_MEANS_KEKRNELS_CUH

#include <cstddef>
#include <cstdint>

template<typename F, typename IDX_T>
struct AssignmentProblemInstance {
public:
	const F* data;
	const F* means;
	const IDX_T* lastAssignment;
	IDX_T* assignment; // the result
	IDX_T dim;
	IDX_T k;
	IDX_T n;

	AssignmentProblemInstance(const F* _data, const F* _means, const IDX_T* _lastAssignment, IDX_T* _assignment, std::size_t _dim, std::size_t _k, std::size_t _n)
		: data(_data), means(_means), lastAssignment(_lastAssignment), assignment(_assignment), dim((IDX_T)_dim), k((IDX_T)_k), n((IDX_T)_n) {}
};


template<typename F, typename IDX_T>
struct UpdateProblemInstance {
public:
	const F* data;
	const IDX_T* assignment;
	F* means; // the result
	IDX_T *clusterSizes;
	IDX_T dim;
	IDX_T k;
	IDX_T n;

	UpdateProblemInstance(const F* _data, const IDX_T* _assignment, F* _means, IDX_T *_clusterSizes, std::size_t _dim, std::size_t _k, std::size_t _n)
		: data(_data), assignment(_assignment), means(_means), clusterSizes(_clusterSizes), dim((IDX_T)_dim), k((IDX_T)_k), n((IDX_T)_n) {}
};


template<typename F, typename IDX_T>
struct FusedProblemInstance {
public:
	const F* data;
	const F* meansIn;
	F* meansOut; // result
	IDX_T* clusterSizes; // tmp
	IDX_T dim;
	IDX_T k;
	IDX_T n;

	FusedProblemInstance(const F* _data, const F* _meansIn, F* _meansOut, IDX_T* _clusterSizes, std::size_t _dim, std::size_t _k, std::size_t _n)
		: data(_data), meansIn(_meansIn), meansOut(_meansOut), clusterSizes(_clusterSizes), dim((IDX_T)_dim), k((IDX_T)_k), n((IDX_T)_n) {}
};


struct CudaExecParameters {
public:
	unsigned int blockSize;
	unsigned int blockCount;
	unsigned int sharedMemorySize;
	std::uint32_t privatizedCopies;
	std::uint32_t itemsPerThread;
	std::uint32_t regsCache;


	CudaExecParameters(unsigned int _blockSize = 256, unsigned int _blockCount = 0, unsigned int _sharedMemorySize = 0,
		std::uint32_t _privatizedCopies = 1, std::uint32_t _itemsPerThread = 1, std::uint32_t _regsCache = 1)
		: blockSize(_blockSize), blockCount(_blockCount), sharedMemorySize(_sharedMemorySize),
		privatizedCopies(_privatizedCopies), itemsPerThread(_itemsPerThread), regsCache(_regsCache)
	{}
};



template<typename F>
void runFillZeros(F* data, std::size_t n);

template<typename F, typename IDX_T, class LAYOUT_MEANS>
void runDivideMeansKernel(const UpdateProblemInstance<F, IDX_T>& in, CudaExecParameters& exec);


#define DECLARE_ASSIGNEMNT_KERNEL(NAME)\
template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>\
class NAME {\
public:\
	static void run(const AssignmentProblemInstance<F, IDX_T>& in, CudaExecParameters& exec);\
};

DECLARE_ASSIGNEMNT_KERNEL(BaseAssignmentKernel)
DECLARE_ASSIGNEMNT_KERNEL(CachedFixedAssignmentKernel)
DECLARE_ASSIGNEMNT_KERNEL(CachedAllMeansAssignmentKernel)
DECLARE_ASSIGNEMNT_KERNEL(Cached2AssignmentKernel)
DECLARE_ASSIGNEMNT_KERNEL(CachedRegsAssignmentKernel)
DECLARE_ASSIGNEMNT_KERNEL(BestCachedAssignmentKernel)


#define DECLARE_UPDATE_KERNEL(NAME)\
template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, int EMUL = false>\
class NAME {\
public:\
	static void run(const UpdateProblemInstance<F, IDX_T>& in, CudaExecParameters& exec);\
};

DECLARE_UPDATE_KERNEL(UpdateAtomicPointKernel)
DECLARE_UPDATE_KERNEL(UpdateAtomicFineKernel)
DECLARE_UPDATE_KERNEL(UpdateAtomicShmKernel)


#define DECLARE_FUSED_KERNEL(NAME)\
template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>\
class NAME {\
public:\
	static void run(const FusedProblemInstance<F, IDX_T>& in, CudaExecParameters& exec);\
};

DECLARE_FUSED_KERNEL(FusedCachedFixedKernel)
DECLARE_FUSED_KERNEL(FusedCachedRegsKernel)

#endif
