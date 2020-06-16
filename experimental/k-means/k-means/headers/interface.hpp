#ifndef K_MEANS_INTERFACE_HPP
#define K_MEANS_INTERFACE_HPP

#include <vector>
#include <cstdint>


template<typename F = float, typename IDX_T = std::uint32_t>
class IKMeansAlgorithm
{
protected:
	std::size_t mDim, mK, mN, mIters;
	std::vector<F> mMeans;
	std::vector<IDX_T> mAssignment;

public:
	IKMeansAlgorithm() : mDim(0), mK(0), mN(0), mIters(1) {}

	virtual ~IKMeansAlgorithm() {}

	std::size_t getIterations() const { return mIters; }

	virtual void initialize(std::size_t dim, std::size_t k, std::size_t n, std::size_t iters,
		const std::vector<F>& data, const std::vector<F>& means)
	{
		mDim = dim;
		mK = k;
		mN = n;
		mIters = iters;
	}
	
	virtual void run() = 0;

	virtual const std::vector<F>& getMeans()
	{
		return mMeans;
	}

	virtual const std::vector<IDX_T>& getAssignment()
	{
		return mAssignment;
	}

	virtual void cleanup()
	{
		mMeans.clear();
		mAssignment.clear();
	}
};


template<typename F = float, typename IDX_T = std::uint32_t>
class IAssignmentAlgorithm
{
protected:
	std::size_t mDim, mK, mN;
	std::vector<IDX_T> mAssignment;

public:
	IAssignmentAlgorithm() : mDim(0), mK(0), mN(0) {}

	virtual ~IAssignmentAlgorithm() {}

	virtual void initialize(std::size_t dim, std::size_t k, std::size_t n,
		const std::vector<F>& data, const std::vector<F>& means)
	{
		mDim = dim;
		mK = k;
		mN = n;
		mAssignment.resize(mN);
	}

	virtual void run() = 0;

	virtual const std::vector<IDX_T>& getAssignment()
	{
		return mAssignment;
	}

	virtual void cleanup()
	{
		mAssignment.clear();
	}
};


#endif
