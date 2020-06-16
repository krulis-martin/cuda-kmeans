#ifndef BUCKETING_CUDA_INTERFACE_HPP
#define BUCKETING_CUDA_INTERFACE_HPP

#include <vector>
#include <cstdint>

template<typename F = float>
class IBucketingAlgorithm
{
protected:
	std::size_t mDim, mK, mN, mIter;
	std::vector<F> mResult;

public:
	IBucketingAlgorithm() : mDim(0), mK(0), mN(0), mIter(1) {}

	virtual ~IBucketingAlgorithm() {}

	virtual void initialize(std::size_t dim, std::size_t k, std::size_t n, std::size_t iter,
		const std::vector<F>& data, const std::vector<std::uint32_t>& indices) 
	{
		mDim = dim;
		mK = k;
		mN = n;
		mIter = iter;
	}
	
	virtual void run() = 0;

	virtual const std::vector<F>& getResult()
	{
		return mResult;
	}

	virtual void cleanup()
	{
		mResult.clear();
	}
};


#endif
