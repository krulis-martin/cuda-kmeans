#ifndef K_MEANS_KMCUDA_ADAPTER_HPP
#define K_MEANS_KMCUDA_ADAPTER_HPP

#include "interface.hpp"
#include "layout_policies.hpp"
#include "metric_policies.hpp"
#include "cuda/cuda.hpp"

class KMCudaAlgorithm
{
private:
	std::size_t mDim, mK, mN, mIters;
	bool mResultReady;

	std::vector<float> mMeans;
	std::vector<std::uint32_t> mAssignment;

	bpp::CudaBuffer<float> mCuData;
	bpp::CudaBuffer<float> mCuMeans;
	bpp::CudaBuffer<std::uint32_t> mCuAssignment;

	void fetchResult();

public:
	KMCudaAlgorithm();
	std::size_t getIterations() const;
	void initialize(std::size_t dim, std::size_t k, std::size_t n, std::size_t iters,
		const std::vector<float>& data, const std::vector<float>& means);
	void run();
	const std::vector<float>& getMeans();
	const std::vector<std::uint32_t>& getAssignment();
};


template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS, class METRIC>
class KMCudaAdapter : public IKMeansAlgorithm<F, IDX_T>
{
public:
	virtual void run() override
	{
		throw std::runtime_error("Algorithm kmcuda is implemented only for floats in AoS layout and L2 metric.");
	}
};


template<>
class KMCudaAdapter<float, std::uint32_t, AoSLayoutPolicy<1>, AoSLayoutPolicy<1>, EuclideanMetricPolicy<float, AoSLayoutPolicy<1>>> : public IKMeansAlgorithm<float, std::uint32_t>
{
private:
	KMCudaAlgorithm mAlgorithm;

public:
	virtual void initialize(std::size_t dim, std::size_t k, std::size_t n, std::size_t iters,
		const std::vector<float>& data, const std::vector<float>& means) override
	{
		mAlgorithm.initialize(dim, k, n, iters, data, means);
	}

	virtual void run() override
	{
		mAlgorithm.run();
		this->mIters = mAlgorithm.getIterations();
	}

	virtual const std::vector<float>& getMeans() override
	{
		return mAlgorithm.getMeans();
	}

	virtual const std::vector<std::uint32_t>& getAssignment() override
	{
		return mAlgorithm.getAssignment();
	}
	
	virtual void cleanup() override {}
};


#endif
