#include "kmcuda_adapter.hpp"
#include "../kmcuda/kmcuda.h"

#include <vector>
#include <stdexcept>

KMCudaAlgorithm::KMCudaAlgorithm() : mDim(0), mK(0), mN(0), mIters(1), mResultReady(false) {}

std::size_t KMCudaAlgorithm::getIterations() const
{
	return mIters;
}

void KMCudaAlgorithm::initialize(std::size_t dim, std::size_t k, std::size_t n, std::size_t iters,
	const std::vector<float>& data, const std::vector<float>& means)
{
	this->mDim = dim;
	this->mK = k;
	this->mN = n;
	this->mIters = iters;
	this->mResultReady = false;

	this->mMeans.resize(k * dim);
	this->mAssignment.resize(n);

	this->mCuData.realloc(data.size());
	this->mCuMeans.realloc(means.size());
	this->mCuAssignment.realloc(n);

	this->mCuData.write(data);
	this->mCuMeans.write(means);
	CUCH(cudaDeviceSynchronize());
}

void KMCudaAlgorithm::run()
{
	int iters = (int)this->mIters;
	KMCUDAResult result = kmeans_cuda(
		kmcudaInitMethodImport, nullptr,
		//kmcudaInitMethodRandom, nullptr,
		0.001f,                           // less than 0.1% of the samples are reassigned in the end
		0.1f,                             // activate Yinyang refinement with 0.1 threshold
		kmcudaDistanceMetricL2,          // Euclidean distance
		(std::uint32_t)this->mN, (std::uint16_t)this->mDim, (std::uint32_t)this->mK,
		0xDEADBEEF,                      // random generator seed
		1,                               // use first CUDA device only
		0,                               // samples are already on GPU #0
		0,                               // not in float16x2 mode
		0,                               // verbosity
		*this->mCuData, *this->mCuMeans, *this->mCuAssignment, nullptr,
		iters);
	if (result != kmcudaSuccess) {
		throw std::runtime_error("KMCUDA algorithm failed.");
	}
	this->mIters = (std::size_t)iters;
}

void KMCudaAlgorithm::fetchResult()
{
	if (this->mResultReady) return;

	this->mCuMeans.read(this->mMeans);
	this->mCuAssignment.read(this->mAssignment);
	CUCH(cudaDeviceSynchronize());
	this->mResultReady = true;
}

const std::vector<float>& KMCudaAlgorithm::getMeans()
{
	this->fetchResult();
	return this->mMeans;
}

const std::vector<std::uint32_t>& KMCudaAlgorithm::getAssignment()
{
	this->fetchResult();
	return this->mAssignment;
}
