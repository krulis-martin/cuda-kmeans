#ifndef BUCKETING_CUDA_CUDA_HPP
#define BUCKETING_CUDA_CUDA_HPP

#include "kernels.cuh"
#include "layout_traits.hpp"
#include "interface.hpp"

#include "cuda/cuda.hpp"

#include <vector>
#include <cstdint>


template<typename F = float, class LAYOUT = AoSLayoutTrait<>, class KERNEL = BaseKernel<F, LAYOUT>>
class CudaBucketingAlgorithm : public IBucketingAlgorithm<F>
{
private:
	CudaExecParameters mExecParams;

	std::vector<bpp::CudaBuffer<F>> mCuData;
	std::vector<bpp::CudaBuffer<std::uint32_t>> mCuIndices;
	std::vector<bpp::CudaBuffer<F>> mCuResult;

public:
	CudaBucketingAlgorithm(const CudaExecParameters& exec) : mExecParams(exec) {}

	virtual void initialize(std::size_t dim, std::size_t k, std::size_t n, std::size_t iter,
		const std::vector<F>& data, const std::vector<std::uint32_t>& indices) override
	{
		IBucketingAlgorithm<F>::initialize(dim, k, n, iter, data, indices);
		this->mResult.clear();

		std::size_t devices = bpp::CudaDevice::count();
		if (devices == 0) throw bpp::RuntimeError("No CUDA devices found!");

		CUCH(cudaSetDevice(0));
		mCuData.resize(this->mIter);
		mCuIndices.resize(this->mIter);
		mCuResult.resize(this->mIter);

		auto dataSize = LAYOUT::size(this->mN, this->mDim);
		auto resSize = LAYOUT::size(this->mK, this->mDim);
		for (std::size_t i = 0; i < this->mIter; ++i) {
			mCuData[i].realloc(dataSize);
			mCuIndices[i].realloc(this->mN);
			mCuResult[i].realloc(resSize * this->mExecParams.privatizedCopies);

			mCuData[i].write(&data[i*dataSize], dataSize);
			mCuIndices[i].write(&indices[i* this->mN], this->mN);
			runFillZeros<F>(*mCuResult[i], resSize * this->mExecParams.privatizedCopies);
		}

		CUCH(cudaDeviceSynchronize());
	}

	virtual void run() override
	{
		for (std::size_t iter = 0; iter < this->mIter; ++iter) {
			ProblemInstance<F> instance(*mCuData[iter], *mCuIndices[iter], *mCuResult[iter], this->mDim, this->mK, this->mN);
			KERNEL::run(instance, mExecParams);
		}
		CUCH(cudaDeviceSynchronize());
	}


	virtual const std::vector<F>& getResult() override
	{
		if (this->mResult.empty()) {
			auto iterResSize = LAYOUT::size(this->mK, this->mDim);
			auto precomputedRes = LAYOUT::precomputeConstants(this->mK, this->mDim);
			this->mResult.reserve(iterResSize * this->mIter);

			for (std::size_t iter = 0; iter < this->mIter; ++iter) {
				std::vector<F> tmpRes(iterResSize);
				mCuResult[iter].read(tmpRes);
				for (std::size_t i = 0; i < this->mK; ++i) {
					for (std::size_t d = 0; d < this->mDim; ++d)
						this->mResult.push_back(LAYOUT::at(&tmpRes[0], i, d, precomputedRes));
				}
			}
		}

		return this->mResult;
	}
};


#endif
