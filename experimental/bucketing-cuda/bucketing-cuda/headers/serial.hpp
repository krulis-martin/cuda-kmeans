#ifndef BUCKETING_CUDA_SERIAL_HPP
#define BUCKETING_CUDA_SERIAL_HPP

#include "layout_traits.hpp"
#include "interface.hpp"

#include <vector>
#include <cstdint>


template<typename F = float, class LAYOUT = AoSLayoutTrait<>>
class SerialBucketingAlgorithm : public IBucketingAlgorithm<F>
{
private:
	const F* mData;
	const std::uint32_t* mIndices;
	bool mResultReady;

public:
	SerialBucketingAlgorithm() : mData(nullptr), mIndices(nullptr), mResultReady(false) {}

	virtual void initialize(std::size_t dim, std::size_t k, std::size_t n, std::size_t iter,
		const std::vector<F>& data, const std::vector<std::uint32_t>& indices) override
	{
		IBucketingAlgorithm<F>::initialize(dim, k, n, iter, data, indices);

		mData = &data[0];
		mIndices = &indices[0];

		this->mResult.resize(LAYOUT::size(this->mK, this->mDim) * this->mIter);
		for (auto&& x : this->mResult) x = (F)0.0;

		mResultReady = false;
	}

	virtual void run() override
	{
		auto precomputedData = LAYOUT::precomputeConstants(this->mN, this->mDim);
		auto precomputedRes = LAYOUT::precomputeConstants(this->mK, this->mDim);
		auto iterDataSize = LAYOUT::size(this->mN, this->mDim);
		auto iterResSize = LAYOUT::size(this->mK, this->mDim);
		const F* data = mData;
		const std::uint32_t* indices = mIndices;
		F* res = &this->mResult[0];
		for (std::size_t iter = 0; iter < this->mIter; ++iter) {
			for (std::size_t i = 0; i < this->mN; ++i) {
				auto target = indices[i];
				for (std::size_t d = 0; d < this->mDim; ++d) {
					LAYOUT::at(res, target, d, precomputedRes) += LAYOUT::at(data, i, d, precomputedData);
				}
			}
			data += iterDataSize;
			indices += this->mN;
			res += iterResSize;
		}
	}


	virtual const std::vector<F>& getResult() override
	{
		if (!mResultReady) {
			// We need to convert the result to AoS layout without any padding...
			std::vector<F> finalResult;
			finalResult.reserve(this->mK * this->mDim);
			auto precomputedRes = LAYOUT::precomputeConstants(this->mK, this->mDim);
			auto iterResSize = LAYOUT::size(this->mK, this->mDim);
			const F* res = &this->mResult[0];
			for (std::size_t iter = 0; iter < this->mIter; ++iter) {
				for (std::size_t i = 0; i < this->mK; ++i) {
					for (std::size_t d = 0; d < this->mDim; ++d)
						finalResult.push_back(LAYOUT::at(res, i, d, precomputedRes));
				}
				res += iterResSize;
			}
			finalResult.swap(this->mResult);
			mResultReady = true;
		}

		return this->mResult;
	}
};


#endif
