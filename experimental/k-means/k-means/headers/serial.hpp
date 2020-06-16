#ifndef K_MEANS_SERIAL_HPP
#define K_MEANS_SERIAL_HPP

#include "metric_policies.hpp"
#include "layout_policies.hpp"
#include "interface.hpp"

#include <vector>
#include <cstdint>


template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = AoSLayoutPolicy<>,
	class LAYOUT_MEANS = LAYOUT, class METRIC = EuclideanMetricPolicy<F, LAYOUT, LAYOUT_MEANS>>
class SerialKMeansAlgorithm : public IKMeansAlgorithm<F, IDX_T>
{
private:
	const F* mData;
	std::vector<F> mPrevMeans;
	std::vector<IDX_T> mClusterSizes;

	IDX_T findNearestMean(IDX_T idx) const
	{
		auto precomputedData = LAYOUT::precomputeConstants(this->mN, this->mDim);
		auto precomputedMeans = LAYOUT_MEANS::precomputeConstants(this->mK, this->mDim);
		IDX_T nearest = 0;
		F minDist = METRIC::distance(mData, idx, precomputedData, mPrevMeans.data(), 0, precomputedMeans, this->mDim);
		for (IDX_T i = 1; i < this->mK; ++i) {
			F dist = METRIC::distance(mData, idx, precomputedData, mPrevMeans.data(), i, precomputedMeans, this->mDim);
			if (minDist > dist) {
				minDist = dist;
				nearest = i;
			}
		}
		return nearest;
	}

	void computeAssignment()
	{
		for (IDX_T i = 0; i < this->mN; ++i) {
			this->mAssignment[i] = findNearestMean(i);
		}
	}

	void updateMeans()
	{
		// Clear new means...
		auto precomputedData = LAYOUT::precomputeConstants(this->mN, this->mDim);
		auto precomputedMeans = LAYOUT_MEANS::precomputeConstants(this->mK, this->mDim);
		for (IDX_T i = 0; i < this->mK; ++i) {
			mClusterSizes[i] = 0;
			for (IDX_T d = 0; d < this->mDim; ++d) {
				LAYOUT_MEANS::at(this->mMeans.data(), i, d, precomputedMeans) = (F)0;
			}
		}

		// Accumulate...
		for (IDX_T i = 0; i < this->mN; ++i) {
			IDX_T target = this->mAssignment[i];
			++mClusterSizes[target];
			for (IDX_T d = 0; d < this->mDim; ++d) {
				LAYOUT_MEANS::at(this->mMeans.data(), target, d, precomputedMeans) += LAYOUT::at(mData, i, d, precomputedData);
			}
		}

		// Divide...
		for (IDX_T i = 0; i < this->mK; ++i) {
			if (mClusterSizes[i] == 0) continue; // if means were original points, this should not happen
			F divisor = (F)mClusterSizes[i];
			for (IDX_T d = 0; d < this->mDim; ++d) {
				LAYOUT_MEANS::at(this->mMeans.data(), i, d, precomputedMeans) /= divisor;
			}
		}
	}

public:
	SerialKMeansAlgorithm() : mData(nullptr) {}

	virtual void initialize(std::size_t dim, std::size_t k, std::size_t n, std::size_t iters,
		const std::vector<F>& data, const std::vector<F>& means) override
	{
		IKMeansAlgorithm<F, IDX_T>::initialize(dim, k, n, iters, data, means);
		mData = data.data();
		mPrevMeans.resize(LAYOUT_MEANS::size(this->mK, this->mDim));
		this->mMeans = means;
		mClusterSizes.resize(this->mK);
		this->mAssignment.resize(this->mN);
	}

	virtual void run() override
	{
		for (std::size_t iter = 0; iter < this->mIters; ++iter) {
			this->mMeans.swap(mPrevMeans);
			computeAssignment();
			updateMeans();
		}
	}

	virtual void cleanup() override
	{
		mPrevMeans.clear();
		mClusterSizes.clear();
	}
};


template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = AoSLayoutPolicy<>,
	class LAYOUT_MEANS = LAYOUT, class METRIC = EuclideanMetricPolicy<F, LAYOUT, LAYOUT_MEANS>>
class SerialAssignmentAlgorithm : public IAssignmentAlgorithm<F, IDX_T>
{
private:
	const F* mData;
	const F* mMeans;

public:
	SerialAssignmentAlgorithm() : mData(nullptr), mMeans(nullptr) {}

	virtual void initialize(std::size_t dim, std::size_t k, std::size_t n,
		const std::vector<F>& data, const std::vector<F>& means) override
	{
		IAssignmentAlgorithm<F>::initialize(dim, k, n, data, means);

		mData = &data[0];
		mMeans = &means[0];
	}

	virtual void run() override
	{
		auto precomputedLayoutData = LAYOUT::precomputeConstants(this->mN, this->mDim);
		auto precomputedLayoutMeans = LAYOUT_MEANS::precomputeConstants(this->mK, this->mDim);
		for (IDX_T i = 0; i < this->mN; ++i) {
			IDX_T closestIdx = 0;
			F closestDist = METRIC::distance(mData, i, precomputedLayoutData, mMeans, 0, precomputedLayoutMeans, this->mDim);
			for (IDX_T j = 1; j < this->mK; ++j) {
				F dist = METRIC::distance(mData, i, precomputedLayoutData, mMeans, j, precomputedLayoutMeans, this->mDim);
				if (dist < closestDist) {
					closestDist = dist;
					closestIdx = j;
				}
			}
			this->mAssignment[i] = closestIdx;
		}
	}


	virtual const std::vector<IDX_T>& getAssignment() override
	{
		return this->mAssignment;
	}
};


#endif
