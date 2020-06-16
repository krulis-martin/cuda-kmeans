#ifndef K_MEANS_CUDA_HPP
#define K_MEANS_CUDA_HPP

#include "kernels.cuh"
#include "metric_policies.hpp"
#include "layout_policies.hpp"
#include "interface.hpp"

#include "cuda/cuda.hpp"

#include <vector>
#include <cstdint>


template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = AoSLayoutPolicy<>,
	class LAYOUT_MEANS = LAYOUT, class METRIC = EuclideanMetricPolicy<F, LAYOUT, LAYOUT_MEANS>,
	class ASGN_KERNEL = BaseAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>,
	class UPDATE_KERNEL = UpdateAtomicShmKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, false>>
class CudaKMeansAlgorithm : public IKMeansAlgorithm<F, IDX_T>
{
private:
	CudaExecParameters mExecParams;
	bool mResultReady;

	bpp::CudaBuffer<F> mCuData;
	bpp::CudaBuffer<F> mCuMeans[2];
	bpp::CudaBuffer<IDX_T> mCuAssignment[2];
	bpp::CudaBuffer<IDX_T> mCuClusterSizes;
	IDX_T mLast;

	void fetchResult()
	{
		if (mResultReady) return;

		mCuMeans[mLast].read(this->mMeans);
		mCuAssignment[mLast].read(this->mAssignment);
		CUCH(cudaDeviceSynchronize());
		mResultReady = true;
	}

public:
	CudaKMeansAlgorithm(const CudaExecParameters& exec) : mExecParams(exec), mResultReady(false), mLast(0) {}

	virtual void initialize(std::size_t dim, std::size_t k, std::size_t n, std::size_t iters,
		const std::vector<F>& data, const std::vector<F>& means) override
	{
		IKMeansAlgorithm<F, IDX_T>::initialize(dim, k, n, iters, data, means);
		mResultReady = false;

		std::size_t devices = bpp::CudaDevice::count();
		if (devices == 0) throw bpp::RuntimeError("No CUDA devices found!");

		CUCH(cudaSetDevice(0));
		mCuData.realloc(LAYOUT::size(this->mN, this->mDim));
		mCuClusterSizes.realloc(this->mK);
		for (std::size_t i = 0; i < 2; ++i) {
			mCuMeans[i].realloc(LAYOUT_MEANS::size(this->mK, this->mDim));
			mCuAssignment[i].realloc(this->mN);
		}

		mCuData.write(data);

		mLast = 0;
		mCuMeans[mLast].write(means);
		runFillZeros<IDX_T>(*mCuAssignment[mLast], this->mN);
		runFillZeros<IDX_T>(*mCuClusterSizes, this->mK);
		CUCH(cudaDeviceSynchronize());
	}

	virtual void run() override
	{
		for (std::size_t iter = 0; iter < this->mIters; ++iter) {
			auto next = (mLast + 1) % 2;
			AssignmentProblemInstance<F, IDX_T> assignment(*mCuData, *mCuMeans[mLast], *mCuAssignment[mLast], *mCuAssignment[next], this->mDim, this->mK, this->mN);
			ASGN_KERNEL::run(assignment, mExecParams);
			CUCH(cudaGetLastError());

			runFillZeros<F>(*mCuMeans[next], LAYOUT_MEANS::size(this->mK, this->mDim));
			runFillZeros<IDX_T>(*mCuClusterSizes, this->mK);

			UpdateProblemInstance<F, IDX_T> update(*mCuData, *mCuAssignment[next], *mCuMeans[next], *mCuClusterSizes, this->mDim, this->mK, this->mN);
			UPDATE_KERNEL::run(update, mExecParams);
			CUCH(cudaGetLastError());

			mLast = next;
		}
		CUCH(cudaDeviceSynchronize());
	}

	virtual const std::vector<F>& getMeans() override
	{
		this->fetchResult();
		return this->mMeans;
	}

	virtual const std::vector<IDX_T>& getAssignment() override
	{
		this->fetchResult();
		return this->mAssignment;
	}

	virtual void cleanup() override
	{
		IKMeansAlgorithm<F, IDX_T>::cleanup();
		mCuData.free();
		mCuMeans[0].free();
		mCuMeans[1].free();
		mCuAssignment[0].free();
		mCuAssignment[1].free();
		mCuClusterSizes.free();
	}
};



template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = AoSLayoutPolicy<>,
	class LAYOUT_MEANS = LAYOUT, class METRIC = EuclideanMetricPolicy<F, LAYOUT, LAYOUT_MEANS>,
	class KERNEL = BaseAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>>
class CudaFusedKMeansAlgorithm : public IKMeansAlgorithm<F, IDX_T>
{
private:
	CudaExecParameters mExecParams;
	bool mResultReady;

	const F* mData;

	bpp::CudaBuffer<F> mCuData;
	bpp::CudaBuffer<F> mCuMeans[2];
//	bpp::CudaBuffer<IDX_T> mCuAssignment[2];
	bpp::CudaBuffer<IDX_T> mCuClusterSizes;
	IDX_T mLast;

	void fetchResult()
	{
		if (mResultReady) return;

		mCuMeans[mLast].read(this->mMeans);
		//mCuAssignment[mLast].read(this->mAssignment);
		CUCH(cudaDeviceSynchronize());
		mResultReady = true;
	}

public:
	CudaFusedKMeansAlgorithm(const CudaExecParameters& exec) : mExecParams(exec), mResultReady(false), mData(nullptr), mLast(0) {}

	virtual void initialize(std::size_t dim, std::size_t k, std::size_t n, std::size_t iters,
		const std::vector<F>& data, const std::vector<F>& means) override
	{
		IKMeansAlgorithm<F, IDX_T>::initialize(dim, k, n, iters, data, means);
		mResultReady = false;

		mData = &data[0];

		std::size_t devices = bpp::CudaDevice::count();
		if (devices == 0) throw bpp::RuntimeError("No CUDA devices found!");

		CUCH(cudaSetDevice(0));
		mCuData.realloc(LAYOUT::size(this->mN, this->mDim));
		mCuClusterSizes.realloc(this->mK);
		for (std::size_t i = 0; i < 2; ++i) {
			mCuMeans[i].realloc(LAYOUT_MEANS::size(this->mK, this->mDim) * mExecParams.privatizedCopies);
			//mCuAssignment[i].realloc(this->mN);
		}

		mCuData.write(data);

		mLast = 0;
		mCuMeans[mLast].write(means);
		//runFillZeros<IDX_T>(*mCuAssignment[mLast], this->mN);
		runFillZeros<IDX_T>(*mCuClusterSizes, this->mK);
		CUCH(cudaDeviceSynchronize());
	}

	virtual void run() override
	{
		for (std::size_t iter = 0; iter < this->mIters; ++iter) {
			auto next = (mLast + 1) % 2;
			runFillZeros<F>(*mCuMeans[next], LAYOUT_MEANS::size(this->mK, this->mDim) * mExecParams.privatizedCopies);
			runFillZeros<IDX_T>(*mCuClusterSizes, this->mK);

			FusedProblemInstance<F, IDX_T> fused(*mCuData, *mCuMeans[mLast], *mCuMeans[next], *mCuClusterSizes, this->mDim, this->mK, this->mN);
			KERNEL::run(fused, mExecParams);
			CUCH(cudaGetLastError());

			//UpdateProblemInstance<F, IDX_T> update(*mCuData, *mCuAssignment[next], *mCuMeans[next], *mCuClusterSizes, this->mDim, this->mK, this->mN);
			//UPDATE_KERNEL::run(update, mExecParams);

			mLast = next;
		}
		CUCH(cudaDeviceSynchronize());
	}

	virtual const std::vector<F>& getMeans() override
	{
		this->fetchResult();
		return this->mMeans;
	}

	virtual const std::vector<IDX_T>& getAssignment() override
	{
		this->fetchResult();
		this->mAssignment.resize(this->mN);

		// TODO -- assignment should be saved in the last iteration of fused kernel
		auto precomputedLayoutData = LAYOUT::precomputeConstants(this->mN, this->mDim);
		auto precomputedLayoutMeans = LAYOUT_MEANS::precomputeConstants(this->mK, this->mDim);
		for (IDX_T i = 0; i < this->mN; ++i) {
			IDX_T closestIdx = 0;
			F closestDist = METRIC::distance(mData, i, precomputedLayoutData, &this->mMeans[0], 0, precomputedLayoutMeans, this->mDim);
			for (IDX_T j = 1; j < this->mK; ++j) {
				F dist = METRIC::distance(mData, i, precomputedLayoutData, &this->mMeans[0], j, precomputedLayoutMeans, this->mDim);
				if (dist < closestDist) {
					closestDist = dist;
					closestIdx = j;
				}
			}
			this->mAssignment[i] = closestIdx;
		}

		return this->mAssignment;
	}

	virtual void cleanup() override
	{
		IKMeansAlgorithm<F, IDX_T>::cleanup();
		mCuData.free();
		mCuMeans[0].free();
		mCuMeans[1].free();
//		mCuAssignment[0].free();
//		mCuAssignment[1].free();
		mCuClusterSizes.free();
	}
};



template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = AoSLayoutPolicy<>,
	class LAYOUT_MEANS = LAYOUT, class METRIC = EuclideanMetricPolicy<F, LAYOUT, LAYOUT_MEANS>,
	class KERNEL = BaseAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>>
class CudaAssignmentAlgorithm : public IAssignmentAlgorithm<F, IDX_T>
{
private:
	CudaExecParameters mExecParams;
	bool mResultReady;

	bpp::CudaBuffer<F> mCuData;
	bpp::CudaBuffer<F> mCuMeans;
	bpp::CudaBuffer<IDX_T> mCuAssignment;

public:
	CudaAssignmentAlgorithm(const CudaExecParameters& exec) : mExecParams(exec), mResultReady(false) {}

	virtual void initialize(std::size_t dim, std::size_t k, std::size_t n,
		const std::vector<F>& data, const std::vector<F>& means) override
	{
		IAssignmentAlgorithm<F, IDX_T>::initialize(dim, k, n, data, means);
		mResultReady = false;

		std::size_t devices = bpp::CudaDevice::count();
		if (devices == 0) throw bpp::RuntimeError("No CUDA devices found!");

		CUCH(cudaSetDevice(0));
		mCuData.realloc(LAYOUT::size(this->mN, this->mDim));
		mCuMeans.realloc(LAYOUT_MEANS::size(this->mK, this->mDim));
		mCuAssignment.realloc(this->mN);

		mCuData.write(data);
		mCuMeans.write(means);

		CUCH(cudaDeviceSynchronize());
	}

	virtual void run() override
	{
		AssignmentProblemInstance<F, IDX_T> instance(*mCuData, *mCuMeans, nullptr, *mCuAssignment, this->mDim, this->mK, this->mN);
		KERNEL::run(instance, mExecParams);
		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize());
	}

	virtual const std::vector<IDX_T>& getAssignment() override
	{
		if (!mResultReady) {
			mResultReady = true;
			mCuAssignment.read(this->mAssignment);
			CUCH(cudaDeviceSynchronize());
		}

		return this->mAssignment;
	}

	virtual void cleanup() override
	{
		IAssignmentAlgorithm<F, IDX_T>::cleanup();
		mCuData.free();
		mCuMeans.free();
		mCuAssignment.free();
	}
};


#endif
