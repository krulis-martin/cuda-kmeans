#ifndef K_MEANS_DATA_HPP
#define K_MEANS_DATA_HPP

#include "layout_policies.hpp"
#include "metric_policies.hpp"

#include "system/file.hpp"

#include <algorithm>
#include <vector>
#include <random>
#include <limits>
#include <iostream>

/**
 *
 */
template<typename T = std::uint32_t, class DIST = std::uniform_int_distribution<T>>
void generate_random(std::vector<T> & data, DIST distribution, std::size_t seed = std::numeric_limits<std::size_t>::max())
{
	std::random_device rd;
	std::mt19937 generator(seed == std::numeric_limits<std::size_t>::max() ? rd() : (unsigned int)seed);
	for (auto&& x : data) {
		x = distribution(generator);
	}
}


/**
 *
 */
template<typename F = float, class LAYOUT = SoALayoutPolicy<32>>
void generate_clustered_data(std::vector<F> & data, std::size_t n, std::size_t k, std::size_t dim, std::size_t seed = std::numeric_limits<std::size_t>::max())
{
	std::random_device rd;
	std::mt19937 generator(seed == std::numeric_limits<std::size_t>::max() ? rd() : (unsigned int)seed);

	std::uniform_real_distribution<F> uni((F)0, (F)1);
	std::uniform_int_distribution<std::size_t> uniIndex(0, k - 1);
	std::normal_distribution<F> norm((F)0, (F)0.025);

	data.resize(LAYOUT::size(n, dim));
	auto precomputedData = LAYOUT::precomputeConstants(n, dim);
	for (std::size_t i = 0; i < k; ++i) {
		for (std::size_t d = 0; d < dim; ++d) {
			LAYOUT::at(data.data(), i, d, precomputedData) = uni(generator);
		}
	}

	for (std::size_t i = k; i < n; ++i) {
		auto target = uniIndex(generator);
		for (std::size_t d = 0; d < dim; ++d) {
			LAYOUT::at(data.data(), i, d, precomputedData) = norm(generator) + LAYOUT::at(data.data(), target, d, precomputedData);
		}
	}
}


/**
 *
 */
template<typename F, typename IDX_T, class LAYOUT, class LAYOUT_MEANS>
void prepare_means(const std::vector<F>& data, std::vector<F>& means, std::size_t n, std::size_t k, std::size_t dim, bool shuffle, std::size_t seed = std::numeric_limits<std::size_t>::max())
{
	std::vector<IDX_T> indices(n);
	std::generate(indices.begin(), indices.end(), [i = 0]() mutable { return i++; });

	if (shuffle) {
		std::random_device rd;
		std::mt19937 generator(seed == std::numeric_limits<std::size_t>::max() ? rd() : (unsigned int)seed);
		std::shuffle(indices.begin(), indices.end(), generator);
	}

	means.resize(LAYOUT_MEANS::size(k, dim));
	for (std::size_t i = 0; i < k; ++i) {
		auto source = indices[i];
		for (std::size_t d = 0; d < dim; ++d) {
			LAYOUT_MEANS::at(means.data(), i, d, LAYOUT_MEANS::precomputeConstants(k, dim)) = LAYOUT::at(data.data(), source, d, LAYOUT::precomputeConstants(n, dim));
		}
	}
}


/**
 *
 */
template<typename F = float>
bool verify(const std::vector<F> & res, const std::vector<F> & correctRes, std::size_t dims, std::size_t k)
{
	if (res.size() != correctRes.size()) {
		std::cerr << std::endl << "Error: Result size mismatch (" << res.size() << " values found, but " << correctRes.size() << " values expected)!" << std::endl;
		return false;
	}

	if (res.size() % dims != 0) {
		std::cerr << std::endl << "Error: Result size (" << res.size() << ") is not divisible by dimension (" << dims << ")!" << std::endl;
		return false;
	}

	if (res.size() / dims % k != 0) {
		std::cerr << std::endl << "Error: Number of result vectors (" << res.size() / dims << ") is not divisible by # of buckets (" << k << ")!" << std::endl;
		return false;
	}

	std::size_t errorCount = 0;
	constexpr double floatTolerance = 0.00005;
	for (std::size_t i = 0; i < res.size(); ++i) {
		double d1 = (double)res[i], d2 = (double)correctRes[i];
		double divisor = std::max(std::abs(d1) + std::abs(d2), floatTolerance);
		double err = std::abs(d1 - d2) / divisor;
		if (err > floatTolerance) {
			if (errorCount == 0) std::cerr << std::endl;

			if (++errorCount <= 10) {
				std::cerr << "Error at iteration #" << (i / dims) / k << " [" << (i / dims) % k << "][" << (i % dims) << "]: " << res[i] << " != " << correctRes[i] << std::endl;
			}
		}
	}

	if (errorCount > 0) {
		std::cerr << "Total errors found: " << errorCount << std::endl;
	}

	return errorCount == 0;
}


/**
 *
 */
template<typename IDX_T = std::uint32_t>
std::size_t verify_assignment(const std::vector<IDX_T> & res, const std::vector<IDX_T> & correctRes, std::size_t dims, std::size_t n)
{
	if (res.size() != correctRes.size()) {
		std::cerr << std::endl << "Error: Assignment size mismatch (" << res.size() << " values found, but " << correctRes.size() << " values expected)!" << std::endl;
		return n;
	}

	if (res.size() != n) {
		std::cerr << std::endl << "Error: Assignment size (" << res.size() << ") is not equal to number of points (" << n << ")!" << std::endl;
		return n;
	}

	std::size_t errorCount = 0;
	for (std::size_t i = 0; i < res.size(); ++i) {
		if (res[i] != correctRes[i]) {
			if (errorCount == 0) std::cerr << std::endl;

			if (++errorCount <= 10) {
				std::cerr << "Error at [" << i << "]: " << res[i] << " != " << correctRes[i] << std::endl;
			}
		}
	}

	if (errorCount > 0) {
		std::cerr << "Total assignment errors found: " << errorCount << std::endl;
	}

	return errorCount;
}


template<typename F = float, class LAYOUT = SoALayoutPolicy<32>, class METRIC = EuclideanMetricPolicy<F, LAYOUT, LAYOUT>>
std::size_t verify_means(const std::vector<F>& res, const std::vector<F>& correctRes, std::size_t dims, std::size_t k)
{
	if (res.size() != correctRes.size()) {
		std::cerr << std::endl << "Error: Means size mismatch (" << res.size() << " values found, but " << correctRes.size() << " values expected)!" << std::endl;
		return correctRes.size();
	}

	if (res.size() != LAYOUT::size(k, dims)) {
		std::cerr << std::endl << "Error: Means vector size (" << res.size() << ") is not equal to expected size (" << LAYOUT::size(k, dims) << ")!" << std::endl;
		return LAYOUT::size(k, dims);
	}

	std::size_t errorCount = 0;
	auto precomputed = LAYOUT::precomputeConstants(k, dims);
	F distSum = (F)0.0;
	for (std::size_t i = 0; i < k; ++i) {
		F dist = METRIC::distance(res.data(), i, precomputed, correctRes.data(), i, precomputed, dims);
		distSum += dist;
		if (dist > 0.1) {
			if (errorCount == 0) std::cerr << std::endl;

			if (++errorCount <= 10) {
				std::cerr << "Error at [" << i << "]: means dist is " << dist << std::endl;
			}
		}
	}

	if (errorCount > 0) {
		std::cerr << "Total significant errors found: " << errorCount << std::endl;
	}

	distSum /= (F)k;
	if (distSum > 0.01) {
		++errorCount;
		std::cerr << "Average displacement of means is " << distSum << " !" << std::endl;
	}

	return errorCount;
}


/**
 *
 */
template<typename F = float, class LAYOUT = SoALayoutPolicy<32>>
void writeRList(bpp::TextWriter & writer, const std::vector<F> & data, std::size_t n, std::size_t dim, std::size_t selectedDim)
{
	writer.write("c(");
	auto precomputedData = LAYOUT::precomputeConstants(n, dim);
	for (std::size_t i = 0; i < n; ++i) {
		writer.write(LAYOUT::at(data.data(), i, selectedDim, precomputedData));
		if (i < n - 1) {
			writer.write(", ");
			if (i % 128 == 127) writer.writeLine();
		}
	}
	writer.write(")");
	writer.writeLine();
}


/**
 *
 */
template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = SoALayoutPolicy<32>, class LAYOUT_MEANS = SoALayoutPolicy<32>>
void saveAsRScript(const std::string & fileName, const std::vector<F> & data, const std::vector<F> & means, const std::vector<IDX_T> & assignment,
	std::size_t n, std::size_t k, std::size_t dim)
{
	std::cerr << "Saving results as R script '" << fileName << "'" << std::endl;
	bpp::File file(fileName);
	file.open();
	bpp::TextWriter writer(file, "\n", "");

	writer.write("x = ");
	writeRList<F, LAYOUT>(writer, data, n, dim, 0);
	writer.write("y = ");
	writeRList<F, LAYOUT>(writer, data, n, dim, 1);

	writer.write("plot(x, y, col=rgb(0,100,0,50,maxColorValue=255), pch=16)");
	writer.writeLine();

	std::vector<std::size_t> clusterSizes;
	clusterSizes.resize(k);
	for (auto target : assignment) {
		++clusterSizes[target];
	}

	std::cout << k << std::endl;
	auto precomputedMeans = LAYOUT_MEANS::precomputeConstants(k, dim);
	for (std::size_t i = 0; i < k; ++i) {
		writer.write("draw.circle(");
		writer.write(LAYOUT_MEANS::at(means.data(), i, 0, precomputedMeans));
		writer.write(", ");
		writer.write(LAYOUT_MEANS::at(means.data(), i, 1, precomputedMeans));
		writer.write(", ");
		writer.write(0.2 * std::sqrt((double)clusterSizes[i] / (double)n));
		writer.write(")");
		writer.writeLine();
	}
}




#endif
