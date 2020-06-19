#define _CRT_SECURE_NO_WARNINGS

#include "serial.hpp"
#include "cuda.hpp"
#include "kmcuda_adapter.hpp"
#include "interface.hpp"
#include "data.hpp"
#include "metric_policies.hpp"
#include "layout_policies.hpp"

#include "cli/args.hpp"
#include "system/stopwatch.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <memory>
#include <cstdint>


void getCudaExecParameters(bpp::ProgramArguments& args, CudaExecParameters& exec)
{
	cudaDeviceProp props;
	CUCH(cudaGetDeviceProperties(&props, 0));

	exec.blockSize = (unsigned int)args.getArgInt("cudaBlockSize").getValue();
	if (exec.blockSize > (unsigned int)props.maxThreadsPerBlock)
		throw bpp::RuntimeError() << "Requested CUDA block size (" << exec.blockSize
			<< ") exceeds device capabilities (" << props.maxThreadsPerBlock << ").";

	exec.blockCount = (unsigned int)args.getArgInt("cudaBlockCount").getValue();
	if (exec.blockCount > (unsigned int)props.maxGridSize[0])
		throw bpp::RuntimeError() << "Requested CUDA block count (" << exec.blockCount
			<< ") exceeds device capabilities (" << props.maxGridSize[0] << ").";

	if (args.getArgInt("cudaSharedMemorySize").isPresent()) {
		exec.sharedMemorySize = (unsigned int)args.getArgInt("cudaSharedMemorySize").getValue();
		if (exec.sharedMemorySize > (unsigned int)props.sharedMemPerBlock)
			throw bpp::RuntimeError() << "Requested CUDA shared memory per block (" << exec.sharedMemorySize
				<< ") exceeds device capabilities (" << props.sharedMemPerBlock << ").";
	}
	else
		exec.sharedMemorySize = (unsigned int)props.sharedMemPerBlock;

	exec.privatizedCopies = args.getArgInt("privatizedCopies").getAsUint32();
	exec.itemsPerThread = args.getArgInt("itemsPerThread").getAsUint32();
	exec.regsCache = args.getArgInt("regsCache").getAsUint32();

	std::cerr << "Cuda device #0 selected with CC " << props.major << "." << props.minor << " (" << exec.sharedMemorySize/1024 << "kB shared mem)" << std::endl;
}


template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = SoALayoutPolicy<32>,
	class LAYOUT_MEANS = SoALayoutPolicy<32>, class METRIC = EuclideanMetricPolicy<F, LAYOUT, LAYOUT_MEANS>>
void run_assignment(bpp::ProgramArguments& args, std::vector<F> &data, std::vector<F> &means)
{
	std::size_t dim = (std::size_t)args.getArgInt("dim").getValue();
	std::size_t k = (std::size_t)args.getArgInt("k").getValue();
	std::size_t n = (std::size_t)args.getArgInt("N").getValue();
	auto algoName = args.getArgString("algorithm").getValue();

	CudaExecParameters cudaExecParams;
	getCudaExecParameters(args, cudaExecParams);

	using map_t = std::map<std::string, std::unique_ptr<IAssignmentAlgorithm<F>>>;
	map_t algorithms;
	algorithms["serial"] = std::make_unique<SerialAssignmentAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>>();
	algorithms["cuda_base"] = std::make_unique<CudaAssignmentAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, BaseAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>>>(cudaExecParams);
	algorithms["cuda_cached_fixed"] = std::make_unique<CudaAssignmentAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, CachedFixedAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>>>(cudaExecParams);
	algorithms["cuda_cached_means"] = std::make_unique<CudaAssignmentAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, CachedAllMeansAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>>>(cudaExecParams);
	algorithms["cuda_cached2"] = std::make_unique<CudaAssignmentAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, Cached2AssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>>>(cudaExecParams);
	algorithms["cuda_cached_regs"] = std::make_unique<CudaAssignmentAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, CachedRegsAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>>>(cudaExecParams);

	auto it = algorithms.find(algoName);
	if (it == algorithms.end()) {
		throw (bpp::RuntimeError() << "Unkown algorithm '" << algoName << "'.");
	}

	std::cerr << "Selected assignment-only algorithm: " << algoName << std::endl;
	auto algorithm = std::move(it->second);

	bpp::Stopwatch stopwatch;

	std::cerr << "Initializing ... "; std::cerr.flush();
	stopwatch.start();
	algorithm->initialize(dim, k, n, data, means);
	stopwatch.stop();
	std::cerr << stopwatch.getMiliseconds() << " ms" << std::endl;

	std::cerr << "Running (assignment only) ... "; std::cerr.flush();
	stopwatch.start();
	algorithm->run();
	stopwatch.stop();
	std::cout << stopwatch.getMiliseconds(); std::cout.flush();
	std::cerr << " ms (which is"; std::cerr.flush();
	std::cout << " " << (stopwatch.getMiliseconds() * 1000000.0 / (double)(dim * n)); std::cout.flush();
	std::cerr << " ns/input float)"; std::cerr.flush();
	std::cout << std::endl;

	if (args.getArgBool("verify").getValue() && algoName != "serial") {
		std::cerr << "Verifying results ... "; std::cerr.flush();
		SerialAssignmentAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC> baseAlgorithm;
		baseAlgorithm.initialize(dim, k, n, data, means);
		baseAlgorithm.run();
		if (verify_assignment<IDX_T>(algorithm->getAssignment(), baseAlgorithm.getAssignment(), dim, n) == 0)
			std::cerr << "OK" << std::endl;
		else
			std::cerr << "FAILED" << std::endl;
	}

	algorithm->cleanup();
}


template<typename F = float, typename IDX_T = std::uint32_t, class LAYOUT = SoALayoutPolicy<32>,
	class LAYOUT_MEANS = SoALayoutPolicy<32>, class METRIC = EuclideanMetricPolicy<F, LAYOUT, LAYOUT_MEANS>>
void run_kmeans(bpp::ProgramArguments& args, std::vector<F>& data, std::vector<F>& means)
{
	std::size_t dim = (std::size_t)args.getArgInt("dim").getValue();
	std::size_t k = (std::size_t)args.getArgInt("k").getValue();
	std::size_t n = (std::size_t)args.getArgInt("N").getValue();
	std::size_t iters = (std::size_t)args.getArgInt("iterations").getValue();
	auto algoName = args.getArgString("algorithm").getValue();

	CudaExecParameters cudaExecParams;
	getCudaExecParameters(args, cudaExecParams);

	using map_t = std::map<std::string, std::unique_ptr<IKMeansAlgorithm<F>>>;
	map_t algorithms;
	algorithms["serial"] = std::make_unique<SerialKMeansAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>>();
	algorithms["cuda_base"] = std::make_unique<CudaKMeansAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, BaseAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>, UpdateAtomicPointKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS>>>(cudaExecParams);
	algorithms["cuda_base_shm"] = std::make_unique<CudaKMeansAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, BaseAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>, UpdateAtomicShmKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS>>>(cudaExecParams);
	algorithms["cuda_best"] = std::make_unique<CudaKMeansAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, BestCachedAssignmentKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>, UpdateAtomicShmKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS>>>(cudaExecParams);
	algorithms["cuda_fused_fixed"] = std::make_unique<CudaFusedKMeansAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, FusedCachedFixedKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>>>(cudaExecParams);
	algorithms["cuda_fused_regs"] = std::make_unique<CudaFusedKMeansAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC, FusedCachedRegsKernel<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>>>(cudaExecParams);
	algorithms["kmcuda"] = std::make_unique<KMCudaAdapter<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC>>();

	auto it = algorithms.find(algoName);
	if (it == algorithms.end()) {
		throw (bpp::RuntimeError() << "Unkown algorithm '" << algoName << "'.");
	}

	std::cerr << "Selected k-means algorithm: " << algoName << std::endl;
	auto algorithm = std::move(it->second);

	bpp::Stopwatch stopwatch;

	std::cerr << "Initializing ... "; std::cerr.flush();
	stopwatch.start();
	algorithm->initialize(dim, k, n, iters, data, means);
	stopwatch.stop();
	std::cerr << stopwatch.getMiliseconds() << " ms" << std::endl;

	std::cerr << "Running (" << iters << " iterations) ... "; std::cerr.flush();
	stopwatch.start();
	algorithm->run();
	stopwatch.stop();
	
	if (algorithm->getIterations() != iters) {
		iters = algorithm->getIterations();
		std::cerr << "[stopped after " << iters << " iters.] ";
	}

	std::cout << stopwatch.getMiliseconds() / (double)iters; std::cout.flush();
	std::cerr << " ms/iter (which is"; std::cerr.flush();
	std::cout << " " << (stopwatch.getMiliseconds() * 1000000.0 / (double)(iters * dim * n)); std::cout.flush();
	std::cerr << " ns/input float)"; std::cerr.flush();
	std::cout << std::endl;

	if (args.getArgBool("verify").getValue() && algoName != "serial") {
		std::cerr << "Verifying results ... "; std::cerr.flush();
		SerialKMeansAlgorithm<F, IDX_T, LAYOUT, LAYOUT_MEANS, METRIC> baseAlgorithm;
		baseAlgorithm.initialize(dim, k, n, iters, data, means);
		baseAlgorithm.run();
		auto errors = verify_assignment<IDX_T>(algorithm->getAssignment(), baseAlgorithm.getAssignment(), dim, n)
			+ verify_means<F, LAYOUT_MEANS>(algorithm->getMeans(), baseAlgorithm.getMeans(), dim, k);
		if (errors == 0)
			std::cerr << "OK" << std::endl;
		else
			std::cerr << "FAILED" << std::endl;
	}

	if (args.getArg("Rscript").isPresent()) {
		saveAsRScript<F, IDX_T, LAYOUT, LAYOUT_MEANS>(args.getArgString("Rscript").getValue(), data, algorithm->getMeans(), algorithm->getAssignment(), n, k, dim);
	}

	algorithm->cleanup();
}


template<typename F = float, class LAYOUT = SoALayoutPolicy<32>, class LAYOUT_MEANS = SoALayoutPolicy<32>, class METRIC = EuclideanMetricPolicy<F, LAYOUT, LAYOUT_MEANS>>
void run(bpp::ProgramArguments& args)
{
	std::size_t dim = (std::size_t)args.getArgInt("dim").getValue();
	std::size_t k = (std::size_t)args.getArgInt("k").getValue();
	std::size_t n = (std::size_t)args.getArgInt("N").getValue();

	std::cerr << "Generating input data (" << n << " x " << dim << ") ..." << std::endl;
	std::size_t seed = (args.getArg("seed").isPresent()) ? args.getArgInt("seed").getValue() : std::numeric_limits<std::size_t>::max();
	if (args.getArg("seed").isPresent()) {
		std::cerr << "Seed is set to " << seed << std::endl;
	}
	
	std::vector<F> data(LAYOUT::size(n, dim));
	std::vector<F> means(LAYOUT_MEANS::size(k, dim));
	
	//generate_random(data, std::uniform_real_distribution<F>((F)0.0, (F)1.0), seed);
	generate_clustered_data<F, LAYOUT>(data, n, k, dim, seed);
	prepare_means<F, std::uint32_t, LAYOUT, LAYOUT_MEANS>(data, means, n, k, dim, true, seed);	// extract random points from data
	
	bool assignmentOnly = args.getArgBool("assignmentOnly").getValue();
	if (assignmentOnly) {
		run_assignment<F, std::uint32_t, LAYOUT, LAYOUT_MEANS, METRIC>(args, data, means);
	}
	else {
		run_kmeans<F, std::uint32_t, LAYOUT, LAYOUT_MEANS, METRIC>(args, data, means);
	}

	std::cerr << "And we're done here." << std::endl;
}


template<typename F, class LAYOUT, class LAYOUT_MEANS>
void select_metric_and_run(bpp::ProgramArguments& args)
{
	// TODO (after proper args are available)
	// Currently, only Euclidean is implemented in some kernels (revision required)
	run<F, LAYOUT, EuclideanMetricPolicy<F, LAYOUT, LAYOUT_MEANS>>(args);
}


template<typename F, class LAYOUT>
void select_layout2_and_run(bpp::ProgramArguments& args)
{
	if (!args.getArgEnum("layoutMeans").isPresent()) {
		run<F, LAYOUT, LAYOUT>(args);
		return;
	}
	
	auto layout = args.getArgEnum("layoutMeans").getValue();

	// Yay, string switch by multiple if-elses, what a beauty...
	if (layout == "soa") {
		run<F, LAYOUT, SoALayoutPolicy<1>>(args);
	}
	else if (layout == "aos") {
		run<F, LAYOUT, AoSLayoutPolicy<1>>(args);
	}
/*
	else if (layout == "soa32") {
		run<F, LAYOUT, SoALayoutPolicy<32>>(args);
	}
	else if (layout == "aos32") {
		run<F, LAYOUT, AoSLayoutPolicy<32>>(args);
	}
*/
	else {
		throw (bpp::RuntimeError() << "Unknown data layout '" << layout << "'.");
	}
}


template<typename F>
void select_layout_and_run(bpp::ProgramArguments& args)
{
	auto layout = args.getArgEnum("layout").getValue();

	// Yay, string switch by multiple if-elses, what a beauty...
	if (layout == "soa") {
		select_layout2_and_run<F, SoALayoutPolicy<1>>(args);
	}
	else if (layout == "aos") {
		select_layout2_and_run<F, AoSLayoutPolicy<1>>(args);
	}
/*
	else if (layout == "soa32") {
		select_layout2_and_run<F, SoALayoutPolicy<32>>(args);
	}
	else if (layout == "aos32") {
		select_layout2_and_run<F, AoSLayoutPolicy<32>>(args);
	}
*/
	else {
		throw (bpp::RuntimeError() << "Unknown data layout '" << layout << "'.");
	}
}


int main(int argc, char* argv[])
{
	/*
	 * Arguments
	 */
	bpp::ProgramArguments args(0, 0);

	try {
		args.registerArg<bpp::ProgramArguments::ArgInt>("dim", "Vector dimension.", false, 20, 2);
		args.registerArg<bpp::ProgramArguments::ArgInt>("k", "Number of buckets.", false, 128, 2);
		args.registerArg<bpp::ProgramArguments::ArgInt>("N", "Number of vectors in a batch.", false, 10240, 128);
		args.registerArg<bpp::ProgramArguments::ArgInt>("iterations", "Iterations to be performed.", false, 1, 1);
		args.registerArg<bpp::ProgramArguments::ArgInt>("seed", "Random generator seed.", false, 42, 0);

		args.registerArg<bpp::ProgramArguments::ArgBool>("assignmentOnly", "Run only assignment part of the algorithm (microbenchmark).");
		args.getArg("assignmentOnly").conflictsWith("iterations");
		args.registerArg<bpp::ProgramArguments::ArgString>("algorithm", "Which algorithm is to be tested.", false, "serial");

		args.registerArg<bpp::ProgramArguments::ArgEnum>("layout", "Type of data layout and alignment.", false, true, "aos");
		args.getArgEnum("layout").addOptions({ "aos", "soa" /*, "aos32", "soa32" */ });
		args.registerArg<bpp::ProgramArguments::ArgEnum>("layoutMeans", "Type of data layout and alignment (override for means only).", false, true, "aos");
		args.getArgEnum("layoutMeans").addOptions({ "aos", "soa" /*, "aos32", "soa32" */ });

#ifndef NO_DOUBLES
		args.registerArg<bpp::ProgramArguments::ArgBool>("double", "Use doubles instead of floats.");
#endif
		args.registerArg<bpp::ProgramArguments::ArgBool>("verify", "Verify results against base (serial) algorithm.");

		args.registerArg<bpp::ProgramArguments::ArgString>("Rscript", "Path to a file where the results are saved as a R script.");

		args.registerArg<bpp::ProgramArguments::ArgInt>("privatizedCopies", "Number of privatized copies in global memory (relevant only to some algorithms).", false, 1, 1, 4096);
		args.registerArg<bpp::ProgramArguments::ArgInt>("itemsPerThread", "Number of computations performed per CUDA thread.", false, 1, 1, 4096);
		args.registerArg<bpp::ProgramArguments::ArgInt>("regsCache", "Size of the registry cache (extreme caching algorithms).", false, 1, 1);

		// CUDA specific
		args.registerArg<bpp::ProgramArguments::ArgInt>("cudaBlockSize", "Number of threads in each block.", false, 256, 1, 4096);
		args.registerArg<bpp::ProgramArguments::ArgInt>("cudaBlockCount", "Number of blocks executed in a grid (some alogrithms may choose to ignore it).", false, 0, 0);
		args.registerArg<bpp::ProgramArguments::ArgInt>("cudaSharedMemorySize", "Amount of shared memory allocated per block.", false, 0, 0);

		// Process the arguments ...
		args.process(argc, argv);
	}
	catch (bpp::ArgumentException& e) {
		std::cout << "Invalid arguments: " << e.what() << std::endl << std::endl;
		args.printUsage(std::cout);
		return 1;
	}

	try {
		// Here might be a fork to run the algorithm using doubles instead of floats...
#ifndef NO_DOUBLES
		if (args.getArgBool("double").getValue())
			select_layout_and_run<double>(args);
		else
#endif
			select_layout_and_run<float>(args);
	}
	catch (std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl << std::endl;
		return 2;
	}

	return 0;
}
