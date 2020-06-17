#define _CRT_SECURE_NO_WARNINGS

#include "serial.hpp"
#include "cuda.hpp"
#include "interface.hpp"
#include "layout_traits.hpp"

#include "cli/args.hpp"
#include "system/stopwatch.hpp"
#include "system/file.hpp"

#include <iostream>
#include <map>
#include <string>
#include <memory>
#include <cstdint>


template<typename T = std::uint32_t, class DIST = std::uniform_int_distribution<T>>
void generate_random(std::vector<T>& data, DIST distribution, std::size_t seed = std::numeric_limits<std::size_t>::max())
{
	std::random_device rd;
	std::mt19937 generator(seed == std::numeric_limits<std::size_t>::max() ? rd() : (unsigned int)seed);
	for (auto&& x : data) {
		x = distribution(generator);
	}
}


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
	exec.itemsPerThread= args.getArgInt("itemsPerThread").getAsUint32();

	std::cerr << "Cuda device #0 selected with CC " << props.major << "." << props.minor << " (" << exec.sharedMemorySize/1024 << "kB shared mem)" << std::endl;
}


template<typename F = float, class LAYOUT = SoALayoutTrait<32>, int EMUL = false>
void createAtomicAlgorithms(std::map<std::string, std::unique_ptr<IBucketingAlgorithm<F>>> &algorithms, CudaExecParameters& cudaExecParams)
{
	algorithms["cuda_atomic_point"] = std::make_unique<CudaBucketingAlgorithm<F, LAYOUT, AtomicPointKernel<F, LAYOUT, EMUL>>>(cudaExecParams);
	algorithms["cuda_atomic_warp"] = std::make_unique<CudaBucketingAlgorithm<F, LAYOUT, AtomicWarpKernel<F, LAYOUT, EMUL>>>(cudaExecParams);
	algorithms["cuda_atomic_fine"] = std::make_unique<CudaBucketingAlgorithm<F, LAYOUT, AtomicFineKernel<F, LAYOUT, EMUL>>>(cudaExecParams);
	algorithms["cuda_atomic_shm"] = std::make_unique<CudaBucketingAlgorithm<F, LAYOUT, AtomicShmKernel<F, LAYOUT, EMUL>>>(cudaExecParams);
	algorithms["cuda_atomic_shm2"] = std::make_unique<CudaBucketingAlgorithm<F, LAYOUT, AtomicShm2Kernel<F, LAYOUT, EMUL>>>(cudaExecParams);
	algorithms["cuda_fake"] = std::make_unique<CudaBucketingAlgorithm<F, LAYOUT, AtomicFakeKernel<F, LAYOUT, EMUL>>>(cudaExecParams);
	algorithms["cuda_fake_shm"] = std::make_unique<CudaBucketingAlgorithm<F, LAYOUT, AtomicShmFakeKernel<F, LAYOUT, EMUL>>>(cudaExecParams);
}


template<typename F = float, class LAYOUT = SoALayoutTrait<32>>
std::unique_ptr<IBucketingAlgorithm<F>> getAlgorithm(bpp::ProgramArguments& args)
{
	using map_t = std::map<std::string, std::unique_ptr<IBucketingAlgorithm<F>>>;
	auto algoName = args.getArgString("algorithm").getValue();
	bool emul = args.getArgBool("emul").getValue();

	CudaExecParameters cudaExecParams;
	getCudaExecParameters(args, cudaExecParams);

	map_t algorithms;
	algorithms["serial"] = std::make_unique<SerialBucketingAlgorithm<F, LAYOUT>>();
	algorithms["cuda_base"] = std::make_unique<CudaBucketingAlgorithm<F, LAYOUT, BaseKernel<F, LAYOUT>>>(cudaExecParams);
	if (emul) {
		createAtomicAlgorithms<F, LAYOUT, true>(algorithms, cudaExecParams);
	}
	else {
		createAtomicAlgorithms<F, LAYOUT, false>(algorithms, cudaExecParams);
	}

	auto it = algorithms.find(algoName);
	if (it == algorithms.end()) {
		throw (bpp::RuntimeError() << "Unkown algorithm '" << algoName << "'.");
	}

	std::cerr << "Selected algorithm: " << algoName << std::endl;
	return std::move(it->second);
}


template<typename F = float>
bool verify(const std::vector<F>& res, const std::vector<F>& correctRes, std::size_t dims, std::size_t k)
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


template<typename F = float>
void saveResults(const std::string& fileName, const std::vector<F>& result, std::size_t dim)
{
	bpp::File file(fileName);
	file.open();
	bpp::TextWriter writer(file);
	
	for (std::size_t i = 0; i < result.size(); /* i is incremented in the loop */) {
		for (std::size_t d = 0; d < dim; ++d)
			writer.writeToken(result[i++]);

		writer.writeLine();
	}
}


template<typename F = float, class LAYOUT = SoALayoutTrait<32>>
void run(bpp::ProgramArguments& args)
{
	std::size_t dim = (std::size_t)args.getArgInt("dim").getValue();
	std::size_t k = (std::size_t)args.getArgInt("k").getValue();
	std::size_t n = (std::size_t)args.getArgInt("N").getValue();
	std::size_t iter = (std::size_t)args.getArgInt("iterations").getValue();

	std::vector<F> data(LAYOUT::size(n, dim) * iter);
	std::vector<std::uint32_t> indices(n * iter);

	std::cerr << "Generating input data (" << n << " x " << dim << "), " << iter << " iterations ..." << std::endl;
	std::size_t seed = (args.getArg("seed").isPresent()) ? args.getArgInt("seed").getValue() : std::numeric_limits<std::size_t>::max();
	if (args.getArg("seed").isPresent()) {
		std::cerr << "Seed is set to " << seed << std::endl;
	}
	generate_random(data, std::uniform_real_distribution<F>((F)0.0, (F)1.0), seed);
	generate_random(indices, std::uniform_int_distribution<std::uint32_t>(0, (std::uint32_t)k - 1), seed);

	auto algorithm = getAlgorithm<F, LAYOUT>(args);
	bpp::Stopwatch stopwatch;

	std::cerr << "Initializing ... "; std::cerr.flush();
	stopwatch.start();
	algorithm->initialize(dim, k, n, iter, data, indices);
	stopwatch.stop();
	std::cerr << stopwatch.getMiliseconds() << " ms" << std::endl;
	
	std::cerr << "Running ... "; std::cerr.flush();
	stopwatch.start();
	algorithm->run();
	stopwatch.stop();
	std::cout << (stopwatch.getMiliseconds() / (double)iter); std::cout.flush();
	std::cerr << " ms/iter (which is"; std::cerr.flush();
	std::cout << " " << (stopwatch.getMiliseconds() * 1000000.0 / (double)(iter * dim * n)); std::cout.flush();
	std::cerr << " ns/input float)"; std::cerr.flush();
	std::cout << std::endl;

	if (args.getArgBool("verify").getValue()) {
		std::cerr << "Verifying results ... "; std::cerr.flush();
		SerialBucketingAlgorithm<F, LAYOUT> baseAlgorithm;
		baseAlgorithm.initialize(dim, k, n, iter, data, indices);
		baseAlgorithm.run();
		if (verify(algorithm->getResult(), baseAlgorithm.getResult(), dim, k))
			std::cerr << "OK" << std::endl;
		else
			std::cerr << "FAILED" << std::endl;
	}

	if (args.getArg("out").isPresent()) {
		auto outFile = args.getArgString("out").getValue();
		std::cerr << "Saving results to " << outFile << " ..." << std::endl;
		saveResults(outFile, algorithm->getResult(), dim);
	}

	algorithm->cleanup();
	std::cerr << "And we're done here." << std::endl;
}


template<typename F>
void select_layout_and_run(bpp::ProgramArguments& args)
{
	auto layout = args.getArgEnum("layout").getValue();

	// Yay, string switch by multiple if-elses, what a beauty...
	if (layout == "soa") {
		run<F, SoALayoutTrait<1>>(args);
	}
	else if (layout == "soa32") {
		run<F, SoALayoutTrait<32>>(args);
	}
	else if (layout == "aos") {
		run<F, AoSLayoutTrait<1>>(args);
	}
	else if (layout == "aos32") {
		run<F, AoSLayoutTrait<32>>(args);
	}
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
		args.registerArg<bpp::ProgramArguments::ArgInt>("k", "Number of buckets.", false, 512, 2);
		args.registerArg<bpp::ProgramArguments::ArgInt>("N", "Number of vectors in a batch.", false, 10240, 128);
		args.registerArg<bpp::ProgramArguments::ArgInt>("iterations", "Iterations to be performed (i.e., number of batches).", false, 1, 1);
		args.registerArg<bpp::ProgramArguments::ArgInt>("seed", "Random generator seed.", false, 42, 0);

		args.registerArg<bpp::ProgramArguments::ArgString>("algorithm", "Which algorithm is to be tested.", false, "serial");
		args.registerArg<bpp::ProgramArguments::ArgBool>("emul", "Emulate atomic additions with atomic CAS operations.");

		args.registerArg<bpp::ProgramArguments::ArgEnum>("layout", "Type of data layout and alignment.", false, true, "aos");
		args.getArgEnum("layout").addOptions({ "aos", "soa", "aos32", "soa32" });

#ifndef NO_DOUBLES
		args.registerArg<bpp::ProgramArguments::ArgBool>("double", "Use doubles instead of floats.");
#endif
		args.registerArg<bpp::ProgramArguments::ArgBool>("verify", "Verify results against base (serial) algorithm.");

		args.registerArg<bpp::ProgramArguments::ArgString>("out", "Path to a file where the results are saved.");

		args.registerArg<bpp::ProgramArguments::ArgInt>("privatizedCopies", "Number of privatized copies in global memory (relevant only to some algorithms).", false, 1, 1, 4096);
		args.registerArg<bpp::ProgramArguments::ArgInt>("itemsPerThread", "Number of computations performed per CUDA thread.", false, 1, 1);

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
