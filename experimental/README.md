# Experimental Code

This folder holds the experimental code used for our measurements. It is not meant to be directly used as k-means library, but individual parts (most importantly the kernels) may be studied or taken individually. The experimental projects use generated random data of given size and measure execution times (i.e., the performance).

The code is divided into two parts for historical reasons. The `bucketing-cuda` project originally investigated the performance of atomic updates and it becomes basis for the implementation of the update step only. The `k-means` project was created later and introduced also the assignment step and the complete k-means solutions (both combined executions of assignment+update kernels and fused kernels).

Furthermore, both project require CUDA 10.2 already installed. The code should be compatible with CUDA 10.x, but the VS project files have the explicit version hardcoded in the project configuration.

## Compilation

Both projects requires [bpplib](https://github.com/krulis-martin/bpplib) library, a C++ header-only library. The easiest way is to download/clone the library and then provide link to its `include` directory. 

* **Windows** - Set up an environmental variable `BPPLIB_DIR` pointing to `include` directory of the `bpplib`. The VS projects are already use this variable in include paths.
* **Linux** - You will probably need to edit the `Makefile`. Both make files have configuration variables set at the beginning. Update the `INCLUDE` variable (fix the `bpplib` path inside).

After that the projects should be easily compiled by VS Studio / GNU make. We have tested it only with the most recent VS Studio 2019 and g++ 8.3.

## Execution

Both projects create an application with command-line interface. The measured results are printed to stdout and additional comments to stderr. If stdout and stderr is interleaved on console, it makes human-readable output; for automated (batch) processing, the stderr should be redirected (e.g., to `/dev/null`).

All execution arguments use "long" format only (i.e. `--` prefix). Numeric argument values may use `k`, `M`, or `G` suffixes that multiply the value by `2^10`, `2^20`, and `2^30` respectively (e.g., `--N 4k` means that 4096 points will be used).

Common arguments for both projects:
* `--N <int>` - number of points (vectors in matrix X)
* `--k <int>` - number of clusters (vectors in W)
* `--dim <int>` - data dimensionality
* `--algorithm <string>` - name of the tested algorithm (explained below)
* `--layout <string>` - data layout type (`aos` and `soa` correspond to Array-of-structures and Structure-of-arrays; `aos32` and `soa32` are their aligned counterparts)
* `--seed <int>` - initialization number for the random generator (for deterministic debugging and experiments)
* `--verify` - if present, the results will be verified by slow serial algorithm (so we can test our CUDA algorithms work correctly)
* `--double` - if present, all data are represented as 64-bit floats (instead of 32-bit); this feature was not thoroughly tested and works only on GPUs with CC 6.0 and above 
* `--cudaBlockSize` - kernel execution parameter -- number of threads per block (total number of threads is not affected, only division in blocks)
* `--cudaSharedMemorySize` - maximal size of shared memory allocated per CUDA block (affects only algorithms that utilize shared memory) 
* `--privatizedCopies` - number of privatized copies of matrix W in global memory allocated for the update step with atomic synchronization (not applied to other algorithms); if more than 1, reduction is performed after the update kernel 
* `--itemsPerThread` - how many items (depending on selected algorithm) is processed by each CUDA thread (total number of spawned threads is divided by this number)


### Update Step

The update step computes new version of centroids (means) from the points and their cluster assignment. Basically, this step performs cluster-wise reduction which computes an average per dimension. The experiments are implemented in the `bucketing-cuda` project. The most efficient algorithms use atomic operations for updates.

Example:

```
$> ./bucketing-cuda --algorithm cuda_atomic_fine --layout aos --N 1M --dim 20 --k 2k
```

The algorithm names correspond to the names in the paper and the allowed values are the following:

* `serial` - naive serial implementation (that uses one CPU thread)
* `cuda_base` - baseline algorithm that does not require synchronization (runs `d` x `k` threads, each thread iterates over all `N` points)
* `cuda_atomic_point` - runs kernel with atomic updates where each thread process one point (`N` threads, each performs `d` atomic updates)
* `cuda_atomic_warp` - runs kernel with atomic updates where warp of threads cooperatively process one point (`32N` threads, each performs `d/32` atomic updates on average)
* `cuda_atomic_fine` - runs kernel with atomic updates where each thread process one point (`N` threads, each performs `d` atomic updates)
* `cuda_atomic_shm` - like `cuda_atomic_fine`, but uses shared memory to cache a private copy of the W matrix (try to hold as large part of W as the shm size permits, whole matrix if possible)
* `cuda_atomic_shm2` - extension of shm algorithm that uses multiple privatized copies in shared memory (if shared memory size permits) 
* `cuda_fake` - fake algorithm performs only the atomic writes (no data reads), so it tests the maximal throughput of memory updates 
* `cuda_fake_shm` - similar to `cuda_fake`, but performs the updates in shared memory


### Assignment Step

The assignment step is actually the first step of each iteration and it identifies the nearest cluster (centroid) for each data point using Euclidean metrics. To test this step alone, the `k-means` project must be executed with `--assignmentOnly` argument.

Example:

```
$> ./k-means --assignmentOnly --algorithm cuda_cached_means --layout soa --N 512k --dim 32 --k 128
```

Algorithms:
* `serial` - naive serial implementation (that uses one CPU thread)
* `cuda_base` - naive CUDA implementation with no explicit optimizations (one thread per point, thread computes distances to all means and finds the nearest one)
* `cuda_cached_means` - optimized algorithm that caches all means in shared memory
* `cuda_cached_fixed` - optimized algorithm with fixed caching window
* `cuda_cached_regs` - optimized algorithm that cache slice of means in shared memory and some input values in registers as well

Additionally, the cached algorithms may be affected by `--regsCache` command line argument. It specifies, how many registers are used for caching (details are in the paper). In case of `cuda_cached_regs`, the `regsCache` value is used both as `r_p` and `r_q` (the cached matrix is a square).


### Complete k-means

Complete k-means use two type of algorithms: *Combined* algorithms execute the kernels from the previous steps in a sequence (without any modifications), *fused* algorithms execute kernels that were created by fusing the code from the two steps into one function (so some data may be passed by registers or shared memory from one step to the other).

Algorithms:
* `serial` - naive serial implementation (that uses one CPU thread)
* `cuda_base` - combined `cuda_base` kernel for assignment and `cuda_atomic_point` kernel for update step (this is probably the closest algorithm to many open-source implementations we have encountered)
* `cuda_base_shm` - same as `cuda_base` but `cuda_atomic_shm` is used for update (most straightforward attempt to optimize the baseline using shared memory)
* `cuda_best` - combined algorithm that selects the best assignment kernel (`regs` or `fixed`) based on given configuration (`k` and `dim`) and uses `cuda_atomic_shm` for update
* `cuda_fused_fixed` - fused kernel based on the `fixed` assignment and `atomic_ponts` update
* `cuda_fused_regs` - fused kernel based on the `regs` assignment and special version of atomic update that is somewhere between `atomic_point` and `atomic_fine` (multiple number of threads cooperate on one point -- the number is equal to `itemsPerThread * regsCount`)
* `kmcuda` - our integration of the [kmcuda](https://github.com/src-d/kmcuda) implementation of YinYang optimization; the code was retaken from the original repository and integrated only for the purposes of measuring the performance under the same conditions (do not use this code from our repository)

Full k-means have one important additional command-line argument: `--iterations <int>` which limits the number of iterative refinements. In case of our implementations, this is the exact number of iterations (we do not employ additional testing for algorithm convergence), in case of [kmcuda](https://github.com/src-d/kmcuda), this value represents an upper limit (algorithm can terminate early).


## Results

The most important output of the experimental code is the measured wall-clock time. The measurements comprise only the kernel execution, not the data transfers. Two values are yielded to the stdout -- first one is the *total time* per iteration (in milliseconds) and the second value is *normalized time* (in nanoseconds), which is basically the total time per iteration divided by `dim * n`.

Some additional information (e.g., initialization time which also includes the time required to copy data to the GPU) are printed to stderr.
