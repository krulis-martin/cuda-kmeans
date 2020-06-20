# Highly optimized CUDA implementation of k-means algorithm

A novel, highly-optimized CUDA implementation of the k-means clustering algorithm. The approach is documented in paper *Detailed Analysis and Optimization of CUDA K-means Algorithm*, currently accepted to ICPP'20 conference (in print).

This repository contains:

- [k-means imeplentation and experimental code](experimental/) used for benchmarking
  - the actual [CUDA k-means](experimental/k-means/)
  - a [microbenchmark](experiemental/bucketing-cuda/) of bucket-wise sum of matrices in CUDA
- The [measured results](results/) for several recent GPUs

The code is ready to be extracted and used for other projects. We hope to provide wrappers for several popular programming and scientific computing environments (Python/numpy and R) in near future.

## How fast is it?

We measured a speedup between roughly 10x and 1000x (depending on data size and dimensionality) over the current best open-source implementations (kmcuda). Our approach does not use any indexing structures, and relies only on the low-level optimizations and raw throughput of the GPUs. For 1024 clusters on 2 million datapoints in 32 dimensions, the implementation can run one k-means iteration in around

- **104ms** on nVidia GTX 980
- **25ms** on nVidia V100 SMX2

All collected measurements are [available here](results/).

## License

The code is available under MIT license.

If you find any part of this project useful for your scientific research, please cite the paper mentioned above. (We will add a full citation when the paper is published).

