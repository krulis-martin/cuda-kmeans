# Highly optimized CUDA implementation of k-means algorithm

A novel, highly-optimized CUDA implementation of the k-means clustering algorithm. The approach is documented in a conference paper here (link to the paper text can be found [here](https://www.ksi.mff.cuni.cz/~kratochvil/)):

> Kruliš, Martin, and Miroslav Kratochvíl. "[Detailed Analysis and Optimization of CUDA K-means Algorithm.](https://dl.acm.org/doi/abs/10.1145/3404397.3404426)" *49th International Conference on Parallel Processing -- ICPP*. 2020.

This repository contains:

- [k-means imeplentation and experimental code](experimental/) used for benchmarking
  - the actual [CUDA k-means](experimental/k-means/)
  - a [microbenchmark](experiemental/bucketing-cuda/) of bucket-wise sum of matrices in CUDA
- The [measured results](results/) for several recent GPUs

The code is ready to be extracted and used for other projects. We hope to provide wrappers for several popular programming and scientific computing environments (Python/numpy and R) in near future.

## How fast is it?

We measured a speedup between roughly 10x and 1000x (depending on data size and dimensionality) over the current best open-source implementations (kmcuda). Our approach does not use any indexing structures, and relies only on the low-level optimizations and raw throughput of the GPUs. For 1024 clusters on 2 million datapoints in 32 dimensions, the implementation can run one k-means iteration in around

- **104ms** on nVidia GTX 980
- **25ms** on nVidia V100 (SXM2)
- **18ms** on nVidia A100 (SXM4, hosted at [MeluXina supercomputer by LuxProvide](https://luxprovide.lu/))

All collected measurements are [available here](results/).

## License

The code is available under MIT license.

If you find any part of this project useful for your scientific research, please cite the paper mentioned above. The BibTeX entry would be as follows:

```
@inproceedings{krulis2020kmeans,
 author = {Kruli\v{s}, Martin and Kratochv\'{\i}l, Miroslav},
 title = {Detailed Analysis and Optimization of CUDA K-Means Algorithm},
 year = {2020},
 isbn = {9781450388160},
 publisher = {Association for Computing Machinery},
 address = {New York, NY, USA},
 url = {https://doi.org/10.1145/3404397.3404426},
 doi = {10.1145/3404397.3404426},
 booktitle = {49th International Conference on Parallel Processing - ICPP},
 articleno = {69},
 numpages = {11},
 keywords = {performance optimization, clustering, cuda, k-means},
 location = {Edmonton, AB, Canada},
 series = {ICPP '20}
}
```

