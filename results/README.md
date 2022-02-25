# Experimental Results

This directory holds all measured values from our experiments. For your convenience, here is a quick preview:

| GPU | data points | dimension | k=16 | k=64 | k=256 | k=1024 | k=4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GTX 980 | 512k | 4 | 0.45ms | 0.68ms | 1.72ms | 6.36ms | 24.98ms |
| GTX 980 | 512k | 8 | 0.6ms | 0.79ms | 2.2ms | 8.28ms | 32.75ms |
| GTX 980 | 512k | 16 | 0.94ms | 1.28ms | 3.41ms | 12.89ms | 50.41ms |
| GTX 980 | 512k | 32 | 1.85ms | 2.52ms | 7.27ms | 25.55ms | 99.47ms |
| GTX 980 | 512k | 128 | 8.48ms | 12.64ms | 36.58ms | 127.53ms | 498.63ms |
| A100 SXM4 | 1M | 4 | 0.56ms | 0.86ms | 1.00ms | 2.33ms | 8.69ms |
| A100 SXM4 | 1M | 8 | 0.68ms | 1.10ms | 1.27ms | 3.18ms | 12.12ms |
| A100 SXM4 | 1M | 16 | 0.90ms | 1.26ms | 1.81ms | 5.18ms | 19.94ms |
| A100 SXM4 | 1M | 32 | 1.35ms | 1.75ms | 2.98ms | 9.42ms | 35.91ms |
| GTX 980 | 1M | 64 | 7.59ms | 10.22ms | 28.01ms | 98.31ms | 380.5ms |
| A100 SXM4 | 1M | 64 | 1.94ms | 2.84ms | 5.92ms | 19.96ms | 74.85ms |
| A100 SXM4 | 1M | 128 | 2.83ms | 5.27ms | 17.16ms | 63.34ms | 246.90ms |
| GTX 980 | 2M | 4 | 1.97ms | 2.71ms | 7.23ms | 24.95ms | 98.96ms |
| GTX 980 | 2M | 8 | 3.02ms | 3.7ms | 9.08ms | 33.01ms | 130ms |
| GTX 980 | 2M | 16 | 3.35ms | 6.27ms | 13.43ms | 51.89ms | 201.77ms |
| GTX 980 | 2M | 32 | 7.31ms | 10.25ms | 29.74ms | 104.48ms | 398.59ms |
| V100 SXM2 | 2M | 4 | 0.82ms | 0.96ms | 1.33ms | 5.11ms | 20.09ms |
| V100 SXM2 | 2M | 8 | 0.86ms | 1.07ms | 1.99ms | 7.59ms | 30.06ms |
| V100 SXM2 | 2M | 16 | 1.05ms | 1.31ms | 3.34ms | 12.96ms | 50.8ms |
| V100 SXM2 | 2M | 32 | 1.58ms | 2.32ms | 6.6ms | 25.45ms | 98.87ms |
| V100 SXM2 | 2M | 128 | 5.52ms | 8.28ms | 26.89ms | 100.16ms | 382.98ms |
| A100 SXM4 | 2M | 4 | 0.98ms | 1.61ms | 1.97ms | 4.61ms | 17.22ms |
| A100 SXM4 | 2M | 8 | 1.28ms | 2.04ms | 2.51ms | 6.30ms | 24.07ms |
| A100 SXM4 | 2M | 16 | 1.82ms | 2.49ms | 3.55ms | 10.23ms | 39.57ms |
| A100 SXM4 | 2M | 32 | 2.86ms | 3.68ms | 6.11ms | 18.83ms | 71.62ms |
| A100 SXM4 | 2M | 64 | 3.75ms | 5.53ms | 11.79ms | 39.36ms | 148.53ms |
| A100 SXM4 | 2M | 128 | 5.53ms | 10.33ms | 34.23ms | 126.04ms | 493.19ms |
| V100 SXM2 | 4M | 64 | 4.57ms | 7.36ms | 25.73ms | 98.19ms | 380.61ms |
| V100 SXM2 | 8M | 4 | 3.85ms | 4.84ms | 6.35ms | 21.14ms | 83.51ms |
| V100 SXM2 | 8M | 8 | 4.19ms | 5.02ms | 8.04ms | 30.63ms | 121.39ms |
| V100 SXM2 | 8M | 16 | 6.7ms | 7.25ms | 14.26ms | 51.95ms | 207.12ms |
| V100 SXM2 | 8M | 32 | 12.13ms | 16.56ms | 30.06ms | 102.86ms | 396.29ms |

The **results from A100 GPUs** are not reported in the paper as they were
collected _after_ the publication; we only provide them in this table to
complement the quickly accessible performance overview.

### Data Formats

The results are kept in common CSV format where tabs are delimiters. Each type of experiments is in a separate set of files:

* `assignment` -- only the assignment step
* `updates` -- only the update step
* `kmeans` -- complete k-means solution
* `atomic-fake` -- additional tests to determine throughput of atomic operations

The files are suffixed with the identification of the GPU on which the results were measured:

* `gtx980` for Maxwell (CC 5.2) architecture (GTX 980 gaming card)
* `v100` for Volta (CC 7.0) architecture (Tesla V100 SXM2 server card) 

The first line of the CSV file contains the name of the columns which roughly corresponds to the command line attributes (see the [experimental code](../experimental) readme). Additionally, `totalTime` column holds total time per iteration (in milliseconds) and `normalizedTime` column holds the normalized time (in nanoseconds).
