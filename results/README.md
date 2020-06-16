# Experimental Results

This directory holds all measured values from our experiments. For your convenience, here is a quick preview:

| GPU | data points | dimension | k=16 | k=64 | k=256 | k=1024 | k=4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GTX 980 | 512k | 4 | 0.45ms | 0.68ms | 1.72ms | 6.36ms | 24.98ms |
| GTX 980 | 512k | 8 | 0.6ms | 0.79ms | 2.2ms | 8.28ms | 32.75ms |
| GTX 980 | 512k | 16 | 0.94ms | 1.28ms | 3.41ms | 12.89ms | 50.41ms |
| GTX 980 | 512k | 32 | 1.85ms | 2.52ms | 7.27ms | 25.55ms | 99.47ms |
| GTX 980 | 512k | 128 | 8.48ms | 12.64ms | 36.58ms | 127.53ms | 498.63ms |
| GTX 980 | 1M | 64 | 7.59ms | 10.22ms | 28.01ms | 98.31ms | 380.5ms |
| GTX 980 | 2M | 4 | 1.97ms | 2.71ms | 7.23ms | 24.95ms | 98.96ms |
| GTX 980 | 2M | 8 | 3.02ms | 3.7ms | 9.08ms | 33.01ms | 130ms |
| GTX 980 | 2M | 16 | 3.35ms | 6.27ms | 13.43ms | 51.89ms | 201.77ms |
| GTX 980 | 2M | 32 | 7.31ms | 10.25ms | 29.74ms | 104.48ms | 398.59ms |
| V100ms SXM2 | 2M | 4 | 0.82ms | 0.96ms | 1.33ms | 5.11ms | 20.09ms |
| V100ms SXM2 | 2M | 8 | 0.86ms | 1.07ms | 1.99ms | 7.59ms | 30.06ms |
| V100ms SXM2 | 2M | 16 | 1.05ms | 1.31ms | 3.34ms | 12.96ms | 50.8ms |
| V100ms SXM2 | 2M | 32 | 1.58ms | 2.32ms | 6.6ms | 25.45ms | 98.87ms |
| V100ms SXM2 | 2M | 128 | 5.52ms | 8.28ms | 26.89ms | 100.16ms | 382.98ms |
| V100ms SXM2 | 4M | 64 | 4.57ms | 7.36ms | 25.73ms | 98.19ms | 380.61ms |
| V100ms SXM2 | 8M | 4 | 3.85ms | 4.84ms | 6.35ms | 21.14ms | 83.51ms |
| V100ms SXM2 | 8M | 8 | 4.19ms | 5.02ms | 8.04ms | 30.63ms | 121.39ms |
| V100ms SXM2 | 8M | 16 | 6.7ms | 7.25ms | 14.26ms | 51.95ms | 207.12ms |
| V100ms SXM2 | 8M | 32 | 12.13ms | 16.56ms | 30.06ms | 102.86ms | 396.29ms |

### Data Formats

The results are kept in common CSV format.

TODO - explain headers
