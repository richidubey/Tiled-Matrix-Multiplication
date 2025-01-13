# Tiled Matrix Muplication with GPU

Visualize performance benefits (in microsecond granularity) of mupltiplying a matrix with tiling in GPU as
compared to using naive O(n^3) multiplication by CPU.

## Set up

To download the repository and compile the program, run the following commands:    

```bash
git clone https://github.com/richidubey/Tiled-Matrix-Multiplication.git
cd Tiled-Matrix-Multiplication
nvcc matmul.cu
``` 

## Running

Run the following command to start the program. You can choose to specify the width of the matrix (default is 64)

```bash
./a.out -m 256
```

## Sample Runs

Output on a system with AMD EPYC 7502 32-Core Processor and NVIDIA A30 with 24 gigabytes (GB) of GPU memory with a bandwidth of 933 (GB/s):

```
$ ./a.out -m 200
GPU took: 2.89792
CPU took: 32.714
Speedup: 11.2888
```

```
$ ./a.out -m 2000
GPU took: 87.6943
CPU took: 33052.5
Speedup: 376.906
```
