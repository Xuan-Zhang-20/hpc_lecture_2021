# Student ID:20M30781, Name: ZHANG XUAN
# Testing Result on Tsubame (N=256)
## MPI(example code)
command:
module load gcc
moudle load intelmpi
mpicxx example.cpp -std=c++11
mpirun -np 4 ./a.out
| Time(s) | GFlops | Error  |
| -------- | -------- | -------- |
| 0.042151 | 0.793224 | 0.000016 |
## MPI+OpenMP
command:
module load gcc
moudle load intelmpi
mpicxx mpi_openmp.cpp -fopenmp -std=c++11
mpirun -np 4 ./a.out
| Time(s) | GFlops | Error  |
| -------- | -------- | -------- |
| 0.030098 | 1.077768 | 0.000016 |
## MPI+OpenMP+SIMD
command:
module load gcc
moudle load intelmpi
mpicxx mpi_openmp_simd.cpp -fopenmp -std=c++11 -fopt-info-optimized -march=native -o3
mpirun -np 4 ./a.out
| Time(s) | GFlops | Error  |
| -------- | -------- | -------- |
| 0.022396 | 0.035716 | 0.000016 |
## MPI+OpenMP+SIMD+CacheBlocking
command:
module load gcc
moudle load intelmpi
mpicxx mpi_openmp_simd_cacheblocking.cpp -fopenmp -std=c++11 -fopt-info-optimized -march=native -o3
mpirun -np 4 ./a.out
| Time(s) | GFlops | Error  |
| -------- | -------- | -------- |
| 0.023263 | 0.013397 | 0.000016 |
## MPI+CUDA
command:
module load cuda/11.2.146
module load openmpi
nvcc mpi_cuda.cu -lmpi -std=c++11
./a.out
| Time(s) | GFlops | Error  |
| -------- | -------- | -------- |
| 0.000007 | 78.593220 | 63.84586 |
# hpc_lecture

|          | Topic                                | Sample code               |
| -------- | ------------------------------------ | ------------------------- |
| Class 1  | Introduction to parallel programming |                           |
| Class 2  | Shared memory parallelization        | 02_openmp                 |
| Class 3  | Distributed memory parallelization   | 03_mpi                    |
| Class 4  | SIMD parallelization                 | 04_simd                   |
| Class 5  | GPU programming 1                    | 05_openacc                |
| Class 6  | GPU programming 2                    | 06_cuda                   |
| Class 7  | Parallel programing models           | 07_starpu                 |
| Class 8  | Cache blocking                       | 08_cache_cpu,08_cache_gpu |
| Class 9  | High Performance Python              | 09_python                 |
| Class 10 | I/O libraries                        | 10_io                     |
| Class 11 | Parallel debugger                    | 11_debugger               |
| Class 12 | Parallel profiler                    | 12_profiler               |
| Class 13 | Containers                           |                           |
| Class 14 | Scientific computing                 | 14_pde                    |
| Class 15 | Deep Learning                        | 15_dl                     |
