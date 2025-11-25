/*
 * Demonstrates 1D grid and explains the block size limit.
 */

#include <stdio.h>

const int VECTOR_SIZE = 2048;

/*
 * Kernel function to add two vectors together. 
 */
__global__ void add(int* a, int* b, int* c) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] + b[i];
}

/*
 * Vectors as managed memory. 
 */
__managed__ int vector_a[VECTOR_SIZE], vector_b[VECTOR_SIZE], vector_c[VECTOR_SIZE];



int main() {
    for (int i = 0; i < VECTOR_SIZE; i++) {
        vector_a[i] = i;
        vector_b[i] = (VECTOR_SIZE - i);
    }

    /*
     * Blocks are limited to 1024 threads. 
     * (This may vary, older cards have a lower limit. Really, your program should find this info through the CUDA API.)
     * 
     * Larger jobs need to be divided into a grid of blocks. I.e. 2 x 1024 like here. 
     * 
     * Nice block sizes are divisible by 32, as that is the count of threads 
     * in a standard SM module. (Look at whitepapers for NVIDIA GPU architectures.)
     * We want to avoid "partial warp usage" when possible. 
     * 
     * It is not necessarily a good idea to hit the 1024 thread limit, nor to make blocks as small as possible. 
     * You should experiment with different block sizes for your workloads. 
     * 
     * The computation can fail to complete if blocks are oversized. 
     */
    add<<<2, 1024>>>(vector_a, vector_b, vector_c);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaError::cudaSuccess) {
        printf("An error occurred: %s", cudaGetErrorString(err));
        return 1;
    }

    int result_sum = 0;

    for (int i = 0; i < VECTOR_SIZE; i++) {
        result_sum += vector_c[i];
    }

    printf("Result: sum = %d\n", result_sum);
}
