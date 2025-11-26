/*
 * Demonstrates unified memory programming on CUDA Managed Memory.
 * Official reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming
 *
 * On systems where managed memory is supported, we experience seamless use of memory across CPU and GPU. 
 * The necessary copying of data between CPU memory and GPU memory happens "behind the scenes". 
 * 
 * A program using managed memory may need to check the support level of the system. 
 * Some systems treat all pointers as unified memory, while others need specific use of "cudaMallocManaged" and "cudaFree". 
 */

#include <stdio.h>



/*
 * Adds vector a and vector b and stores the result in vector a.
 */
__global__ void add(int* a, int* b) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    a[i] = a[i] + b[i];
}



int main() {
    int* vector_a; 
    cudaMallocManaged(&vector_a, 256 * sizeof(int));
    int* vector_b; 
    cudaMallocManaged(&vector_b, 256 * sizeof(int));

    for (int i = 0; i < 256; i++) {
        vector_a[i] = i;
        vector_b[i] = 256 - i;
    }

    add<<<1, 256>>>(vector_a, vector_b);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaError::cudaSuccess) {
        printf("A CUDA error occurred: %s", cudaGetErrorString(err));
        return 1;
    }

    int result_sum = 0;

    for (int i = 0; i < 256; i++) {
        result_sum += vector_a[i];
    }

    printf("Result: sum = %d\n", result_sum);

    cudaFree(vector_a);
    cudaFree(vector_b);
}
