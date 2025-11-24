/*
 * First example for any CUDA tutorial project. Add two vectors. 
 *
 * Checkout the Fireship video with this example. 
 * https://www.youtube.com/watch?v=pPStdjuYzSI 
 * 
 * Variants of this simple example appear in several intros to CUDA. 
 * 
 * To run it, 
 * - Obtain the CUDA Toolkit at https://developer.nvidia.com/cuda-toolkit 
 * - Get the Microsoft C++ compiler (cl.exe) included with Visual Studio. 
 * (May need to add to path manually.)
 */

#include <stdio.h>

/*
 * A kernel function. Here to add vectors together.
 * The same instruction will run in parallel on multiple CUDA kernels, with distinct indexing.
 */
__global__ void add(int* a, int* b, int* c) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] + b[i];
}

/*
 * Managed memory is an alternative to manually allocating GPU and CPU shared memory. 
 */
__managed__ int vector_a[256], vector_b[256], vector_c[256];

int main() {
    for (int i = 0; i < 256; i++) {
        vector_a[i] = i;
        vector_b[i] = 256 - i;
    }

    add<<<1, 256>>>(vector_a, vector_b, vector_c);  // Instructs the GPU to add these vectors using 256 kernels. 
    cudaError_t status = cudaDeviceSynchronize();   // Await GPU. 

    /* This is likely insufficient error handling. Still nice to know that it is a thing. */
    if (status != ::cudaSuccess) {
        printf("An error occurred.");
        return 1;
    }

    int result_sum = 0;

    for (int i = 0; i < 256; i++) {
        result_sum += vector_c[i];
    }

    printf("Result: sum = %d\n", result_sum);
}
