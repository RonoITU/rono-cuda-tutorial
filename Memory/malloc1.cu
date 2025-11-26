/*
 * Demonstrates standard device memory allocation and data transfers on CUDA.
 * Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory
 *
 * While newer systems support CUDA Managed Memory, there may be reasons to allocate device memory and 
 * manage data transfers manually. 
 * 
 * With Managed Memory available, it is suggested that programmers wait
 * tackle the problem of parallelization and creating the application first, becore considering manual 
 * allocation and data flow setups. 
 * Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-introduction
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
    // CUDA Device Memory allocation. 
    // Recall the error handling necessary for malloc in general. 
    // cudaMallocManaged may return an error value. 
    int* device_vector_a; 
    cudaMalloc(&device_vector_a, 256 * sizeof(int));
    int* device_vector_b; 
    cudaMalloc(&device_vector_b, 256 * sizeof(int));

    // Make sure to keep track of host vs. device pointers in the code. 
    // Type systems or a naming convention will be of great aid here. 

    // Allocate space on host for vector data. 
    int* host_vector = (int*) malloc(256 * sizeof(int));

    for (int i = 0; i < 256; i++)
        host_vector[i] = i;

    cudaMemcpy(device_vector_a, host_vector, 256 * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < 256; i++)
        host_vector[i] = 256 - i;

    cudaMemcpy(device_vector_b, host_vector, 256 * sizeof(int), cudaMemcpyHostToDevice);

    add<<<1, 256>>>(device_vector_a, device_vector_b);

    // In this version, the synchronize call is optional. 
    // The Memcpy calls give adequate synchronization, 
    // so it is here to catch any other errors that may have occurred. 
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaError::cudaSuccess) {
        printf("A CUDA error occurred: %s", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(host_vector, device_vector_a, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    int result_sum = 0;

    for (int i = 0; i < 256; i++) {
        result_sum += host_vector[i];
    }

    printf("Result: sum = %d\n", result_sum);

    // CUDA Memory free. 
    // cudaFree may return an error value. 
    cudaFree(device_vector_a);
    cudaFree(device_vector_b);
    free(host_vector);
}
