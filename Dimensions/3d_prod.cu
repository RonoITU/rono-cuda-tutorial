/*
 * Demonstrates managed "in-place" mod of a 3D grid with scalar multiplication. 
 * Notice the similarity between this file and 2d_prod.cu. 
 */

#include <stdio.h>



const int N = 8;
const int VECTOR_SIZE = N*N*N;

#define RIDX(i,j,k,n) ((i)*(n)*(n)+(j)*(n)+(k))



/*
 * Kernel function to write the Hilbert matrix.
 * For this to work correctly, indexing must be correct. 
 */
__global__ void hilbert_3D(float* a) {
    int column = threadIdx.x + blockDim.x * blockIdx.x;
    int row    = threadIdx.y + blockDim.y * blockIdx.y;
    int layer  = threadIdx.z + blockDim.z * blockIdx.z;
    a[layer * N * N + row * N + column] = 1.0 / float(1+column+row+layer);
}



void displayMatrix_3D(float* v) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                printf("%.2f ", v[RIDX(i, j, k, N)]);
            }
            printf("\n");
        }
        printf("\n");
    }
}



/*
 * Vectors as managed memory. 
 */
__managed__ float vector_a[VECTOR_SIZE];

int main() {
    // Set entire array to 8.88
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                vector_a[RIDX(i, j, k, N)] = 8.88;
            }   
        }
    }

    printf("Starting matrix: \n\n");
    displayMatrix_3D(vector_a); // Before.

    // The "dim3" struct is used for 2D and 3D organization of operations. 
    dim3 block(8, 4, 1); // = 32 threads or one "warp". 
    dim3 grid(N/block.x, N/block.y, N/block.z); // (1, 2, 8) -> Upper and lower half. One layer at a time.

    // The dim3 info is passed with the kernel call. 
    // Recall: <<<grid_size, block_size>>>
    hilbert_3D<<<grid, block>>>(vector_a);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaError::cudaSuccess) {
        printf("An error occurred: %s", cudaGetErrorString(err));
        return 1;
    }

    printf("After running kernel: \n\n");
    displayMatrix_3D(vector_a); // After.

    return 0;
}
