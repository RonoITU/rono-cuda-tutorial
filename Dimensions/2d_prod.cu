/*
 * Demonstrates managed "in-place" mod of a 2D grid with scalar multiplication. 
 */

#include <stdio.h>



const int N = 8;
const int VECTOR_SIZE = N*N;

#define RIDX(i,j,n) ((i)*(n)+(j))



/*
 * Kernel function to write the Hilbert matrix.
 * For this to work correctly, indexing must be correct. 
 */
__global__ void hilbert(float* a) {
    int column = threadIdx.x + blockDim.x * blockIdx.x;
    int row    = threadIdx.y + blockDim.y * blockIdx.y;
    a[row * N + column] = 1.0 / float(1+column+row);
}



void displayMatrix(float* v) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", v[RIDX(i, j, N)]);
        }
        printf("\n");
    }
    printf("\n");
}



/*
 * Vectors as managed memory. 
 */
__managed__ float vector_a[VECTOR_SIZE];

int main() {
    // Set entire array to 8.88
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            vector_a[RIDX(i, j, N)] = 8.88;
        }
    }

    displayMatrix(vector_a); // Before.

    // The "dim3" struct is used for 2D and 3D organization of operations. 
    dim3 block(8, 4); // = 32 threads or one "warp". 
    dim3 grid(N/block.x, N/block.y); // (1, 2) -> Upper and lower half.

    // The dim3 info is passed with the kernel call. 
    // Recall: <<<grid_size, block_size>>>
    hilbert<<<grid, block>>>(vector_a);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaError::cudaSuccess) {
        printf("An error occurred: %s", cudaGetErrorString(err));
        return 1;
    }

    displayMatrix(vector_a); // After.

    return 0;
}
