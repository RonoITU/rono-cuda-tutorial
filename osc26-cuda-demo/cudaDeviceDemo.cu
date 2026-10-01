#include <stdio.h>
#include <sys/time.h>

const int N = 1 << 28;



/*
 * Adds vector a and vector b and stores the result in vector a.
 */
template <const uint BLOCKSIZE>
__global__ void add (
    int* __restrict__ a, // __restrict__ : "These point to distinct areas of memory."
    int* __restrict__ b, 
    int* __restrict__ c,
    const int n          // Realistically, this kernel needs to support varying size workloads. 
) 
{
    const unsigned int bound = min ( BLOCKSIZE + BLOCKSIZE * blockIdx.x, n );
    unsigned int i = threadIdx.x + BLOCKSIZE * blockIdx.x;
    while ( i < bound ) {
        c[i] = a[i] + b[i];
        i += blockDim.x;
    }
}



int main() {
    int* a;
    int* b;
    int* r;



    // Using the unified memory management. This means our data
    // transfers between host and device as required with little 
    // setup. 
    printf ( "allocating arrays...\n" );
    cudaError_t e1 = cudaMallocManaged ( &a, N * sizeof(int) );
    cudaError_t e2 = cudaMallocManaged ( &b, N * sizeof(int) );
    cudaError_t e3 = cudaMallocManaged ( &r, N * sizeof(int) );

    if (e1 || e2 || e3) {
        printf ( "CUDA error on allocation.\n" );
        cudaFree (a);
        cudaFree (b);
        cudaFree (r);
        return 1;
    }



    printf ( "populating arrays...\n" );
    for ( unsigned int i = 0 ; i < N ; i++ ) {
        a[i] = 1;
        b[i] = 2;
    }



    // Kernel launch tuning parameters. 
    const int BLOCKSIZE = 1 << 14;
    const int THREADS_PER_BLOCK = 128;



    // In a realistic scenario we have calculate how many blocks are needed
    // at runtime based on the size of a, b, r. Not from constants. 

    int blocks = N / BLOCKSIZE + ( N % BLOCKSIZE > 0 );
    printf ( "%d blocks are needed.\n", blocks );



    struct timeval ts, te;
    printf ( "Kernel launch! (clock started)\n" );
    gettimeofday ( &ts, NULL );

    add <BLOCKSIZE><<<blocks, THREADS_PER_BLOCK>>> (a, b, r, N);



    // We can launch any sequence of kernels as needed to do the job. 
    // But before the host can touch the results again, we have to synchronize
    // with the device, blocking the host if necessary to wait. 
    cudaError_t err = cudaDeviceSynchronize ();

    gettimeofday(&te, NULL);
    printf ( "Kernel finished! (clock stopped)\n" );

    if ( err != cudaError::cudaSuccess ) {
        printf ( "%s\n", cudaGetErrorString(err) );
    }

    

    printf ( "took %g seconds to execute.\n", ( te.tv_sec - ts.tv_sec + (te.tv_usec - ts.tv_usec)/1.e6 ) );

    for ( unsigned int i = 0 ; i < N ; i++ ) {
        if (r[i] != 3) {
            printf ( "This has not done the job right :P\n" ); 
            return 1;
        }
    }

    return 0;
}
