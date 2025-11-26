/*
 * This is a basic program to fetch key properties about CUDA GPUs on the system. 
 * Far more data is available for different applications. Check out "driver_types.h" of the Computing Toolkit. 
 */

#include <stdio.h>
#include <map>
#include <string>

int deviceCount(void)
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        std::exit(EXIT_FAILURE); 
    }
    return deviceCount;
}

void displayMemoryInfo() 
{
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("Free GPU memory:  %zd MB\n", freeMem/(1024*1024));
    printf("Total GPU memory: %zd MB\n", totalMem/(1024*1024));
}

void displayDeviceProperties(cudaDeviceProp* prop, int deviceNumber)
{
    printf("Device %d: \n", deviceNumber);
    printf("  Name: %s\n", prop->name);
    printf("  Compute capability: %d.%d\n", prop->major, prop->minor);
    printf("  Multiprocessor count: %d\n", prop->multiProcessorCount);
    
    printf("  Device memory: %zd MB\n", prop->totalGlobalMem/(1024*1024));
    printf("  Shared memory per block: %zd KB \n", prop->sharedMemPerBlock/1024);
    printf("  Warp size: %d\n", prop->warpSize);
    printf("  Max threads per block: %d\n", prop->maxThreadsPerBlock);

    // Managed memory support:
    printf("  Managed memory support: %s\n", prop->managedMemory ? "Yes" : "No");               // - Windows or CUDA < 6.0. 
    printf("  Concurrent managed access: %s\n", prop->concurrentManagedAccess ? "Yes" : "No");  // - 6.x or better not on Windows. 
    printf("  Pageable memory access: %s\n", prop->pageableMemoryAccess ? "Yes" : "No");        // - Allows CUDA to access regular OS allocations (malloc). 
}

int main() 
{
    printf("Hello CUDA dev!\n");

    displayMemoryInfo();
    printf("\n");

    int devices = deviceCount();
    for (int i = 0; i < devices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        displayDeviceProperties(&prop, i);
        printf("\n");
    }
    return 0;
}

