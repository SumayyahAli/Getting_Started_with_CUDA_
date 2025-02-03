#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// ------------[CUDA Kernel function to add elements of two arrays]---------------------
__global__ void add(int* a, int* b, int* c) 
{
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}

int main() 
{
    // Array size
    const int n = 10;
    int size = n * sizeof(int);

    // Host arrays [CPU]
    int h_a[n], h_b[n], h_c[n];

    // -------------------[Initialization]--------------------------
    for (int i = 0; i < n; i++)
{
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device arrays [GPU]
    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copying data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    /* 
    Define the number of threads per block and the number of blocks using dim3
    -----------------------------[ What is dim3? ]----------------------------------------
    The [dim3] data type in CUDA is used to define the dimensions of blocks and grids.
    It allows you to specify the number of threads in each block and the number of blocks in each grid.
    You can think of dim3 as a 3D vector with x, y, and z dimensions. In most simple cases 
    */

    dim3 threadsPerBlock(n, 1, 1);
    dim3 blocksPerGrid(1, 1, 1);

    //------------------------------[Launch kernel]-------------------------------
    add << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c);

    // Copying the result back to host [CPU]
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // ----------------------[Display the results]---------------------------------
    for (int i = 0; i < n; i++) 
    {
        cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;
    }

    // Free the device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
