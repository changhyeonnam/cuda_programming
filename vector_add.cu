//
// Created by changhyeonnam on 2023/01/10.
//

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// CUDA kernel for vector addition
// __global__ means this is called from CPU, and runs on the GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N){
    // Calculate global thread ID
    // blockDim = 1 dim (just integer)
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid<N)
        // Each thread adds a single element
        c[tid] = a[tid] + b[tid];
}

// Initialize vector of size n to int between 0~99
void matrix_init(int* a, int n){
    for(int i=0; i<n; i++){
        a[i] = rand() % 100;
    }
}
// Check vector add result
void error_check(int* a, int* b, int* c, int n){
    for(int i=0; i<n; i++){
        assert(c[i] == a[i] + b[i]);
    }
}

// print vector add result
void print_result(int* a, int* b, int* c, int n){
    for(int i=0; i<n; i++){
        if(i%100==0)
            std::cout<<"c["<<i<<"]="<<c[i]<<" = "<<"a["<<i<<"]="<<a[i]<<" + " <<"b["<<i<<"]="<<b[i]<<'\n';
    }
}


int main(){
    // Vector size of 2^16 (65536 elements)
    int n = 1<<16;

    // Host vector pointers
    int *h_a, *h_b, *h_c;

    // Device vector pointers
    int *d_a, *d_b, *d_c;

    // Allocation size for all vectors
    size_t bytes = sizeof(int) * n;

    // Allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // Allocate device(gpu) memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    /* There is something called unified memory.
     * one set of memory that gets migrated between the GPU and CPU viceversa.
     * [next lecture]
     */

    // Initialize vectors a and b with random values between 0 and 99
    matrix_init(h_a, n);
    matrix_init(h_b, n);

    // Copy data from the CPU(HOST) to the GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Threadblock size
    // it's generally good to do this a size of 32 because these have to translate it to warps.
    // which are of size 32.
    int NUM_THREADS = 256;

    // Grid size
    // NUM_THREAD * NUM_BLOCKS = NUMBER of Elements.
    int NUM_BLOCKS = (int)ceil(n/NUM_THREADS);

    // Launch kernel on default strem w/o
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);

    // Copy sum vector from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result for errors
    error_check(h_a, h_b, h_c, n);
    print_result(h_a, h_b, h_c, n);
    printf("COMPLETED SUCCESFULLY\n");
    return 0;
}
