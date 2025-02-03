# 1. Introduction to GPU and CUDA C/C++ for Parallel Computing 

<div align="center">
  <img src="https://github.com/user-attachments/assets/1435ae44-1a2c-40c1-9372-da6a48296326" alt="image" width="500">
</div>

## Overview

This guide will introduce you to:

- What's GPU? 
- What's the diffrence between GPU and CPU?
- What's CUDA and why?
- The basics of CUDA and why it’s significant?
- Core concepts in CUDA programming
- Your first CUDA Application

Let’s dive in!

## What's GPU? 
A Graphics Processing Unit (GPU) is designed for parallel processing, making it ideal for handling large-scale computations. While originally developed for rendering graphics, GPUs are now widely used for tasks like scientific simulations, deep learning, and big data analytics.

## GPU vs CPU
<div align="center">
  <img src="https://github.com/user-attachments/assets/dd93e384-2f59-4bb2-ae2b-d59449650b36" alt="image" width="250">
</div>

We can think about it like if we planed to prepare a big feast..

You have to chop vegetables, cook dishes, bake desserts, and set the table. How would you do all that work? we have two scenarios: 

### The CPU: 
Like a "Solo Chef"
Think of a CPU like a skilled chef working alone in the kitchen. This chef is great at multitasking maybe they can chop vegetables while keeping an eye on the oven and stirring a pot. 
But no matter how skilled they are, they can only do a few things at once. If there’s a ton of work to do, things will start to accumulating up.

### The GPU: 
Like a "Team of Cooks"
Now, imagine we have a team of 100 cooks instead of just one chef. Each cook is assigned a single, specific task, one person only chops carrots, another stirs a pot, someone else sets the table. 

They might not be as versatile as the solo chef, but they can do all work at the same time. With all these cooks working in parallel, the entire feast is prepared much faster.

This is how a GPU works. While a CPU handles a few complex tasks at once, a GPU breaks down the work into thousands of smaller, simpler tasks and handles them all simultaneously.

***The Big Difference:***
- **CPU:** Great for doing different tasks, but it handles them one after the other or in small groups.
- **GPU:** Best for doing a lot of similar tasks all at once. It’s built for parallelism lots of things happening at the same time.

## Why Learn CUDA?
So, Once we’ve got our team of cooks (the GPU), but how do you tell them what to do? That’s where CUDA comes in.
so CUDA is like the recipe book and the work plan we give to our cooking team. CUDA is a toolkit provided by NVIDIA that allows us to tells the GPU how to organize its tasks. CUDA gives you the power to program the GPU for tasks beyond just graphics, like data processing, machine learning, and scientific simulations.

In today's world, applications are increasingly data-heavy and complex. Traditional CPUs process tasks sequentially, which limits performance for parallel workloads. GPUs, with their thousands of cores, can execute many tasks simultaneously, making them ideal for high-performance applications like simulations, machine learning, and image processing. CUDA unlocks the full potential of GPU parallelism, enabling substantial acceleration in computing.

## Real-World Applications

- **Signal Processing and Communications:** CUDA boosts performance in real-time tasks like radar processing, communication protocols, and sensor networks by enabling fast and efficient algorithms such as FFTs and filtering.

- **Embedded and Autonomous Systems:** CUDA powers applications in automotive and aerospace by enabling rapid prototyping and real-time processing for tasks like sensor fusion and control systems.

- **Video and Image Processing:** From smoother streaming and faster video editing to enhanced image analysis, CUDA accelerates tasks like real-time encoding, decoding, and object detection, making it essential for AR, computer vision, and surveillance systems.

- **Scientific Simulations:** Whether it’s simulating molecular interactions or large-scale physical models, CUDA handles the heavy lifting where traditional CPUs fall short in speed and scale.

- **Advanced Data Analytics:** For research labs dealing with massive datasets, CUDA speeds up statistical modeling, experimental data analysis, and other high-dimensional data tasks, delivering results faster and more efficiently.

## Basic Concepts in CUDA

Before you start coding or cooking :) , it’s important to grasp some basic CUDA concepts:

1. **Host and Device:**  
   The Host refers to the CPU, while the Device refers to the GPU. CUDA programming involves data transfer between the Host and Device.

2. **Kernel Functions:**  
   A kernel is a function that runs on the GPU, executed in parallel by multiple threads or the recipe instruction set that all our cooks follow.

3. **Threads, Blocks, and Grids:**  
   CUDA follows a hierarchical structure:
   - **Threads:** The smallest unit of execution or the individual cook doing a single task .
   - **Blocks:** Groups of threads or like a group of cooks working on a specific part.
   - **Grids:** Groups of blocks or all the cooks in the kitchen :).

4. **Why Blocks and Grids?**
   - GPUs organize threads into blocks and grids to manage parallelism efficiently.
   - Each block can have up to 1024 threads (on most GPUs), so for large datasets, we use multiple blocks.
   - For example, processing 5000 elements with 256 threads per block requires 20 blocks (5000 / 256  Approx. = 19.53 → 20 blocks).
   

So, you’ll need to define the number of threads, blocks, and grids based on the problem you’re solving or the dishes you plan to cook :).
 
<div align="center">

<img width="800" alt="image" src="https://github.com/user-attachments/assets/043ed75a-2540-4d43-8874-a50ff8e80128">
</div>

## CUDA Programming Model

The CUDA Toolkit makes it easy to manage thousands of parallel threads (cooks or workers) without complicated code. It provides simple commands for controlling how these tasks run and how they use memory.

This way, we can focus more on creating efficient algorithms and less on the technical details of parallel processing...

## Setting Up Your CUDA Environment

To get started with CUDA development, follow the installation guides for your operating system:
- [NVIDIA CUDA Installation Guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

## Writing the First CUDA application!

Clone the repo:

clone `/My_First_CUDA.cu` via:

```
git clone https://github.com/SumayyahAli/Getting_Started_with_CUDA_.git
```


`My_First_CUDA.cu`:
```cu
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
```
## How our Code Works?:
<div align="center">

![image](https://github.com/user-attachments/assets/bc447e36-7807-4e9d-8d2f-6232f9addf8f)

</div>

**1. Kernel Function (`add`):**

The `__global__` keyword defines the `add` function as a CUDA kernel, meaning it runs directly on the GPU. 

Each thread " worker :)" operates on one element of the arrays, adding corresponding values from `a` and `b` and storing the result in `c`.

**2. Memory Management:** 

  Memory is allocated on both the host (CPU) and device (GPU) using cudaMalloc. 
  
  The data is then copied from the host(CPU) arrays (`h_a`, `h_b`) to the device (GPU) arrays (`d_a`, `d_b`) using `cudaMemcpy`.

**3. Grid and Block Configuration:**

The kernel is launched with a configuration defined by `dim3 name(x, y, z);`:
- `dim3 threadsPerBlock(n, 1, 1)`; sets n threads in a block (each thread handles one element).
- `dim3 blocksPerGrid(1, 1, 1)`; specifies a single block in the grid (since our data is small).
This setup ensures that all n elements are processed in parallel.

**4. Kernel Launch:**

 The kernel is launched using `add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);`. This syntax specifies how the work is distributed across the GPU.

**5. Copied Results and Cleanup:**

After computation, the results are copied back to the host, displayed, and the device memory is freed using `cudaFree`.

## Output
This shows that the arrays are added correctly in parallel.
<div align="center">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/98f6adfa-75ab-46a6-b3ce-3806153f1f17">
</div>


## The End of the "Beginning" :) 

You’ve taken your first steps into the world of GPU programming with CUDA.

From understanding how GPUs work to writing your first parallel application!

However remember, this is just the end of the beginning... CUDA opens up an entirely new world of possibilities, and there’s so much more to explore.

***with more practice, you can tackle even more complex challenges.....***


## Further Learning Resources:
- [CUDA by Example:](https://developer.nvidia.com/cuda-example) The best starting book for GPU Programming and beginner friendly.
- [CUDA Programming Guide:](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) A comprehensive guide with deeper insights into CUDA.
- [An Even Easier Introduction to CUDA:](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) Practical beginner friendly resource with practical examples to enhance your skills.


