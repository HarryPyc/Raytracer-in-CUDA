
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>


void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA error: %s: %s. \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


__global__ void render(uchar4 *pos, int width, int height) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) 
		return;

	int index = j * width + i;

	unsigned char r = int(float(i) / width * 255.99) & 0xff;
	unsigned char g = int(float(j) / height * 255.99) & 0xff;
	unsigned char b = (70) & 0xff;

	pos[index].w = 0;
	pos[index].x = r;
	pos[index].y = g;
	pos[index].z = b;
}

extern "C" void launch_kernel(uchar4* pos, unsigned int w, unsigned int h) {

	int tx = 8;
	int ty = 8;

	dim3 blocks(w / tx + 1, h / ty + 1);
	dim3 threads(tx, ty);
	render <<<blocks, threads >>> (pos, w, h);


	cudaThreadSynchronize();
	checkCUDAError("kernel failed!");
}

