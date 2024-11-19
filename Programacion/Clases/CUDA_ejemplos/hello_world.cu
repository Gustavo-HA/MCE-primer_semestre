#include <stdio.h>

__global__ void helloCUDA(float e)
{
    int idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello, I am thread %d of block %d, with value e=%f and idx_global=%d\n", threadIdx.x, blockIdx.x, e, idx_global);
}

int main(int argc, char **argv)
{
    helloCUDA<<<3, 4>>>(2.71828f);

    // Sin cudadevicereset
    cudaDeviceReset();
    system("pause");
    return (0);
}