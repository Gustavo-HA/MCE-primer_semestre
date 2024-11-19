#include <stdlib.h>
#include <iostream>
#include <chrono>

using namespace std;

__global__ void incisoA(unsigned int *A, unsigned int *B,
                        unsigned int *C, long int N, long int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * M + idx;
    int id_B = (N - idx - 1) + (M - idy - 1) * M;

    if (idx < N && idy < M)
    {
        C[id] = A[id] + B[id_B];
    }
}

void incisoA_secuencial(unsigned int *A, unsigned int *B,
                        unsigned int *C, long int N, long int M)
{
    long int i, j, idx, idx_b;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            idx = i * M + j;
            idx_b = (N - i - 1) * M + (M - j - 1);
            C[idx] = A[idx] + B[idx_b];
        }
    }
}

__global__ void incisoB(unsigned int *A, unsigned int *B,
                        float *C, float a, long int N, long int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * M + idx;

    if (idx < N && idy < M)
    {
        C[id] = a * A[id] + (1 - a) * B[id];
    }
}

void incisoB_secuencial(unsigned int *A, unsigned int *B,
                        float *C, float a, long int N, long int M)
{
    long int i, j, idx;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            idx = i * M + j;
            C[idx] = a * A[idx] + (1 - a) * B[idx];
        }
    }
}

int main()
{
    long int N, M, i, j;
    float a;
    unsigned int *A_h, *B_h, *C_1h; // Host
    float *C_2h;
    unsigned int *A_d, *B_d, *C_1d; // Device
    float *C_2d;
    cudaEvent_t tiempo1p, tiempo2p;
    float tiempo_paralelo = 0;

    cout << "Tamano de las matrices NxM: ";
    cin >> N;
    cin >> M;
    cout << "Valor de alfa: ";
    cin >> a;

    size_t size = N * M * sizeof(unsigned int), size_c2 = N * M * sizeof(float);

    // Asignamos memoria en el host
    A_h = (unsigned int *)malloc(size);
    B_h = (unsigned int *)malloc(size);
    C_1h = (unsigned int *)malloc(size);
    C_2h = (float *)malloc(size_c2);

    // Asignamos memoria en el device
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&A_d, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector A en el device\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&B_d, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector B en el device\n");
        exit(EXIT_FAILURE);
    }

    // Asignamos valores a matrices del host
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            A_h[i * M + j] = i * j;
            B_h[i * M + j] = i + j;
        }
    }

    // Copiamos las matrices llenadas al device.
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Configuración del lanzamiento de kernels.
    int BLOCK_SIZE = 16;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks((M - 1 + block_size.x) / block_size.x, (N - 1 + block_size.y) / block_size.y);

    // ----------- INCISO A) -----------------------
    cout << "\tInciso A):\n";
    err = cudaMalloc((void **)&C_1d, size); // Asignamos memoria para el inciso A
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector C_1 en el device\n");
        exit(EXIT_FAILURE);
    }
    //             PARALELO
    // Sin tomar tiempo de transferencia de memoria.
    cudaEventCreate(&tiempo1p);
    cudaEventCreate(&tiempo2p);
    cudaEventRecord(tiempo1p);
    incisoA<<<n_blocks, block_size>>>(A_d, B_d, C_1d, N, M);
    cudaEventRecord(tiempo2p);

    cudaEventSynchronize(tiempo2p);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&tiempo_paralelo, tiempo1p, tiempo2p);
    tiempo_paralelo = tiempo_paralelo / (1000); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos):\t\t" << tiempo_paralelo << endl;

    // Tomando tiempo de transferencia de memoria.
    cudaEventCreate(&tiempo1p);
    cudaEventCreate(&tiempo2p);
    cudaEventRecord(tiempo1p);
    incisoA<<<n_blocks, block_size>>>(A_d, B_d, C_1d, N, M);
    cudaMemcpy(C_1h, C_1d, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(tiempo2p);

    cudaEventSynchronize(tiempo2p);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&tiempo_paralelo, tiempo1p, tiempo2p);
    tiempo_paralelo = tiempo_paralelo / (1000); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos) con TM:\t" << tiempo_paralelo << endl;

    cudaFree(C_1d); // Ya no necesitamos C_1d

    //          SECUENCIAL
    clock_t inicio, final;
    double tiempo;
    inicio = clock();
    incisoA_secuencial(A_h, B_h, C_1h, N, M);
    final = clock();
    tiempo = ((double)(final - inicio)) / CLOCKS_PER_SEC;
    cout << "Tiempo en secuencial (segundos):\t" << tiempo << endl;

    // -------------- INCISO B) -----------------------
    err = cudaMalloc((void **)&C_2d, size_c2); // Asignamos memoria para el inciso B.
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector C_2 en el device\n");
        exit(EXIT_FAILURE);
    }

    cout << "\tInciso B)\n";
    //          PARALELO
    // Sin tomar tiempo de transferencia de memoria.
    cudaEventRecord(tiempo1p);
    incisoB<<<n_blocks, block_size>>>(A_d, B_d, C_2d, a, N, M);
    cudaEventRecord(tiempo2p);

    cudaEventSynchronize(tiempo2p);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&tiempo_paralelo, tiempo1p, tiempo2p);
    tiempo_paralelo = tiempo_paralelo / (1000); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos):\t\t" << tiempo_paralelo << endl;

    // Tomando tiempo de transferencia de memoria.
    cudaEventRecord(tiempo1p);
    incisoB<<<n_blocks, block_size>>>(A_d, B_d, C_2d, a, N, M);
    cudaMemcpy(C_2h, C_2d, size_c2, cudaMemcpyDeviceToHost);
    cudaEventRecord(tiempo2p);

    cudaEventSynchronize(tiempo2p);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&tiempo_paralelo, tiempo1p, tiempo2p);
    tiempo_paralelo = tiempo_paralelo / (1000); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos) con TM:\t" << tiempo_paralelo << endl;

    cudaFree(C_2d); // Se libera la memoria.

    //          SECUENCIAL
    inicio = clock();
    incisoB_secuencial(A_h, B_h, C_2h, a, N, M);
    final = clock();
    tiempo = ((double)(final - inicio)) / CLOCKS_PER_SEC;
    cout << "Tiempo en secuencial (segundos):\t" << tiempo << endl;

    cudaFree(A_d);
    cudaFree(B_d);
    free(A_h);
    free(B_h);
    free(C_1h);
    free(C_2h);
    cudaEventDestroy(tiempo1p);
    cudaEventDestroy(tiempo2p);
    return 0;
}
