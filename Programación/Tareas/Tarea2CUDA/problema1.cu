#include <stdio.h>
#include <stdlib.h> // malloc y rand
#include <iostream>
#include <chrono>
#include <time.h>
#include <cuda_runtime.h> // funciones cuda

using namespace std;

void inciso_a_s(float *V, float *A, long int N)
{
    long int i;
    for (i = 0; i < N - 1; i++)
    {
        A[i] = V[i] + V[i + 1];
    }
}

void inciso_b_s(float *V, float *A, long int N)
{
    long int i;
    for (i = 1; i < N - 1; i++)
    {
        A[i - 1] = (V[i - 1] + V[i + 1]) / 2;
    }
}

__global__ void incisoA(float *s_d, float *v_d, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N - 1)
    {
        s_d[idx] = v_d[idx] + v_d[idx + 1];
    }
}

__global__ void incisoB(float *s_d, float *v_d, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx > 0 && idx < N - 1)
    {
        s_d[idx - 1] = (v_d[idx + 1] + v_d[idx - 1]) / 2;
    }
}

int main()
{
    float *v_h, *s1_h, *s2_h; // Host
    float *v_d, *s1_d, *s2_d; // Device
    long int i, N;

    cout << "Tamano del vector: ";
    cin >> N;

    size_t size = N * sizeof(float), size_s1 = (N - 1) * sizeof(float), size_s2 = (N - 2) * sizeof(float);

    // Alojamos memoria host
    v_h = (float *)malloc(size);
    s1_h = (float *)malloc(size_s1); // Tamano N-1
    s2_h = (float *)malloc(size_s2); // Tamano N-2

    // Llenamos vector V
    for (i = 0; i < N; i++)
        v_h[i] = i * 3.14159;

    // Alojamos memoria en el device
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&v_d, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector A en el device\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&s1_d, size_s1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector S_1 en el device\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&s2_d, size_s2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector S_2 en el device\n");
        exit(EXIT_FAILURE);
    }

    // Pasamos el arreglo del host al device.
    cudaMemcpy(v_d, v_h, size, cudaMemcpyHostToDevice);

    // Parámetros para el lanzamiento de los kernel
    int block_size = 32;
    int n_blocks = N / block_size + (N % block_size == 0 ? 0 : 1);

    // ------------ Inciso A -----------------
    cout << "\tINCISO A)\n";

    // PARALELO
    float tiempo_paralelo = 0; // Medir tiempo de ejecucion de Kernels CUDA
    cudaEvent_t inicio_p, fin_p;
    cudaEventCreate(&inicio_p);
    cudaEventCreate(&fin_p);

    // PARALELO
    // Tiempo de ejecucion sin transferencia de memoria.
    cudaEventRecord(inicio_p);
    incisoA<<<n_blocks, block_size>>>(s1_d, v_d, N);
    cudaEventRecord(fin_p);

    cudaEventSynchronize(fin_p);
    cudaEventElapsedTime(&tiempo_paralelo, inicio_p, fin_p);
    tiempo_paralelo = tiempo_paralelo / (1000); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos):\t\t" << tiempo_paralelo << endl;

    // Tiempo de ejecucion con transferencia de memoria.
    cudaEventRecord(inicio_p);
    incisoA<<<n_blocks, block_size>>>(s1_d, v_d, N);
    cudaMemcpy(s1_h, s1_d, size_s1, cudaMemcpyDeviceToHost); // Pasar el array del device al host.
    cudaEventRecord(fin_p);

    cudaEventSynchronize(fin_p);
    cudaEventElapsedTime(&tiempo_paralelo, inicio_p, fin_p);
    tiempo_paralelo = tiempo_paralelo / (1000); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos) con TM:\t" << tiempo_paralelo << endl;

    // SECUENCIAL
    auto inicio = chrono::high_resolution_clock::now();
    inciso_a_s(v_h, s1_h, N);
    auto fin = chrono::high_resolution_clock::now();
    chrono::duration<double> duracion = fin - inicio;
    cout << "Tiempo en secuencial (segundos):\t" << duracion.count() << endl;

    // ------------ Inciso B -----------------
    cout << "\tINCISO B)\n";

    // PARALELO
    // Sin transferencia de memoria.
    cudaEventRecord(inicio_p);
    incisoB<<<n_blocks, block_size>>>(s2_d, v_d, N);
    cudaEventRecord(fin_p);

    cudaEventSynchronize(fin_p);
    cudaEventElapsedTime(&tiempo_paralelo, inicio_p, fin_p);
    tiempo_paralelo = tiempo_paralelo / (1000); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos):\t\t" << tiempo_paralelo << endl;

    // Con transferencia de memoria.
    cudaEventRecord(inicio_p);
    incisoB<<<n_blocks, block_size>>>(s2_d, v_d, N);
    cudaMemcpy(s2_h, s2_d, size_s2, cudaMemcpyDeviceToHost); // Pasar el array del device al host.
    cudaEventRecord(fin_p);

    cudaEventSynchronize(fin_p);
    cudaEventElapsedTime(&tiempo_paralelo, inicio_p, fin_p);
    cudaEventDestroy(inicio_p); // Ya no necesitamos los eventos.
    cudaEventDestroy(fin_p);
    tiempo_paralelo = tiempo_paralelo / (1000); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos) con TM:\t" << tiempo_paralelo << endl;

    // SECUENCIAL
    inicio = chrono::high_resolution_clock::now();
    inciso_b_s(v_h, s2_h, N);
    fin = chrono::high_resolution_clock::now();
    duracion = fin - inicio;
    cout << "Tiempo en secuencial (segundos):\t" << duracion.count() << endl;

    free(v_h);
    free(s1_h);
    free(s2_h);
    cudaFree(v_d);
    cudaFree(s1_d);
    cudaFree(s2_d);
    return (0);
}
