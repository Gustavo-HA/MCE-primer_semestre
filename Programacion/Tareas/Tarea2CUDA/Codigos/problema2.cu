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
    int id_B = (N - idy - 1) * M + (M - idx - 1);

    if (idx < M && idy < N)
        C[id] = A[id] + B[id_B];
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

    if (idx < M && idy < N)
        C[id] = a * A[id] + (1 - a) * B[id];
}

void incisoB_secuencial(unsigned int *A, unsigned int *B,
                        float *C, float a, long int N, long int M)
{
    int i, j, idx;

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
    int k, n_iteraciones = 10;
    unsigned int *A_h, *B_h, *C_1h, *verificacion_C1; // Host
    float *C_2h, *verificacion_C2;
    unsigned int *A_d, *B_d, *C_1d; // Device
    float *C_2d;

    // Medir tiempos - Paralelo
    cudaEvent_t tiempo1p, tiempo2p;
    float tiempo_paralelo = 0;
    cudaEventCreate(&tiempo1p);
    cudaEventCreate(&tiempo2p);

    // Inputs
    cout << "Tamano de las matrices NxM: ";
    cin >> N;
    cin >> M;
    cout << "Valor de alfa: ";
    cin >> a;
    size_t size = N * M * sizeof(unsigned int), size_c2 = N * M * sizeof(float);

    // Asignamos memoria en el host
    A_h = (unsigned int *)malloc(size);
    if (A_h == NULL)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector A en el host.\n");
        exit(EXIT_FAILURE);
    }
    B_h = (unsigned int *)malloc(size);
    if (B_h == NULL)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector B en el host.\n");
        exit(EXIT_FAILURE);
    }

    verificacion_C1 = (unsigned int *)malloc(size);
    if (verificacion_C1 == NULL)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector ver1 en el host.\n");
        exit(EXIT_FAILURE);
    }

    verificacion_C2 = (float *)malloc(size_c2);
    if (verificacion_C2 == NULL)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector ver1 en el host.\n");
        exit(EXIT_FAILURE);
    }

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
            A_h[i * M + j] = 2.718281;
            B_h[i * M + j] = 3.14159;
        }
    }

    // Copiamos las matrices llenadas al device.
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Configuración del lanzamiento de kernels.
    int BLOCK_SIZE = 32;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks((M + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    // ----------- INCISO A) -----------------------
    cout << "\tInciso A):\n";

    // ************** SECUENCIAL ***********************

    C_1h = (unsigned int *)malloc(size);
    if (C_1h == NULL)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector C_1 en el host.\n");
        exit(EXIT_FAILURE);
    }

    auto inicio = std::chrono::high_resolution_clock::now();

    for (k = 0; k < n_iteraciones; k++)
        incisoA_secuencial(A_h, B_h, C_1h, N, M);

    auto final = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> tiempo_secuencial =
        (final - inicio) / n_iteraciones;
    cout << "Tiempo en secuencial (segundos):\t"
         << tiempo_secuencial.count() << endl;

    // ************** PARALELO ***********************

    err = cudaMalloc((void **)&C_1d, size); // Asignamos memoria para el inciso A
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector C_1 en el device\n");
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(tiempo1p);

    for (k = 0; k < n_iteraciones; k++)
        incisoA<<<n_blocks, block_size>>>(A_d, B_d, C_1d, N, M);

    cudaEventRecord(tiempo2p);

    cudaEventSynchronize(tiempo2p);
    cudaEventElapsedTime(&tiempo_paralelo, tiempo1p, tiempo2p);
    tiempo_paralelo = tiempo_paralelo / (1000 * n_iteraciones); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos):\t\t" << tiempo_paralelo << "\t" << "Speedup: "
         << tiempo_secuencial.count() / tiempo_paralelo << endl;

    /*          VERIFICACION            */
    cudaMemcpy(verificacion_C1, C_1d, size, cudaMemcpyDeviceToHost);

    for (i = 0; i < N * M; i++)
    {
        if (fabs(verificacion_C1[i] - C_1h[i]) > 1e-3)
        {
            printf("Inciso A - Error en el indice %d\n", i);
            printf("%d != %d\n", verificacion_C1[i], C_1h[i]);
            exit(EXIT_FAILURE);
        }
    }

    cout << "Resultado verificado!\n";

    free(verificacion_C1);
    cudaFree(C_1d); // Ya no necesitamos C_1d

    // -------------- INCISO B) -----------------------

    // ************** SECUENCIAL ***********************
    C_2h = (float *)malloc(size_c2);
    if (C_2h == NULL)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector C_2 en el host.\n");
        exit(EXIT_FAILURE);
    }

    cout << "\tInciso B)\n";

    inicio = std::chrono::high_resolution_clock::now();

    for (k = 0; k < n_iteraciones; k++)
        incisoB_secuencial(A_h, B_h, C_2h, a, N, M);

    final = std::chrono::high_resolution_clock::now();
    tiempo_secuencial = (final - inicio) / n_iteraciones;
    cout << "Tiempo en secuencial (segundos):\t" << tiempo_secuencial.count() << endl;

    // *************** PARALELO **************************
    err = cudaMalloc((void **)&C_2d, size_c2); // Asignamos memoria para el inciso B.
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector C_2 en el device\n");
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(tiempo1p);

    for (k = 0; k < n_iteraciones; k++)
        incisoB<<<n_blocks, block_size>>>(A_d, B_d, C_2d, a, N, M);

    cudaEventRecord(tiempo2p);

    cudaEventSynchronize(tiempo2p);
    cudaEventElapsedTime(&tiempo_paralelo, tiempo1p, tiempo2p);
    tiempo_paralelo = tiempo_paralelo / (1000 * n_iteraciones); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos):\t\t" << tiempo_paralelo << "\t"
         << "Speedup: " << tiempo_secuencial.count() / tiempo_paralelo << endl;

    /*          VERIFICACION            */
    cudaMemcpy(verificacion_C2, C_2d, size_c2, cudaMemcpyDeviceToHost);

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            int index = i * M + j;
            if (fabs(verificacion_C2[index] - C_2h[index]) > (float)1e-3)
            {
                printf("Inciso B - Error en el indice (%d,%d)\n", i, j);
                printf("%.5f != %.5f\n", verificacion_C2[index], C_2h[index]);
                exit(EXIT_FAILURE);
            }
        }
    }

    cout << "Resultado verificado!\n";

    free(verificacion_C2);
    cudaFree(C_2d);

    //*************
    // Se libera la memoria.
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
