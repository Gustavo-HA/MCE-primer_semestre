#include <stdlib.h>
#include <chrono>
#include <iostream>
#define BLOCK_SIZE 32

using namespace std;

// Inciso A
__global__ void mult_matrices_GM(double *A, double *B, double *C,
                                 long int N, long int K, long int M)
{
    long int idx, idy, k, index;
    idx = blockDim.x * blockIdx.x + threadIdx.x;
    idy = blockDim.y * blockIdx.y + threadIdx.y;
    index = idy * M + idx;
    if (idy < N && idx < M)
    {
        double sum = 0.0;
        for (k = 0; k < K; k++)
        {
            sum += A[idy * K + k] * B[k * M + idx];
        }
        C[index] = sum;
    }
}

// Inciso B
__global__ void mult_matrices_SM(double *A, double *B, double *C,
                                 long int N, long int K, long int M)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Indices de inicio, final de la submatriz de A
    // Pasos de la iteración
    int aBegin = K * BLOCK_SIZE * by;
    int aEnd = aBegin + K - 1;
    int aStep = BLOCK_SIZE;

    // Indice de inicio de la submatriz de B
    // junto a sus pasos
    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * M;

    // Valor en cada posición
    double Csub = 0;

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {
        // Se utiliza la memoria compartida
        // para guardar las respectivas
        // submatrices de A y B.
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + K * ty + tx];
        Bs[ty][tx] = B[b + M * ty + tx];

        __syncthreads();

#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = M * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + M * ty + tx] = Csub;
}

void mult_matrices(double *A, double *B, double *C,
                   long int N, long int K, long int M)
{
    long int i, j, k;
    long int idx_a, idx_b, idx_c;
    double sum;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            sum = 0.0;
            for (k = 0; k < K; k++)
            {
                idx_a = i * K + k;
                idx_b = k * M + j;
                sum += A[idx_a] * B[idx_b];
            }
            idx_c = i * M + j;
            C[idx_c] = sum;
        }
    }
}

int main()
{
    long int N, K, M, i, j;
    double *d_A, *d_B, *d_C; // Variables device
    double *h_A, *h_B, *h_C; // Variables host
    double *verificar_C;
    int n_iteraciones = 5, k;

    // Medir tiempos - Paralelo
    cudaEvent_t d_inicio, d_final;
    float tiempo_paralelo = 0;
    cudaEventCreate(&d_inicio);
    cudaEventCreate(&d_final);

    // Inputs
    cout << "Tamano de las matrices 'N K M': ";
    cin >> N;
    cin >> K;
    cin >> M;
    size_t sizeA = N * K * sizeof(double);
    size_t sizeB = K * M * sizeof(double);
    size_t sizeC = N * M * sizeof(double);

    // Asignamos memoria en el host
    h_A = (double *)malloc(sizeA);
    if (h_A == NULL)
    {
        fprintf(stderr, "Fallo al asignar memoria a la matriz A en el host.\n");
        exit(EXIT_FAILURE);
    }
    h_B = (double *)malloc(sizeB);
    if (h_B == NULL)
    {
        fprintf(stderr, "Fallo al asignar memoria a la matriz B en el host.\n");
        exit(EXIT_FAILURE);
    }
    h_C = (double *)malloc(sizeC);
    if (h_C == NULL)
    {
        fprintf(stderr, "Fallo al asignar memoria a la matriz C_1 en el host.\n");
        exit(EXIT_FAILURE);
    }
    verificar_C = (double *)malloc(sizeC);
    if (verificar_C == NULL)
    {
        fprintf(stderr, "Fallo al asignar memoria a la matriz ver. en el host.\n");
        exit(EXIT_FAILURE);
    }

    // Asignamos memoria en el device
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&d_A, sizeA);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria a la matriz A en el device.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_B, sizeB);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria a la matriz B en el device.\n");
        exit(EXIT_FAILURE);
    }

    // Inicializamos las matrices A y B.
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < K; j++)
            h_A[i * K + j] = rand() / (double)RAND_MAX;
    }
    for (i = 0; i < K; i++)
    {
        for (j = 0; j < M; j++)
            h_B[i * M + j] = rand() / (double)RAND_MAX;
    }

    // Copiamos las matrices llenadas al device.
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Configuracion para los lanzamientos de kernels.
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks((M + block_size.x - 1) / block_size.x,
                  (N + block_size.y - 1) / block_size.y);

    // ************** SECUENCIAL ***************
    auto inicio = std::chrono::high_resolution_clock::now();

    for (k = 0; k < n_iteraciones; k++)
        mult_matrices(h_A, h_B, h_C, N, K, M);

    auto final = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> tiempo_secuencial = (final - inicio) / n_iteraciones;
    cout << "Tiempo en secuencial (segundos):\t" << tiempo_secuencial.count() << endl;

    // ************** PARALELO GM **************
    err = cudaMalloc((void **)&d_C, sizeC);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria a la matriz C en el device\n");
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(d_inicio);

    for (k = 0; k < n_iteraciones; k++)
        mult_matrices_GM<<<n_blocks, block_size>>>(d_A, d_B, d_C, N, K, M);

    cudaEventRecord(d_final);

    cudaEventSynchronize(d_final);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaEventElapsedTime(&tiempo_paralelo, d_inicio, d_final);
    tiempo_paralelo = tiempo_paralelo / (1000 * n_iteraciones); // Milisegundos a segundos.
    cout << "Tiempo en paralelo GM (segundos):\t" << tiempo_paralelo << "\t"
         << "Speedup: " << tiempo_secuencial.count() / tiempo_paralelo << endl;

    /*      VERIFICACIÓN        */
    cudaMemcpy(verificar_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            k = i * M + j;
            if (fabs(verificar_C[k] - h_C[k]) > 1e-5)
            {
                printf("GM: Error en la posicion (%d,%d)\n", (int)i, (int)j);
                printf("%.5f != %.5f\n", verificar_C[k], h_C[k]);
                exit(EXIT_FAILURE);
            }
        }
    }

    // *************** PARALELO SM **************
    cudaEventRecord(d_inicio);

    for (k = 0; k < n_iteraciones; k++)
        mult_matrices_SM<<<n_blocks, block_size>>>(d_A, d_B, d_C, N, K, M);

    cudaEventRecord(d_final);

    cudaEventSynchronize(d_final);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaEventElapsedTime(&tiempo_paralelo, d_inicio, d_final);
    tiempo_paralelo = tiempo_paralelo / (1000 * n_iteraciones); // Milisegundos a segundos.
    cout << "Tiempo en paralelo SM (segundos):\t" << tiempo_paralelo << "\t"
         << "Speedup: " << tiempo_secuencial.count() / tiempo_paralelo << endl;

    /*      VERIFICACIÓN        */
    cudaMemcpy(verificar_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            k = i * M + j;
            if (fabs(verificar_C[k] - h_C[k]) > 1e-3)
            {
                printf("SM: Error en la posicion (%d,%d)\n", (int)i, (int)j);
                printf("%.5f != %.5f\n", verificar_C[k], h_C[k]);
                exit(EXIT_FAILURE);
            }
        }
    }

    cout << "Resultado verificado!\n";

    // Se libera la memoria
    cudaFree(d_C);
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_C);
    free(h_A);
    free(h_B);
    free(verificar_C);
    cudaEventDestroy(d_inicio);
    cudaEventDestroy(d_final);
}