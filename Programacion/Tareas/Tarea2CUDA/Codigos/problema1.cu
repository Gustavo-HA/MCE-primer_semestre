#include <stdlib.h> // malloc y rand
#include <iostream>
#include <chrono>

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
    float *v_h, *s1_h, *s2_h, *ver1_h, *ver2_h; // Host
    float *v_d, *s1_d, *s2_d;                   // Device
    long int i, N;
    int k, n_iteraciones = 50;

    // Medir tiempos en PARALELO
    float tiempo_paralelo = 0;
    cudaEvent_t inicio_p, fin_p; // Declaramos los eventos
    cudaEventCreate(&inicio_p);  // Los creamos
    cudaEventCreate(&fin_p);

    cout << "Tamano del vector: ";
    cin >> N;

    size_t size = N * sizeof(float),
           size_s1 = (N - 1) * sizeof(float), // Tamano s_1
        size_s2 = (N - 2) * sizeof(float);    // Tamano s_2

    // Alojamos memoria host
    v_h = (float *)malloc(size);
    s1_h = (float *)malloc(size_s1);
    ver1_h = (float *)malloc(size_s1);

    // Llenamos vector V
    for (i = 0; i < N; i++)
        v_h[i] = 3.14159;

    // Asignamos memoria en el device
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&v_d, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector V en el device\n");
        exit(EXIT_FAILURE);
    }

    // Pasamos el arreglo del host al device.
    cudaMemcpy(v_d, v_h, size, cudaMemcpyHostToDevice);

    // Parámetros para el lanzamiento de los kernel
    int block_size = 1024;
    int n_blocks = (N + block_size - 1) / block_size;

    // ------------ Inciso A -----------------
    cout << "\tINCISO A)\n";

    // ***************** SECUENCIAL ******************
    auto inicio = chrono::high_resolution_clock::now();

    for (k = 0; k < n_iteraciones; k++)
        inciso_a_s(v_h, s1_h, N);

    auto fin = chrono::high_resolution_clock::now();
    chrono::duration<double> duracion = (fin - inicio) / n_iteraciones;
    cout << "Tiempo en secuencial (segundos):\t" << duracion.count() << endl;

    // ************ PARALELO *****************
    // Asignamos memoria
    err = cudaMalloc((void **)&s1_d, size_s1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector S_1 en el device\n");
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(inicio_p);

    for (k = 0; k < n_iteraciones; k++)
        incisoA<<<n_blocks, block_size>>>(s1_d, v_d, N);

    cudaEventRecord(fin_p);

    cudaEventSynchronize(fin_p);
    cudaEventElapsedTime(&tiempo_paralelo, inicio_p, fin_p);
    tiempo_paralelo = tiempo_paralelo / (1000 * n_iteraciones); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos):\t\t" << tiempo_paralelo << "\t"
         << "Speedup: " << duracion.count() / tiempo_paralelo << endl;

    /*          VERIFICACION            */
    cudaMemcpy(ver1_h, s1_d, size_s1, cudaMemcpyDeviceToHost);

    for (i = 0; i < N - 1; i++)
    {
        if (fabs(ver1_h[i] - s1_h[i]) > 1e-3)
        {
            printf("Inciso A - Error en el indice %d\n", i);
            printf("%.5f != %.5f\n", ver1_h[i], s1_h[i]);
            exit(EXIT_FAILURE);
        }
    }

    cout << "Resultado verificado!\n";

    // Liberamos memoria
    cudaFree(s1_d);
    free(s1_h);
    free(ver1_h);

    // ------------ Inciso B -----------------
    cout << "\tINCISO B)\n";

    // *********** SECUENCIAL ****************
    s2_h = (float *)malloc(size_s2);

    inicio = chrono::high_resolution_clock::now();

    for (k = 0; k < n_iteraciones; k++)
        inciso_b_s(v_h, s2_h, N);

    fin = chrono::high_resolution_clock::now();
    duracion = (fin - inicio) / n_iteraciones;
    cout << "Tiempo en secuencial (segundos):\t" << duracion.count() << endl;

    // ************ PARALELO **************
    // Asignamos memoria
    err = cudaMalloc((void **)&s2_d, size_s2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al asignar memoria al vector S_2 en el device\n");
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(inicio_p);

    for (k = 0; k < n_iteraciones; k++)
        incisoB<<<n_blocks, block_size>>>(s2_d, v_d, N);

    cudaEventRecord(fin_p);

    cudaEventSynchronize(fin_p);
    cudaEventElapsedTime(&tiempo_paralelo, inicio_p, fin_p);
    tiempo_paralelo = tiempo_paralelo / (1000 * n_iteraciones); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos):\t\t" << tiempo_paralelo << "\t"
         << "Speedup: " << duracion.count() / tiempo_paralelo << endl;

    /*          VERIFICACION            */
    ver2_h = (float *)malloc(size_s2);
    cudaMemcpy(ver2_h, s2_d, size_s2, cudaMemcpyDeviceToHost);

    for (i = 0; i < N - 2; i++)
    {
        if (fabs(ver2_h[i] - s2_h[i]) > 1e-3)
        {
            printf("Inciso B - Error en el indice %d\n", i);
            printf("%.5f != %.5f\n", ver2_h[i], s2_h[i]);
            exit(EXIT_FAILURE);
        }
    }

    cout << "Resultado verificado!\n";

    // Liberamos memoria
    free(v_h);
    free(s2_h);
    free(ver2_h);
    cudaFree(v_d);
    cudaFree(s2_d);
    return (0);
}
