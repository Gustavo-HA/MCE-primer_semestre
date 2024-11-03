#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

using namespace std;

void leer(unsigned int *A, long int N, long int M);
void mostrar(float *A, long int N, long int M);

void incisoA_paralelo(unsigned int *A, unsigned int *B, float *C, long int N, long int M);
void incisoA_secuencial(unsigned int *A, unsigned int *B, float *C, long int N, long int M);

void incisoB_paralelo(unsigned int *A, unsigned int *B, float *C, float a, long int N, long int M);
void incisoB_secuencial(unsigned int *A, unsigned int *B, float *C, float a, long int N, long int M);

int main()
{
    long int N, M;
    unsigned int *A, *B;
    float *C1, *C2;
    float a;

    /* Input manual.
    cout << "N x M: ";
    cin >> N;
    cin >> M;

    A = (unsigned int *)malloc(N * M * sizeof(unsigned int));
    B = (unsigned int *)malloc(N * M * sizeof(unsigned int));
    C1 = (float *)malloc(N * M * sizeof(float));
    C2 = (float *)malloc(N * M * sizeof(float));

    cout << "Matriz A:\n";
    leer(A, N, M);

    cout << "Matriz B:\n";
    leer(B, N, M);

    cout << "Alfa: ";
    cin >> a;
    */

    /* Input automático. */
    clock_t inicio, final;
    double tiempo;
    N = M = 10000;
    A = (unsigned int *)malloc(N * M * sizeof(unsigned int));
    B = (unsigned int *)malloc(N * M * sizeof(unsigned int));
    C1 = (float *)malloc(N * M * sizeof(float));
    C2 = (float *)malloc(N * M * sizeof(float));

    /* INCISO A)  */
    // en paralelo.
    cout << "\tINCISO A)\n";
    inicio = clock();
    incisoA_paralelo(A, B, C1, N, M);
    final = clock();
    tiempo = ((double)(final - inicio)) / CLOCKS_PER_SEC;
    cout << "Tiempo en paralelo: " << tiempo << " segundos.\n";

    // en secuencial.
    inicio = clock();
    incisoA_secuencial(A, B, C1, N, M);
    final = clock();
    tiempo = ((double)(final - inicio)) / CLOCKS_PER_SEC;
    cout << "Tiempo en secuencial: " << tiempo << " segundos.\n";

    /* INCISO B)  */
    cout << "\tINCISO B)\n";
    // en paralelo.
    inicio = clock();
    incisoB_paralelo(A, B, C2, a, N, M);
    final = clock();
    tiempo = ((double)(final - inicio)) / CLOCKS_PER_SEC;
    cout << "Tiempo en paralelo: " << tiempo << " segundos.\n";

    // en secuencial.
    inicio = clock();
    incisoB_secuencial(A, B, C2, a, N, M);
    final = clock();
    tiempo = ((double)(final - inicio)) / CLOCKS_PER_SEC;
    cout << "Tiempo en secuencial: " << tiempo << " segundos.\n";

    return 0;
}

void leer(unsigned int *A, long int N, long int M)
{
    long int i;
    for (i = 0; i < N * M; i++)
    {
        cin >> A[i];
    }
}

void mostrar(float *A, long int N, long int M)
{
    long int i;
    for (i = 0; i < N * M; i++)
    {
        if (i % M == 0 & i != 0)
            cout << endl;
        cout << A[i] << "\t";
    }
    cout << endl;
}

void incisoA_paralelo(unsigned int *A, unsigned int *B, float *C, long int N, long int M)
{
    long int i, j, idx, idx_b;

#pragma omp parallel for collapse(2) default(none) shared(A, B, C, N, M) private(i, j, idx, idx_b)
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

void incisoA_secuencial(unsigned int *A, unsigned int *B, float *C, long int N, long int M)
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

void incisoB_paralelo(unsigned int *A, unsigned int *B, float *C, float a, long int N, long int M)
{
    long int i, j, idx;

#pragma omp parallel for default(none) shared(A, B, C, N, M, a) private(i, j, idx)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            idx = i * M + j;
            C[idx] = a * A[idx] + (1 - a) * B[idx];
        }
    }
}

void incisoB_secuencial(unsigned int *A, unsigned int *B, float *C, float a, long int N, long int M)
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