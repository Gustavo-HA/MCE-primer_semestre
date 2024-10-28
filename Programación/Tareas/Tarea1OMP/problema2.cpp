#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

void leer_matriz(float *M, long int N);
void multiplicacion_matriz(float *A, float *B, float *C, long int N);
void mostrar_matriz(float *C, long int N);

int main()
{
    float *A, *B, *C;
    long int i, N;

    cout << "N: ";
    cin >> N;

    // Asignamos la memoria a cada matriz.
    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N * sizeof(float));
    C = (float *)malloc(N * N * sizeof(float));

    cout << "Leemos la matriz A:\n";
    leer_matriz(A, N);

    cout << "Leemos la matriz B:\n";
    leer_matriz(B, N);

    multiplicacion_matriz(A, B, C, N);

    cout << "Resultado:";
    mostrar_matriz(C, N);
    cout << endl;

    free(A);
    free(B);
    free(C);
    return 0;
}

void leer_matriz(float *M, long int N)
{
    long int i;
    for (i = 0; i < N * N; i++)
    {
        cin >> M[i];
    }
}

void multiplicacion_matriz(float *A, float *B, float *C, long int N)
{
    long int i, j, k, indiceA, indiceB, indiceC;

#pragma omp parallel for default(none) shared(A, B, C, N) private(i, j, k, indiceA, indiceB, indiceC)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            indiceC = i * N + j;
            C[indiceC] = 0;
            for (k = 0; k < N; k++)
            {
                indiceA = i * N + k;
                indiceB = j + N * k;
                C[indiceC] += A[indiceA] * B[indiceB];
            }
        }
    }
}

void mostrar_matriz(float *M, long int N)
{
    long int i;

    for (i = 0; i < N * N; i++)
    {
        if (i % N == 0)
            cout << endl;
        cout << M[i] << "\t";
    }
    cout << endl;
}