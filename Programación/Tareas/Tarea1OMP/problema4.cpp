#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

void leer(float *A, long int N, long int M);
void mostrar(float *A, long int N, long int M);
void incisoA(float *A, float *B, float *C, long int N, long int M);
void incisoB(float *A, float *B, float *C, float a, long int N, long int M);

int main()
{
    long int N, M;
    float *A, *B, *C1, *C2;
    float a;

    cout << "N x M: ";
    cin >> N;
    cin >> M;

    A = (float *)malloc(N * M * sizeof(float));
    B = (float *)malloc(N * M * sizeof(float));
    C1 = (float *)malloc(N * M * sizeof(float));
    C2 = (float *)malloc(N * M * sizeof(float));

    cout << "Matriz A:\n";
    leer(A, N, M);

    cout << "Matriz B:\n";
    leer(B, N, M);

    cout << "Alfa: ";
    cin >> a;

    // Inciso A.
    incisoA(A, B, C1, N, M);
    cout << "Inciso a):\n";
    mostrar(C1, N, M);

    // Inciso B.
    incisoB(A, B, C2, a, N, M);
    cout << "Inciso b):\n";
    mostrar(C2, N, M);
}

void leer(float *A, long int N, long int M)
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
        cout << A[i] << " ";
    }
    cout << endl;
}

void incisoA(float *A, float *B, float *C, long int N, long int M)
{
    long int i, j, idx, idx_b;

#pragma omp parallel for default(none) shared(A, B, C, N, M) private(i, j, idx, idx_b)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            idx = i * N + j;
            idx_b = (N - i - 1) * N + (M - j - 1);
            C[idx] = A[idx] + B[idx_b];
        }
    }
}

void incisoB(float *A, float *B, float *C, float a, long int N, long int M)
{
    long int i, j, idx;

#pragma omp parallel for default(none) shared(A, B, C, N, M, a) private(i, j, idx)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            idx = i * N + j;
            C[idx] = a * A[idx] + (1 - a) * B[idx];
        }
    }
}