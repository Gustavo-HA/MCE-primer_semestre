#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

void leer(unsigned int *A, long int N, long int M);
void mostrar(unsigned int *A, long int N, long int M);
void incisoA(unsigned int *A, unsigned int *B, unsigned int *C, long int N, long int M);
void incisoB(unsigned int *A, unsigned int *B, unsigned int *C, float a, long int N, long int M);

int main()
{
    long int N, M;
    unsigned int *A, *B, *C1, *C2;
    float a;

    cout << "N x M: ";
    cin >> N;
    cin >> M;

    A = (unsigned int *)malloc(N * M * sizeof(unsigned int));
    B = (unsigned int *)malloc(N * M * sizeof(unsigned int));
    C1 = (unsigned int *)malloc(N * M * sizeof(unsigned int));
    C2 = (unsigned int *)malloc(N * M * sizeof(unsigned int));

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

void leer(unsigned int *A, long int N, long int M)
{
    long int i;
    for (i = 0; i < N * M; i++)
    {
        cin >> A[i];
    }
}

void mostrar(unsigned int *A, long int N, long int M)
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

void incisoA(unsigned int *A, unsigned int *B, unsigned int *C, long int N, long int M)
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

void incisoB(unsigned int *A, unsigned int *B, unsigned int *C, float a, long int N, long int M)
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