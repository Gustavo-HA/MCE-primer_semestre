#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

void mostrar_vector(float *V, long int N);

void inciso_a(float *V, float *A, long int N);

void inciso_b(float *V, float *B, long int N);

int main()
{
    long int N, i;
    float *V, *A, *B;

    cout << "N: ";
    cin >> N;

    // Asignamos memoria
    V = (float *)malloc(N * sizeof(float));
    A = (float *)malloc((N - 1) * sizeof(float));
    B = (float *)malloc((N - 2) * sizeof(float));

    cout << "Vector V:\n";
    for (i = 0; i < N; i++)
    {
        cin >> V[i];
    }

    // Inciso A.
    inciso_a(V, A, N);
    cout << "Inciso A:\n";
    mostrar_vector(A, N - 1);

    // Inciso B.
    inciso_b(V, B, N);
    cout << "Inciso B:\n";
    mostrar_vector(B, N - 2);

    free(V);
    free(A);
    free(B);
    return 0;
}

void inciso_a(float *V, float *A, long int N)
{
    long int i;

#pragma omp parallel for default(none) shared(V, A, N) private(i)
    for (i = 0; i < N - 1; i++)
    {
        A[i] = V[i] + V[i + 1];
    }
}

void inciso_b(float *V, float *A, long int N)
{
    long int i;
#pragma omp parallel for default(none) shared(V, A, N) private(i)
    for (i = 1; i < N - 1; i++)
    {
        A[i - 1] = (V[i - 1] + V[i + 1]) / 2;
    }
}

void mostrar_vector(float *A, long int N)
{
    long int i;
    for (i = 0; i < N; i++)
    {
        cout << A[i] << " ";
    }
    cout << "\n";
}