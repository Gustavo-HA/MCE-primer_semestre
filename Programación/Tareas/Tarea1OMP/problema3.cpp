#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

using namespace std;

void mostrar_vector(float *V, long int N);

void inciso_a_p(float *V, float *A, long int N);
void inciso_a_s(float *V, float *A, long int N);

void inciso_b_p(float *V, float *B, long int N);
void inciso_b_s(float *V, float *B, long int N);

int main()
{
    long int N, i;
    float *V, *A, *B;

    /* Llenado manual
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
    */

    /* Llenado automático */
    N = 10'000'000;
    V = (float *)malloc(N * sizeof(float));
    A = (float *)malloc((N - 1) * sizeof(float));
    B = (float *)malloc((N - 2) * sizeof(float));

    for (i = 0; i < N; i++)
    {
        V[i] = (float)(i);
    }

    clock_t inicio, final;
    double tiempo;

    /*          INCISO A)           */
    cout << "\tINCISO A)\n";
    // Paralelo
    inicio = clock();
    inciso_a_p(V, A, N);
    final = clock();
    tiempo = ((double)(final - inicio)) / CLOCKS_PER_SEC;
    cout << "Tiempo en paralelo: " << tiempo << " segundos." << endl;

    // Secuencial
    inicio = clock();
    inciso_a_s(V, A, N);
    final = clock();
    tiempo = ((double)(final - inicio)) / CLOCKS_PER_SEC;
    cout << "Tiempo en secuencial: " << tiempo << " segundos." << endl;

    /*          INCISO B)           */
    cout << "\tINCISO B)\n";
    // Paralelo
    inicio = clock();
    inciso_b_p(V, B, N);
    final = clock();
    tiempo = ((double)(final - inicio)) / CLOCKS_PER_SEC;
    cout << "Tiempo en paralelo: " << tiempo << " segundos." << endl;

    // Secuencial
    inicio = clock();
    inciso_b_s(V, B, N);
    final = clock();
    tiempo = ((double)(final - inicio)) / CLOCKS_PER_SEC;
    cout << "Tiempo en secuencial: " << tiempo << " segundos." << endl;

    free(V);
    free(A);
    free(B);
    return 0;
}

void inciso_a_p(float *V, float *A, long int N)
{
    long int i;

#pragma omp parallel for default(none) shared(V, A, N) private(i)
    for (i = 0; i < N - 1; i++)
    {
        A[i] = V[i] + V[i + 1];
    }
}

void inciso_a_s(float *V, float *A, long int N)
{
    long int i;
    for (i = 0; i < N - 1; i++)
    {
        A[i] = V[i] + V[i + 1];
    }
}

void inciso_b_p(float *V, float *A, long int N)
{
    long int i;
#pragma omp parallel for default(none) shared(V, A, N) private(i)
    for (i = 1; i < N - 1; i++)
    {
        A[i - 1] = (V[i - 1] + V[i + 1]) / 2;
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

void mostrar_vector(float *A, long int N)
{
    long int i;
    for (i = 0; i < N; i++)
    {
        cout << A[i] << " ";
    }
    cout << "\n";
}