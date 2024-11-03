#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

using namespace std;

double suma_paralelo(double *A, long int N)
{
    double suma = 0;
    long int i;

    // Código en paralelo

#pragma omp parallel for shared(A, N) private(i) reduction(+ : suma)
    for (i = 0; i < N; i++)
    {
        suma += A[i];
    }

    cout << "Resultado en paralelo: " << suma << endl;

    return suma;
}

int main()
{
    double *A, suma, tiempo_p, tiempo_s;
    long int N, i;
    clock_t inicio_p, fin_p;
    clock_t inicio_s, fin_s;

    /* Llenado manual

    cout << "N: ";
    cin >> N;

    A = (double *)malloc(N * sizeof(double));

    cout << "Vector:\n";
    for (i = 0; i < N; i++)
    {
        cin >> A[i];
    }
    */

    /* Llenado automático para pruebas de velocidad. */

    N = 10'000'000;
    A = (double *)malloc(N * sizeof(double));
    for (i = 0; i < N; i++)
    {
        A[i] = static_cast<double>(i);
    }

    // Código paralelo
    inicio_p = clock();
    suma = suma_paralelo(A, N);
    fin_p = clock();
    tiempo_p = ((double)(fin_p - inicio_p)) / CLOCKS_PER_SEC;
    printf("Tiempo paralelo: %.4f segundos.\n", tiempo_p);

    // Código secuencial
    inicio_s = clock();
    suma = 0;
    for (i = 0; i < N; i++)
    {
        suma += A[i];
    }
    cout << "Resultado en secuencial: " << suma << endl;
    fin_s = clock();
    tiempo_s = ((double)(fin_s - inicio_s)) / CLOCKS_PER_SEC;
    printf("Tiempo secuencial: %.4f segundos.\n", tiempo_s);

    // Libera memoria
    free(A);
    return 0;
}