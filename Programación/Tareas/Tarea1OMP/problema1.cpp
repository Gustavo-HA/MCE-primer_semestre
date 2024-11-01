#include <iostream>
#include <stdlib.h>
#include <omp.h>

using namespace std;

int main()
{
    double *A, suma = 0;
    long int N, i;

    cout << "N: ";
    cin >> N;

    A = (double *)malloc(N * sizeof(double));

    // Leemos el vector
    cout << "Vector:\n";
    for (i = 0; i < N; i++)
    {
        cin >> A[i];
    }

    // Código en paralelo
    #pragma omp parallel for shared(A, N) private(i) reduction(+ : suma)
    for (i = 0; i < N; i++)
    {
        suma += A[i];
    }

    cout << "Resultado en paralelo:\n"
         << suma << endl;

    // Código secuencial
    suma = 0;
    for (i = 0; i < N; i++)
    {
        suma += A[i];
    }

    cout << "Resultado en secuencial:\n"
         << suma << endl;

    free(A);
    return 0;
}