#include <iostream>
#include <stdio.h>
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

    // Leemos el vector A
    cout << "Vector A:\n";
    for (i = 0; i < N; i++)
    {
        cin >> A[i];
    }

#pragma omp parallel for shared(A, N) private(i) reduction(+ : suma)
    for (i = 0; i < N; i++)
    {
        suma += A[i];
    }

    cout << "Resultado:\n"
         << suma << endl;

    free(A);
    return 0;
}