#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
    long int N = 4;
    int tid;

#pragma omp parallel for shared(N) private(tid)
    for (long int i = 0; i < N; i++)
    {
        tid = omp_get_thread_num();
#pragma omp critical
        cout << "Soy la iteracion i =\t" << i << "\ten el hilo tid =\t" << tid << endl;
    }

    cout << "Cantidad de hilos utilizados: " << omp_get_max_threads() << endl;

    return 0;
}