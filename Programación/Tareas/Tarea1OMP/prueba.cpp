#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
    long int N = 30;
    int tid;

#pragma omp parallel for shared(N) private(tid)
    for (long int i = 0; i < N; i++)
    {
        tid = omp_get_thread_num();
#pragma omp critical
        cout << "Soy la iteracion i=" << i << " en el hilo tid=" << tid << endl;
    }

    return 0;
}