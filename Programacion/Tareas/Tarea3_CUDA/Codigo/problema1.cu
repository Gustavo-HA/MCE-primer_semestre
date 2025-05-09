#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

using namespace cv;
using namespace std;

void secuencial(Mat A, Mat B, Mat alpha, Mat &salida)
{
    multiply(alpha, A, A);                    // A*alpha
    multiply(Scalar::all(1.0) - alpha, B, B); // B*(1-alpha)
    add(A, B, salida);
}

__global__ void paralelo(const float *A, const float *B,
                         const float *alpha, float *salida,
                         int ancho, int alto, int canales)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < ancho && y < alto)
    {
        int idx = (y * ancho + x) * canales;

        for (int c = 0; c < canales; ++c)
        {
            salida[idx + c] = alpha[idx + c] * A[idx + c] +
                              (1.0f - alpha[idx + c]) * B[idx + c];
        }
    }
}

int main()
{

    // Lee las imagenes
    Mat A = imread("../Imagenes/Problema1/greenscreen.jpg");
    Mat B = imread("../Imagenes/Problema1/fondo.bmp");
    Mat alpha = imread("../Imagenes/Problema1/greenscreenMask.bmp");
    Mat salida;

    // N_iteraciones para medir tiempos de
    // ejeucción
    int n_iteraciones = 100;

    // Caracteristicas de las imagenes
    int ancho = A.cols;
    int alto = A.rows;
    int canales = 3;
    int N = alto * ancho * 3;
    size_t size = N * sizeof(float);

    /*       Pasos previos      */

    // Convertir a float y normalizar alpha
    A.convertTo(A, CV_32FC3, 1.0 / 255);
    B.convertTo(B, CV_32FC3, 1.0 / 255);
    alpha.convertTo(alpha, CV_32FC3, 1.0 / 255);

    // Espacio en host para la salida
    Mat salida_secuencial = Mat::zeros(A.size(), A.type());
    Mat salida_paralelo = Mat::zeros(A.size(), CV_32FC3);

    /*          SECUENCIAL            */
    auto inicio = chrono::high_resolution_clock::now();

    for (int k = 0; k < n_iteraciones; k++)
        secuencial(A, B, alpha, salida_secuencial);

    auto fin = chrono::high_resolution_clock::now();
    chrono::duration<double> duracion = (fin - inicio) / n_iteraciones;
    cout << "Tiempo en secuencial (segundos):\t" << duracion.count() << endl;

    salida_secuencial.convertTo(salida_secuencial, CV_8UC3, 255.0);
    imwrite("../Imagenes/Problema1/p1_secuencial.jpg", salida_secuencial);

    /*          PARALELO            */
    // Medir tiempos en PARALELO
    float tiempo_paralelo = 0;
    cudaEvent_t inicio_p, fin_p; // Declaramos los eventos
    cudaEventCreate(&inicio_p);  // Los creamos
    cudaEventCreate(&fin_p);

    // Alojamos memoria en el device
    float *d_A, *d_B, *d_alpha, *d_salida;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_alpha, size);
    cudaMalloc(&d_salida, size);

    // Copiar al device
    cudaMemcpy(d_A, A.ptr<float>(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.ptr<float>(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, alpha.ptr<float>(), size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((ancho + blockDim.x - 1) / blockDim.x, (alto + blockDim.y - 1) / blockDim.y);

    cudaEventRecord(inicio_p);

    for (int k = 0; k < n_iteraciones; k++)
        paralelo<<<gridDim, blockDim>>>(d_A, d_B, d_alpha, d_salida, ancho, alto, canales);

    cudaEventRecord(fin_p);

    cudaEventSynchronize(fin_p);
    cudaEventElapsedTime(&tiempo_paralelo, inicio_p, fin_p);
    tiempo_paralelo = tiempo_paralelo / (1000 * n_iteraciones); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos):\t\t" << tiempo_paralelo << "\t"
         << "Speedup: " << duracion.count() / tiempo_paralelo << endl;

    // Copiar al host
    cudaMemcpy(salida_paralelo.ptr<float>(), d_salida, size, cudaMemcpyDeviceToHost);

    // Dar salida
    salida_paralelo.convertTo(salida_paralelo, CV_8UC3, 255.0);
    imwrite("../Imagenes/Problema1/p1_paralelo.jpg", salida_paralelo);
    printf("El programa se ejecuto exitosamente...\n");

    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_alpha);
    cudaFree(d_salida);

    return 0;
}