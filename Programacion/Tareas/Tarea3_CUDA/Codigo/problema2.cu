#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__constant__ int k_1[3][3] = {{-1, 0, 1},
                              {-2, 0, 2},
                              {-1, 0, 1}};

__constant__ int k_2[3][3] = {{-1, -2, -1},
                              {0, 0, 0},
                              {1, 2, 1}};

using namespace std;
using namespace cv;

/*          FUNCIONES PARA EL PROCESO EN SECUENCIAL         */

void p2_secuencial(Mat A, Mat &salida, int umbral)
{

    Mat Ix;
    Mat Iy;
    Mat MG = Mat::zeros(A.size(), CV_32F);

    // Para aplicar los filtros rapidamente con la función
    // filter2D, necesitamos matrices del tipo Mat_
    Mat K1 = (Mat_<int>(3, 3) << -1, 0, 1,
              -2, 0, 2,
              -1, 0, 1);

    Mat K2 = (Mat_<int>(3, 3) << -1, -2, -1,
              0, 0, 0,
              1, 2, 1);

    // Calculamos Ix e Iy
    filter2D(A, Ix, CV_32F, K1);
    filter2D(A, Iy, CV_32F, K2);

    // Calculamos MG y al aplicar el umbral, guardamos en salida.
    for (int i = 0; i < A.rows; ++i)
    {
        for (int j = 0; j < A.cols; ++j)
        {
            float ix_ij = Ix.at<float>(i, j);
            float iy_ij = Iy.at<float>(i, j);
            MG.at<float>(i, j) = sqrt(ix_ij * ix_ij + iy_ij * iy_ij);
            // Aplicamos umbral a MG y lo guardamos en salida
            salida.at<uchar>(i, j) = (MG.at<float>(i, j) > umbral) ? 255 : 0;
        }
    }
}

/*          FUNCIONES PARA EL PROCESO EN PARALELO         */

__global__ void p2_paralelo(const float *d_A, float *salida, int ancho, int alto, int umbral)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < ancho - 1 && y > 0 && y < alto - 1)
    {
        int idx = y * ancho + x;

        // Ix e Iy
        float ix = 0.0f;
        float iy = 0.0f;

        for (int i = -1; i <= 1; ++i)
        {
            for (int j = -1; j <= 1; ++j)
            {
                int idx_vecino = (y + i) * ancho + (x + j);
                float valor_pixel = d_A[idx_vecino];

                ix += valor_pixel * k_1[i + 1][j + 1];
                iy += valor_pixel * k_2[i + 1][j + 1];
            }
        }

        // Magnitud
        float magnitud = sqrtf(ix * ix + iy * iy);

        // Aplicamos el umbral.
        salida[idx] = (magnitud > umbral) ? 255.0f : 0.0f;
    }
    else
    {
        int idx = y * ancho + x;
        salida[idx] = 0.0;
    }
}

int main()
{
    // Input
    Mat A = imread("../Imagenes/Problema2/pinzas_gray.png", IMREAD_GRAYSCALE);
    int umbral;
    cout << "Umbral: ";
    cin >> umbral;

    // Espacio en host para la salida
    Mat salida_secuencial = Mat::zeros(A.size(), A.type());
    Mat salida_paralelo = Mat::zeros(A.size(), CV_32F);

    // Medir tiempos
    int n_iteraciones = 50;

    // Caracteristicas de las imagenes
    int ancho = A.cols;
    int alto = A.rows;
    int canales = A.channels();
    int N = alto * ancho;
    size_t size = N * A.elemSize();

    /*          SECUENCIAL          */
    auto inicio = chrono::high_resolution_clock::now();
    p2_secuencial(A, salida_secuencial, umbral);
    auto fin = chrono::high_resolution_clock::now();
    chrono::duration<double> duracion = (fin - inicio) / n_iteraciones;
    cout << "Tiempo en secuencial (segundos):\t" << duracion.count() << endl;

    imwrite("../Imagenes/Problema2/p_2secuencial.jpg", salida_secuencial);

    /*          PARALELO            */
    // Medir tiempos en PARALELO
    float tiempo_paralelo = 0;
    cudaEvent_t inicio_p, fin_p; // Declaramos los eventos
    cudaEventCreate(&inicio_p);  // Los creamos
    cudaEventCreate(&fin_p);

    A.convertTo(A, CV_32F); // Necesitamos hacer A float
    alto = A.rows;
    ancho = A.cols;
    N = alto * ancho;
    size = N * A.elemSize();

    // Alojamos memoria en el device
    float *d_A, *d_salida;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_salida, size);

    // Copiar al device

    cudaMemcpy(d_A, A.ptr<float>(), size, cudaMemcpyHostToDevice);

    // Config CUDA
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((ancho + blockDim.x - 1) / blockDim.x, (alto + blockDim.y - 1) / blockDim.y);

    cudaEventRecord(inicio_p);

    for (int k = 0; k < n_iteraciones; k++)
        p2_paralelo<<<gridDim, blockDim>>>(d_A, d_salida, ancho, alto, umbral);

    cudaEventRecord(fin_p);

    cudaEventSynchronize(fin_p);
    cudaEventElapsedTime(&tiempo_paralelo, inicio_p, fin_p);
    tiempo_paralelo = tiempo_paralelo / (1000 * n_iteraciones); // Milisegundos a segundos.
    cout << "Tiempo en paralelo (segundos):\t\t" << tiempo_paralelo << "\t"
         << "Speedup: " << duracion.count() / tiempo_paralelo << endl;

    // Copiar al host
    cudaMemcpy(salida_paralelo.ptr<float>(), d_salida, size, cudaMemcpyDeviceToHost);
    salida_paralelo.convertTo(salida_paralelo, CV_8U);

    // Dar salida
    imwrite("../Imagenes/Problema2/p2_paralelo.jpg", salida_paralelo);

    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_salida);
    return 0;
}