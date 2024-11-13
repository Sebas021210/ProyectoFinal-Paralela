/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Implementación de Transformada de Hough en CUDA
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "pgm.h"
#include <opencv2/opencv.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

//*****************************************************************
// Declaración de memoria constante para senos y cosenos
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

//*****************************************************************
// The CPU function returns a pointer to the accumulator
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc) {
    printf("Iniciando CPU_HoughTran\n");
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2; //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
    *acc = new int[rBins * degreeBins]; //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
    memset(*acc, 0, sizeof(int) * rBins * degreeBins); //init en ceros
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            int idx = j * w + i;
            if (pic[idx] > 0) {
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                float theta = 0;
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++;
                    theta += radInc;
                }
            }
        }
    printf("Finalizando CPU_HoughTran\n");
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
//TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    extern __shared__ int localAcc[];

    int locID = threadIdx.x;

    //TODO calcular: int gloID = ?
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;
    
    // Inicializar el acumulador local en memoria compartida
    for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
        localAcc[i] = 0;
    }
    
    // Barrera de sincronización para asegurar que todos los hilos hayan inicializado localAcc
    __syncthreads();

    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            
            // Incrementar el acumulador local utilizando operación atómica
            atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
        }
    }
    
    // Barrera para asegurar que todos los hilos hayan completado la actualización de localAcc
    __syncthreads();

    // Loop final para sumar valores de localAcc al acumulador global acc
    for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
        atomicAdd(&acc[i], localAcc[i]);
    }

    //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
    //utilizar operaciones atomicas para seguridad
    //faltara sincronizar los hilos del bloque en algunos lados
}

// Función para verificar los bordes detectados
void verifyEdges(cv::Mat& image, const char* window_name) {
    // Aplicar detección de bordes de Canny
    cv::Mat edges;
    cv::Canny(image, edges, 50, 150);
    
    // Guardar la imagen de bordes para verificación
    cv::imwrite("output_edges_detected.jpg", edges);
    printf("Imagen de bordes guardada como 'output_edges_detected.jpg'\n");
    
    // Copiar los bordes detectados de vuelta a la imagen original
    memcpy(image.data, edges.data, image.total() * image.elemSize());
}

// Función para dibujar las líneas detectadas
void drawDetectedLines(cv::Mat &image, int *acc, int w, int h, float rMax, float rScale, int threshold) {
    printf("Iniciando drawDetectedLines\n");

    // Calcular media y desviación estándar
    float mean = 0;
    float max_val = 0;
    int total_values = rBins * degreeBins;

    // Calcular media y valor máximo
    for (int i = 0; i < total_values; i++) {
        mean += acc[i];
        if (acc[i] > max_val) max_val = acc[i];
    }
    mean /= total_values;

    // Calcular desviación estándar
    float std_dev = 0;
    for (int i = 0; i < total_values; i++) {
        std_dev += (acc[i] - mean) * (acc[i] - mean);
    }
    std_dev = sqrt(std_dev / total_values);

    // Establecer threshold dinámico más selectivo
    int dynamic_threshold = (int)(mean + 2 * std_dev);
    threshold = std::max(dynamic_threshold, threshold);

    printf("Estadísticas del acumulador:\n");
    printf("Media: %.2f\n", mean);
    printf("Desviación estándar: %.2f\n", std_dev);
    printf("Valor máximo: %.2f\n", max_val);
    printf("Threshold dinámico: %d\n", dynamic_threshold);
    printf("Threshold final: %d\n", threshold);

    // Vector para almacenar las líneas más fuertes
    std::vector<std::pair<cv::Point, cv::Point>> strong_lines;

    // Crear una copia de la imagen original en escala de grises
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Contar líneas que superan el threshold
    int lines_above_threshold = 0;
    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            int acc_val = acc[rIdx * degreeBins + tIdx];
            if (acc_val > threshold) {
                lines_above_threshold++;
            }
        }
    }
    printf("Número de líneas que superan el threshold: %d\n", lines_above_threshold);

    // Dibujar solo las líneas más significativas
    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            int acc_val = acc[rIdx * degreeBins + tIdx];
            if (acc_val > threshold) {
                float r = rIdx * rScale - rMax;
                float theta = tIdx * radInc;
                float cosTheta = cos(theta);
                float sinTheta = sin(theta);

                cv::Point pt1, pt2;
                if (fabs(sinTheta) > 0.001) {
                    pt1 = cv::Point(0, cvRound((r - 0 * cosTheta) / sinTheta));
                    pt2 = cv::Point(w - 1, cvRound((r - (w - 1) * cosTheta) / sinTheta));
                } else {
                    pt1 = cv::Point(cvRound(r), 0);
                    pt2 = cv::Point(cvRound(r), h - 1);
                }

                // Verificar que los puntos estén dentro de la imagen
                if (pt1.x >= 0 && pt1.x < w && pt1.y >= 0 && pt1.y < h &&
                    pt2.x >= 0 && pt2.x < w && pt2.y >= 0 && pt2.y < h) {
                    strong_lines.push_back(std::make_pair(pt1, pt2));
                    printf("Línea detectada: (%d,%d) -> (%d,%d) con peso %d\n", 
                           pt1.x, pt1.y, pt2.x, pt2.y, acc_val);
                }
            }
        }
    }

    // Dibujar las líneas sobre la imagen
    for (const auto& line : strong_lines) {
        cv::line(image, line.first, line.second, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }

    printf("Finalizando drawDetectedLines - Se dibujaron %zu líneas\n", strong_lines.size());
}

//*****************************************************************
int main(int argc, char **argv) {
    printf("Iniciando el programa\n");

    int i;

    PGMImage inImg(argv[1]);
    printf("Imagen cargada: %s\n", argv[1]);

    int w = inImg.x_dim;
    int h = inImg.y_dim;

    // Cálculo de senos y cosenos en CPU
    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    // TODO eventualmente volver memoria global
    // Asignación de memoria en la GPU para d_Cos y d_Sin en memoria global
    // Copiar valores de senos y cosenos a la memoria constante del device
    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    // Configuración y copia de datos del host al device
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels;
    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // Medición de tiempo usando CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Iniciando el kernel\n");
    cudaEventRecord(start);

    // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    //1 thread por pixel
    int blockNum = ceil(w * h / 256.0);
    GPU_HoughTran<<<blockNum, 256, degreeBins * rBins * sizeof(int)>>>(d_in, w, h, d_hough, rMax, rScale);

    // Verificar errores de ejecución del kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error en la ejecución del kernel: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();
    printf("Kernel ejecutado correctamente\n");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    err = cudaEventElapsedTime(&milliseconds, start, stop);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error al calcular el tiempo de ejecución: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Tiempo de ejecución del kernel: %.3f ms\n", milliseconds);

    err = cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error al copiar los datos de la GPU a la CPU: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("Datos copiados de la GPU a la CPU\n");

    // Crear una matriz de OpenCV a partir de la imagen de entrada
    cv::Mat img(h, w, CV_8UC1, inImg.pixels);

    // Verificar los bordes antes de procesar
    verifyEdges(img, "Edges");

    // Actualizar los pixels de la imagen original con los bordes detectados
    memcpy(inImg.pixels, img.data, w * h);

    cv::Mat colorImg;
    cv::cvtColor(img, colorImg, cv::COLOR_GRAY2BGR);

    // Establecer un threshold inicial razonable
    int threshold = 50;

    // Dibujar las líneas detectadas
    drawDetectedLines(colorImg, h_hough, w, h, rMax, rScale, threshold);

    // Guardar la imagen de salida con las líneas detectadas
    cv::imwrite("output_image.jpg", colorImg);
    printf("Imagen de salida guardada como 'output_image.jpg'\n");

    // Liberación de memoria
    free(pcCos);
    free(pcSin);
    free(h_hough);
    cudaFree(d_in);
    cudaFree(d_hough);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Programa finalizado\n");
    return 0;
}
