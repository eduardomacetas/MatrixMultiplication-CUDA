#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <stdio.h>
#define ROW 2 //filas Matriz 1
#define ColRow 2// columna - fila Matriz 1 y 2
#define COL 2 // columna Matriz 2

#define N1 ROW*ColRow // Cantidad de elementos en Matriz 1
#define N2 ColRow*COL // Cantidad de elementos en Matriz 2
#define N3 ROW*COL // Cantidad de elementos en Matriz 3

#define THREADS 7
using namespace std;

void createMatrixHost(float**& host, int row, int col, int size){
    host = (float **)malloc(row*sizeof(float*));
    host[0]=(float *)malloc(size);
	
    for (int i=1; i<row;++i){
        host[i]=host[i-1]+col;
    }
}

void createMatrixHostCUDA(float**& host, float**& device, float **& aux, int row, int col, int size){
    host = (float **)malloc(row*sizeof(float*));
    host[0]=(float *)malloc(size);
    aux =(float **)malloc(row*sizeof(float*));

    cudaMalloc((void **)&aux[0],size);
    cudaMalloc((void **)&device,row*sizeof(float*));

    for (int i=1; i<row;++i){
        host[i]=host[i-1]+col;
        aux[i]=aux[i-1]+col;
    }
    cudaMemcpy(device, aux, row*sizeof(float*), cudaMemcpyHostToDevice);
}

void Multiplicacion(float** A, float** B, float** P){
        for(int i=0;i<ROW;i++){
                for(int j=0;j<COL;j++){
                        float Sum=0.0;
                        for(int k=0;k<ColRow;k++){
                                Sum += A[i][k]*B[k][j];
                        }
                        P[i][j] = Sum;
                }
        }
}

__global__ void MatrixMulKernel(float** A, float** B, float** P)
{
        int Row = blockIdx.y*blockDim.y +threadIdx.y;
        int Col = blockIdx.x*blockDim.x +threadIdx.x;

        if((Row < ROW) && (Col < COL)){
                float Pvalue = 0.0;
                for(int k=0;k<ColRow;k++){
                        Pvalue+= A[Row][k] * B[k][Col];
                }
                P[Row][Col] = Pvalue;
        }
}

__global__ void MatrixMulKernel2(float** A, float** B, float** P)
{
        __shared__ float A_b[THREADS][THREADS];
        __shared__ float B_b[THREADS][THREADS];
        __shared__ float R_b[THREADS][THREADS];

        int Row = blockIdx.y * THREADS + threadIdx.y;
        int Col = blockIdx.x * THREADS + threadIdx.x;

	R_b[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads(); 

	for(int i = 0;i < ceil(ColRow/(float)THREADS);i++){
		A_b[threadIdx.y][threadIdx.x] = 0.0;
		B_b[threadIdx.y][threadIdx.x] = 0.0;
		__syncthreads();

		if ((Row<ROW) && (i*THREADS + threadIdx.x<ColRow)){
	                A_b[threadIdx.y][threadIdx.x] = A[Row] [i*THREADS + threadIdx.x];
			
		}
		
		if ((i*THREADS + threadIdx.y<ColRow) && (Col<COL)){
	                B_b[threadIdx.y][threadIdx.x] = B[i*THREADS + threadIdx.y][Col]; 
		}

                __syncthreads();

                for (int k = 0; k < THREADS; k++) {
                        R_b[threadIdx.y][threadIdx.x] += A_b[threadIdx.y][k] * B_b[k][threadIdx.x];
                }
                __syncthreads();
	 }

	 if((Row<ROW) && (Col<COL))
	         P[Row][Col] = R_b[threadIdx.y][threadIdx.x];
}

void llenarVector(float **V, int row, int col){
    for(int i=0;i<row;i++){
	for(int j=0;j<col;j++){
	        V[i][j]=rand()%11;
	}
    }
}

void imprimir(float **M, int row, int col){
        for(int i=0;i<row;i++){
                for(int j=0;j<col;j++){
                        cout<<M[i][j]<<'\t';
                }
                cout<<endl;
        }
        cout<<endl;
}

int main(){
	float **a, **b, **c1,**c2,**c3;
	//////////////////////////////////////////
	float **d_a, **d_b, **d_c2, **d_c3;
	float **a_aux, **b_aux, **c_aux2, **c_aux3;
	///////////////////////////////////////////

	int size1 = N1 * sizeof(float*);
	int size2 = N2 * sizeof(float*);
	int size3 = N3 * sizeof(float*);
	
	dim3 DimGrid(((COL-1)/THREADS)+1, ((ROW-1)/THREADS)+1, 1);
    dim3 DimBlock(THREADS, THREADS, 1);


	createMatrixHost(c1,ROW,COL,size3);
	createMatrixHostCUDA(a,d_a,a_aux,ROW,ColRow,size1);
	createMatrixHostCUDA(b,d_b,b_aux,ColRow,COL,size2);

	createMatrixHostCUDA(c2,d_c2,c_aux2,ROW,COL,size3);
	createMatrixHostCUDA(c3,d_c3,c_aux3,ROW,COL,size3);
	
    llenarVector(a,ROW,ColRow);
	llenarVector(b,ColRow,COL);

    cout<<"----A----"<<endl;
	imprimir(a,ROW,ColRow);
    
    cout<<"----B----"<<endl;
	imprimir(b,ColRow,COL);

	cudaMemcpy(a_aux[0], a[0], size1, cudaMemcpyHostToDevice);
	cudaMemcpy(b_aux[0], b[0], size2, cudaMemcpyHostToDevice);
/**********************************************************************************/
	cudaEvent_t start2;
    cudaEvent_t stop2;
/**********************************************************************************/

/************************************************************************************/
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2,0);
    MatrixMulKernel2<<<DimGrid,DimBlock>>>(d_a,d_b,d_c3);
    cudaEventRecord(stop2,0);
    cudaEventSynchronize(stop2);
    float elapsedTime2;

    cudaMemcpy(c3[0],c_aux3[0], size3, cudaMemcpyDeviceToHost);
    cout<<"----C----"<<endl;
    imprimir(c3,ROW,COL);


    cudaEventElapsedTime(&elapsedTime2,start2,stop2);
    cout<<"Multiplicacion shared en GPU: "<<elapsedTime2<<endl;
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);	

    
}
