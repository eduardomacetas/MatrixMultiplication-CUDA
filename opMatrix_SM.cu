#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>

using namespace std;

#define row 16
#define column 16
#define threadsPB 32


#define tam 16


// Funcion para generar numeros randoms en mi matrix:
void randomsInt(double **& matrix)
{
    for(int i=0;i<row;++i){
	    for(int j=0;j<column;++j){
            matrix[i][j]=rand() % 10 + 1;
        }
    }
}

//Funcion para separar memoria  con CUDA
void createMatrixHostCUDA(double**& host, double**& device, double **& aux, int size, int r, int c ){
    host = (double **)malloc(r*sizeof(double*));
    host[0]=(double *)malloc(size);
    aux =(double **)malloc(r*sizeof(double*));

    cudaMalloc((void **)&aux[0],size);
    cudaMalloc((void **)&device,r*sizeof(double*));

    for (int i=1; i<r;++i){
        host[i]=host[i-1]+c;
        aux[i]=aux[i-1]+c;
    }
    cudaMemcpy(device, aux, r*sizeof(double*), cudaMemcpyHostToDevice);
}


// Kernel de la funcion Multiplicacion de matrices:
__global__ void Multi(double **A, double **B, double **C){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int suma = 0;

    if(i<row && j<column){
        for(int w=0;w<column;w++){
             suma += A[i][w] * B[w][j];
        }
        C[i][j] = suma;
    }
}

// Kernale de la funcion Muliplicacion de matrices, con Shared:
// http://cuda-programming.blogspot.pe/2013/01/cuda-c-program-for-matrix-addition-and.html
__global__ void MultiShared(double **A, double **B, double **C){
    

    //para generar los id's de los threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Toma de la matriz compartida para romper la Matrix 
    // en el mosaico widht y fatch en esa matriz por ele

    //int gy = gridDim.y;
    __shared__ double sA[tam][tam];
    __shared__ double sB[tam][tam];


    if(i<row && j<column){
        double suma = 0;
        for(int x=0; x<tam;x++){ // M indicar el número de fases

            sA[threadIdx.x][threadIdx.y] = A[i][(x*tam)+threadIdx.y];
            sB[threadIdx.x][threadIdx.y] = B[(x*tam)+threadIdx.x][j];
             __syncthreads() ; //para que sincronize todos los hilos
             
             for(int k=0;k<tam;k++)
                 suma +=  sA[threadIdx.x][k] * sB[k][threadIdx.y];
             __syncthreads() ; //para que sincronize todos los hilos
              
        }
        C[i][j] = suma;
    }
}


// Funcion imprimir:
void print(double ** a){
	for(int i=0;i<row;++i){
	    for(int j=0;j<column;++j){
            cout<<a[i][j]<<'\t';
        }       
	cout<<endl;
    }

	cout<<endl;
}

int main()
{
	srand (time(NULL));
	double **a, **b, **c;
	double **d_a, **d_b, **d_c;
	double **a_aux, **b_aux, **c_aux;
	int size = row* column * sizeof(double*);

	
	createMatrixHostCUDA(a,d_a,a_aux,size,row,column);
    createMatrixHostCUDA(b,d_b,b_aux,size,row,column);
	createMatrixHostCUDA(c,d_c,c_aux,size,row,column);
	
    randomsInt(a);
    randomsInt(b);

	cudaMemcpy(a_aux[0], a[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_aux[0], b[0], size, cudaMemcpyHostToDevice);

	
	dim3 threadPerBlock(threadsPB, threadsPB);
	dim3 blockPerGrid((row+threadPerBlock.x-1)/threadPerBlock.x,(column+threadPerBlock.y-1)/threadPerBlock.y);

    //start_tiempo:
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
        
    cudaEventRecord(start,0);
    MultiShared<<<blockPerGrid,threadPerBlock>>>(d_a,d_b,d_c);
    cudaEventRecord(end,0);
    cudaEventSynchronize(end);
    float elapsedTime;

    cudaEventElapsedTime(&elapsedTime,start,end);
    cout<<"El tiempo es:   "<<elapsedTime<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    //end_tiempo

    cudaMemcpy(c[0],c_aux[0], size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
	cudaFree(a_aux[0]);cudaFree(b_aux[0]);cudaFree(c_aux[0]);


    cout<<"----A----"<<endl;
	print (a);

    cout<<"----B----"<<endl;
    print(b);
    
	cout<<"----c----"<<endl;
	print(c);

	free(a); free(b); free(c);
	return 0;
}
