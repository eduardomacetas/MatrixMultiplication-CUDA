#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>
using namespace std;

#define row 500
#define column 500
#define N 500



void randomsInt(double ** matrix, int n)
{
    for(int i=0;i<n;++i){
	    for(int j=0;j<n;++j){
            matrix[i][j]=rand() % 10 + 1;
        }
    }
}



void print_Matrix(double ** matrix, int n){
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            cout<<"\t"<<matrix[i][j];
        }
        cout<<endl;
    }
    
}

void Multi(double **A, double **B, double **C){

        for(int i=0;i<row;i++){
            for(int j=0;j<column;j++){
                int suma = 0;
                for(int w=0;w<column;w++){
                    suma += A[i][w] * B[w][j];
                }
                C[i][j] = suma;
            }
        }
}


int main(){

    double **A;
    A = new double*[N]; 

    for(int i=0;i<N;++i){
        A[i] =new double[N*N];
    }
    
    
    randomsInt(A,N);
    cout<<"----A----"<<endl;
    //print_Matrix(A,N);



    double **B;
    B = new double*[N]; 

    for(int i=0;i<N;++i){
        B[i] =new double[N*N];
    }


    randomsInt(B,N);
    cout<<"----B----"<<endl;
    //print_Matrix(B,N);
    

    double **C;
    C = new double*[N]; 

    for(int i=0;i<N;++i){
        C[i] =new double[N*N];
    }

    // tiempó:
    clock_t start, end;
    


    cout<<"----C----"<<endl;
    start = clock();
    Multi(A,B,C);
    end = clock();
    double tiempo;

    tiempo= (double)(end-start) / CLOCKS_PER_SEC;
    cout<<"Este es el tiempo:  "<<tiempo*1000.0<<endl;
    cout<<endl;
    //print_Matrix(C,N);

    cout<<"Tef"<<endl;
    return 0;
}