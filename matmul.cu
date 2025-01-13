#include <iostream>
#include <ctime> //To get time
#include <chrono>

# define TILING_SIZE 4


__global__ void MatrixMulGPU(float *A, float *B, float *C, int matrixWidth) {
    //Each thread executes this function and there are blocksPerGrid * threadsPerBlock number of threads

    
    //blockDim.y is number of threads in block in y direction
    //To visualize this, think about this form:
    // [a  b] * [e f]  = [i  j]  
    // [c  d]   [g h]    [k  l]
    // Here to get val l, we have row = 1, col = 1. We take row from the first matrix and col from the sec
    // If we think about the number of threads in a block, we have one thread for each of vals in the resultant matrix
    // i.e, we have one thread for i, one for j, one for k and one for l.
    // Hence, for l, the thread's val is : x= 1, y = 1. 
    // But to get the row value from the bigger matrix, we have to shift the row by which location
    // we are in the bigger matrix, we get this by blocks's position (block's x,y location)
    // and we shift it by 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //Shared memory for threads in a block
    __shared__ float blockA[TILING_SIZE][TILING_SIZE];
    __shared__ float blockB[TILING_SIZE][TILING_SIZE];

    //This is the final sum that the thread calculates for one value for C matrix
    float sum=0.0;

    //To get value for a single output (Which is what this particular thread executing MatrixMulGPU is trying to do)
    //we need to go through A and B matrixWidth/TILING_SIZE times. 
    for (int tilenum=0; tilenum < matrixWidth/TILING_SIZE; tilenum++) {
        //Now to get actual value, we need to shift the block appropriately
        //across the A and B matrix
        blockA[threadIdx.y][threadIdx.x] = A[row*matrixWidth  + (tilenum*TILING_SIZE + threadIdx.x)];
        //We are taking value just for the current thread, so we get location in block
        //by first shifting with row count and then locating the approapriate place with tile index

        //To get column in B, we shift through matrixWidth by checking which tile index to access
        //We take threadIdx.y here because the position in B matrix is determined by which row
        //we want to fill in our final block (the resultant intermediate tile from mutplication of A and B)
        //Ex., for 2nd row, we want to shift by 2 of matrixWidth in B before we get to our block  
        blockB[threadIdx.y][threadIdx.x] = B[ (tilenum*TILING_SIZE + threadIdx.y)*matrixWidth + col];

        __syncthreads();
        //Synchronise across all threads across the block (So, all the threadsPerBlock will fill in the both blockA and block B with 
        // TILING_SIZE*TILING_SIZE values)

        //Do the multiplication and addition

        for(int i=0; i<TILING_SIZE; i++){
            //Now, blockA and block B are small blocks that we took out from A,B
            //To get the final sum, we dont care about row and col, just thread indexes
            sum += blockA[threadIdx.y][i] * blockB[i][threadIdx.x];
        }
    }

    //This is what this thread was trying to calculate all along
    C[row*matrixWidth + col] = sum;
}

//Check performance improvement of doing matrix multiplication over GPU (as compared to CPU)
int main(int argc, char* argv[]){
    int matrixWidth = 64;

    if(argc != 3 || strcmp(argv[1], "-m")!=0 ) {
        std::cout<<"Using matrixWidth 64, can pass a different size with -m parameter"<<std::endl;
    } else {
        matrixWidth = atoi(argv[2]);
    }

    int matrixSize = matrixWidth * matrixWidth;

    float *A, *B, *resGPU, *resCPU;

    cudaMallocManaged(&A, matrixSize* sizeof(float));
    cudaMallocManaged(&B, matrixSize* sizeof(float));
    cudaMallocManaged(&resGPU, matrixSize* sizeof(float));
    cudaMallocManaged(&resCPU, matrixSize* sizeof(float));

    //initialize A and B randomnly

    std::srand(std::time(nullptr)); //Initialize randomness with current time
    for(int i=0;i<matrixSize;i++) {
        A[i] = static_cast<float>(rand())/RAND_MAX;
        B[i] = static_cast<float>(rand())/RAND_MAX;
    }

    //First multiply with GPU
    dim3 blocksPerGrid(matrixWidth/TILING_SIZE, matrixWidth/TILING_SIZE); // 16, 16 blocks for TILING SIZE 4 and 64x64 matrix
    dim3 threadsPerBlock(TILING_SIZE, TILING_SIZE); //4,4 for TILING SIZE 4


    cudaEvent_t startT, stopT;

    cudaEventCreate(&startT); cudaEventCreate(&stopT);

    float gpu_time = 0, cpu_time=0;

    cudaEventRecord(startT);
    //Do the GPU calculation
    
    MatrixMulGPU <<< blocksPerGrid, threadsPerBlock>>> (A, B, resGPU, matrixWidth);

    cudaEventRecord(stopT);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&gpu_time, startT, stopT);
    
    
    auto start_time = std::chrono::high_resolution_clock::now();
    //Multiply with CPU
    for(int i=0;i<matrixWidth;i++) {
        for(int j=0;j<matrixWidth;j++) {
            resCPU[i*matrixWidth + j] = 0.0;

            for(int k=0;k<matrixWidth;k++){
                resCPU[i*matrixWidth + j]  += A[i*matrixWidth + k] * B[k*matrixWidth + j];
            }
        }
    }   

    auto end_time = std::chrono::high_resolution_clock::now();
    cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    cpu_time = cpu_time / 1000;

    //Compare the results
    float err;
    for(int i=0;i<matrixWidth*matrixWidth;i++){
        err = std::abs(resCPU[i] - resGPU[i]);

        if(err > 1e-3)
        {
            std::cout<<"Error: CPU and GPU didnt return the same result.\n";
            return -1;
        }
    }

    std::cout<<"GPU took: "<<gpu_time<<std::endl;
    std::cout<<"CPU took: "<<cpu_time<<std::endl;
    std::cout<<"Speedup: "<<cpu_time/gpu_time<<std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(resCPU);
    cudaFree(resGPU);

    return 0;
}