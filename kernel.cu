/*
* This is a program to test 
* matrix multiplication efficiency
* On CPU and GPU
*
* @Author: zutterhao Nanjing University
* @Date: 2019/5/24
*/
#include <iostream>
#include <string>
#include <vector>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

using namespace std;

const unsigned int THREAD_PER_BLOCK = 1024;
const int BLOCK_SIZE = 16;



/*
* @func: print matrix(small size)
*
* @para: matrix print matrix
*/
__host__ void printMatrix(float* matrix,unsigned int row,unsigned int col)
{
	if (col > 16 || row > 16)
	{
		cout << "Matrix size to large !" << endl;
		return;
	}
	cout << endl << "*******************************************" << endl
		<< "Print Matrix :" << endl;
	for (size_t i = 0; i < row; ++i)
	{
		for (size_t j = 0; j < col; ++j)
		{
			cout << matrix[i * col + j] << " ";
		}
		cout << endl;
	}
	cout << "*******************************************" << endl;
	return;
}
/*
* @func: array to vector
*
* @para: arr pointer to array
*		 num  array element number
*/
vector<float> arrayToVector(float *arr,unsigned int num)
{
	vector<float> result(num);
	for (size_t i = 0; i < num; ++i)
	{
		result[i] = arr[i];
	}
	return result;
}


/*
* @func: matrix multiplication on cpu
*
* @para: m_a  left multi matrix
*		 m_b  right multi matrix
*		 m_r  result matrix
*		 m    left matrix rows
*        n    left matrix cols
*        k    right matrix cols
*/
__host__ void matrixMulOnCPU(float* m_a, float* m_b, float* m_r, unsigned int m, unsigned int n, unsigned int k)
{
	if (m <= 0 || n <= 0 || k <= 0)
	{
		cout << "Matrix size error !" << endl;
		return;
	}
	clock_t multi_start = clock();
	for (size_t i = 0; i < m; ++i)
	{
		for (size_t j = 0; j < k; ++j)
		{
			float tmp = 0.0f;
			unsigned int index = i * k + j;
			for (size_t t = 0; t < n; ++t)
			{
				tmp += m_a[n * i + t] * m_b[k * t + j];
			}
			m_r[index] = tmp;
		}
	}
	clock_t multi_end = clock();
	cout << endl << "*******************************************" << endl
		<< "Matrix A size : " << m << " * " << n << endl
		<< "Matrix B size : " << n << " * " << k << endl
		<< "CPUMultiTime : " << (multi_end - multi_start) << " ms" << endl
		<< "*******************************************"
		<< endl;
	return;
}


/*
* @func: generate random matrix
*
* @para: m_a  left multi matrix
*		 m_b  right multi matrix
*		 m    left matrix rows
*        n    left matrix cols
*        k    right matrix cols
*/
__host__ void genMatrixValue(float** m_a, float** m_b, float** m_r, unsigned int m, unsigned int n, unsigned int k)
{
	if (m <= 0 || n <= 0 || k <= 0)
	{
		cout << "Generate matrix size error !" << endl;
		return;
	}
	clock_t gen_start = clock();
	// malloc memory
	size_t m_a_size = m * n * sizeof(float);
	size_t m_b_size = n * k * sizeof(float);
	size_t m_r_size = m * k * sizeof(float);

	*m_a = (float *)malloc(m_a_size);
	*m_b = (float *)malloc(m_b_size);
	*m_r = (float *)malloc(m_r_size);

	// generate m_a
	for (size_t i = 0; i < m; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			(*m_a)[n * i + j] = float(n * i + j);
		}
	}
	// generate m_b
	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < k; ++j)
		{
			(*m_b)[k * i + j] = float(n * k - k * i -j - 1);
		}
	}
	clock_t gen_end = clock();
	cout << "Generate matrix_a and matrix_b succeed ! Time : " << (gen_end - gen_start) << " ms" << endl;
}

/*
* @func: matrix multiplication use eigen 
*
* @para: m_a  left multi matrix
*		 m_b  right multi matrix
*		 m_r  result matrix
*		 m    left matrix rows
*        n    left matrix cols
*        k    right matrix cols
*/
__host__ void matrixMulUseEigen(float* m_a, float* m_b, float* m_r, unsigned int m, unsigned int n, unsigned int k)
{
	if (m <= 0 || n <= 0 || k <= 0)
	{
		cout << "Matrix size error !" << endl;
		return;
	}
	// Matrix variable
	Eigen::MatrixXd left = Eigen::MatrixXd::Zero(m, n);
	Eigen::MatrixXd right = Eigen::MatrixXd::Zero(n, k);
	Eigen::MatrixXd result;

	// copy matrix from array to eigen matrix
	for (size_t i = 0; i < m; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			left(i, j) = m_a[i * n + j];
		}
	}
	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < k; ++j)
		{
			right(i, j) = m_b[i * k + j];
		}
	}
	clock_t eigen_start = clock();
	result = left * right;
	clock_t eigen_end = clock();
	
	for (size_t i = 0; i < m; ++i)
	{
		for (size_t j = 0; j < k; ++j)
		{
			m_r[i * k + j] = result(i, j);
		}
	}
	cout << endl << "*******************************************" << endl
		<< "Matrix A size : " << m << " * " << n << endl
		<< "Matrix B size : " << n << " * " << k << endl
		<< "EigenMultiTime : " << (eigen_end - eigen_start) << " ms" << endl
		<< "*******************************************"
		<< endl;
	return;
}

/*
* @func: matrix multiplication on gpu
*
* @para: m_a  left multi matrix
*		 m_b  right multi matrix
*		 m_r  result matrix
*		 m    left matrix rows
*        n    left matrix cols
*        k    right matrix cols
*/
__global__ void matrixMulOnGPU(float* m_a, float* m_b, float* m_r, unsigned int m, unsigned int n, unsigned int k)
{
	int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= n * k)
		return;

	int row = threadId / k;
	int col = threadId % k;

	m_r[threadId] = 0;
	for (size_t i = 0; i < n; ++i)
	{
		m_r[threadId] += m_a[row * n + i] * m_b[i * k + col];
	}
}
/*
* @func: matrix multiplication use gpu accelerate
*
* @para: m_a  left multi matrix
*		 m_b  right multi matrix
*		 m_r  result matrix
*		 m    left matrix rows
*        n    left matrix cols
*        k    right matrix cols
*/
__host__ void matrixMulUseGPU(float* m_a, float* m_b, float* m_r, unsigned int m, unsigned int n, unsigned int k)
{
	if (m <= 0 || n <= 0 || k <= 0)
	{
		cout << "Matrix size error !" << endl;
		return;
	}
	float *MA, *MB, *MR;

	unsigned int a_num = m * n;
	unsigned int b_num = n * k;
	unsigned int r_num = m * k;

	// Allocated memory
	cudaMalloc((void**)&MA, a_num * sizeof(float));
	cudaMalloc((void**)&MB, b_num * sizeof(float));
	cudaMalloc((void**)&MR, r_num * sizeof(float));

	// copy data from cpu to gpu
	clock_t todevice_start = clock();
	cudaMemcpy(MA, m_a, a_num * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(MB, m_b, b_num * sizeof(float), cudaMemcpyHostToDevice);
	clock_t todevice_end = clock();

	// calculate grids and blocks
	unsigned int thread_num = min(r_num, THREAD_PER_BLOCK);
	unsigned int block_num = (r_num % thread_num != 0) ? (r_num / thread_num + 1) : (r_num / thread_num);
	
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, NULL);
	matrixMulOnGPU << <block_num, thread_num >> > (MA, MB, MR, m, n, k);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float gpu_time = 0.0f;
	cudaEventElapsedTime(&gpu_time, start, stop);

	// copy data from gpu to cpu
	clock_t tohost_start = clock();
	cudaMemcpy(m_r, MR, r_num * sizeof(float), cudaMemcpyDeviceToHost);
	clock_t tohost_end = clock();

	cout << endl << "*******************************************" << endl
		<< "Matrix A size : " << m << " * " << n << endl
		<< "Matrix B size : " << n << " * " << k << endl
		<< "HostToDevice : " << (todevice_end - todevice_start) << " ms" << endl
		<< "GPUMultiTime : " << gpu_time << " ms" << endl
		<< "DeviceToHost : " << (tohost_end - tohost_start) << " ms" << endl
		<< "TotalTime : " << (tohost_end - todevice_start) << " ms" << endl
		<< "*******************************************"
		<< endl;
	
	// free memory
	cudaFree(MA);
	cudaFree(MB);
	cudaFree(MR);

	return;
}

/*
* @func: matrix multiplication on gpu use shared memory
*
* @para: m_a  left multi matrix
*		 m_b  right multi matrix
*		 m_r  result matrix
*		 m    left matrix rows
*        n    left matrix cols
*        k    right matrix cols
*/
template<int BLOCK_SIZE>
__global__ void matrixMulOnGPUWithShared(float* m_a, float* m_b, float* m_r, unsigned int m, unsigned int n, unsigned int k)
{
	// thread location
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;

	if ((thread_y + block_y * blockDim.y) * k + block_x * blockDim.x + thread_x >= m * k)
	{
		return;
	}

	// blockDim.x == blockDim.y == BLOCK_SIZE here
	const int begin_a = block_y * blockDim.y * n;
	const int end_a = begin_a + n - 1;
	const int step_a = blockDim.x;

	const int begin_b = block_x * blockDim.x;
	const int step_b = blockDim.y * k;

	float result_temp = 0.0f;

	for (int index_a = begin_a, int index_b = begin_b; index_a < end_a; index_a += step_a, index_b += step_b)
	{
		// shared memory
		__shared__ float SubMat_A[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float SubMat_B[BLOCK_SIZE][BLOCK_SIZE];

		// copy data to shared memory
		SubMat_A[thread_y][thread_x] = m_a[index_a + thread_y * n + thread_x];
		SubMat_B[thread_y][thread_x] = m_b[index_b + thread_y * k + thread_x];

		__syncthreads();

		for (int i = 0; i < BLOCK_SIZE; ++i)
		{
			result_temp += SubMat_A[thread_y][i] * SubMat_B[i][thread_x];
		}

		__syncthreads();
	}

	int begin_result = block_y * blockDim.y * k + begin_b;
	m_r[begin_result + thread_y * k + thread_x] = result_temp;
}

/*
* @func: matrix multiplication on gpu use shared memory
*
* @para: m_a  left multi matrix
*		 m_b  right multi matrix
*		 m_r  result matrix
*		 m    left matrix rows
*        n    left matrix cols
*        k    right matrix cols
*/
__host__ void matrixMulUseGPUWithShared(float* m_a, float* m_b, float* m_r, unsigned int m, unsigned int n, unsigned int k)
{
	if (m <= 0 || n <= 0 || k <= 0)
	{
		cout << "Matrix size error !" << endl;
		return;
	}
	float *MA, *MB, *MR;

	unsigned int a_num = m * n;
	unsigned int b_num = n * k;
	unsigned int r_num = m * k;

	// Allocated memory
	cudaMalloc((void**)&MA, a_num * sizeof(float));
	cudaMalloc((void**)&MB, b_num * sizeof(float));
	cudaMalloc((void**)&MR, r_num * sizeof(float));

	// copy data from cpu to gpu
	clock_t todevice_start = clock();
	cudaMemcpy(MA, m_a, a_num * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(MB, m_b, b_num * sizeof(float), cudaMemcpyHostToDevice);
	clock_t todevice_end = clock();

	// define grids and blocks size
	unsigned int thread_num = min(r_num, THREAD_PER_BLOCK);
	unsigned int block_num = (r_num % thread_num != 0) ? (r_num / thread_num + 1) : (r_num / thread_num);

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, NULL);
	matrixMulOnGPUWithShared<16> << <grid, block >> > (MA, MB, MR, m, n, k);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float gpu_time = 0.0f;
	cudaEventElapsedTime(&gpu_time, start, stop);


	// copy data from gpu to cpu
	clock_t tohost_start = clock();
	cudaMemcpy(m_r, MR, r_num * sizeof(float), cudaMemcpyDeviceToHost);
	clock_t tohost_end = clock();

	cout << endl << "*******************************************" << endl
		<< "Matrix A size : " << m << " * " << n << endl
		<< "Matrix B size : " << n << " * " << k << endl
		<< "HostToDevice : " << (todevice_end - todevice_start) << " ms" << endl
		<< "GPUMultiTimeShared : " << gpu_time << " ms" << endl
		<< "DeviceToHost : " << (tohost_end - tohost_start) << " ms" << endl
		<< "TotalTime : " << (tohost_end - todevice_start) << " ms" << endl
		<< "*******************************************"
		<< endl;

	// free memory
	cudaFree(MA);
	cudaFree(MB);
	cudaFree(MR);

	return;
}

/*
* @func: matrix multiplication on gpu use reduction algorithm
*
* @para: m_a  left multi matrix
*		 m_b  right multi matrix
*		 m_r  result matrix
*		 m    left matrix rows
*        n    left matrix cols
*        k    right matrix cols
*/
__global__ void matrixMulUseGPUWithReduction(float* m_a, float* m_b, float* m_r, unsigned int m, unsigned int n, unsigned int k)
{

}


/*
* @func: matrix multiplication on gpu use cublas
*
* @para: m_a  left multi matrix
*		 m_b  right multi matrix
*		 m_r  result matrix
*		 m    left matrix rows
*        n    left matrix cols
*        k    right matrix cols
*/
__host__ void matrixMulUseGPUWithCublas(float* m_a, float* m_b, float* m_r, unsigned int m, unsigned int n, unsigned int k)
{
	if (m <= 0 || n <= 0 || k <= 0)
	{
		cout << "Matrix size error !" << endl;
		return;
	}
	float *MA, *MB, *MR;

	unsigned int a_num = m * n;
	unsigned int b_num = n * k;
	unsigned int r_num = m * k;

	// Allocated memory
	cudaMalloc((void**)&MA, a_num * sizeof(float));
	cudaMalloc((void**)&MB, b_num * sizeof(float));
	cudaMalloc((void**)&MR, r_num * sizeof(float));

	// copy data from cpu to gpu
	clock_t todevice_start = clock();
	cudaMemcpy(MA, m_a, a_num * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(MB, m_b, b_num * sizeof(float), cudaMemcpyHostToDevice);
	clock_t todevice_end = clock();

	float alpha = 1.0f;
	float beta = 0.0f;
	cublasHandle_t handle;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cublasCreate(&handle);
	cudaEventRecord(start, NULL);
	cublasSgemm(handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		k,
		m,
		n,
		&alpha,
		MB,
		k,
		MA,
		n,
		&beta,
		MR,
		k);
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float cublas_time = 0.0f;
	cudaEventElapsedTime(&cublas_time, start, stop);

	// copy data from gpu to cpu
	clock_t tohost_start = clock();
	cudaMemcpy(m_r, MR, r_num * sizeof(float), cudaMemcpyDeviceToHost);
	clock_t tohost_end = clock();

	cout << endl << "*******************************************" << endl
		<< "Matrix A size : " << m << " * " << n << endl
		<< "Matrix B size : " << n << " * " << k << endl
		<< "HostToDevice : " << (todevice_end - todevice_start) << " ms" << endl
		<< "GPUMultiTimeCublas : " << cublas_time << " ms" << endl
		<< "DeviceToHost : " << (tohost_end - tohost_start) << " ms" << endl
		<< "TotalTime : " << (tohost_end - todevice_start) << " ms" << endl
		<< "*******************************************"
		<< endl;

	// free memory
	cudaFree(MA);
	cudaFree(MB);
	cudaFree(MR);

	return;
}

//************* MAIN FUNCTION ***************//
int main()
{
// 1、 Defining variables
	// left and right matrix size
	// m: left row  
	// n: left col|right row
	// k: right col
	unsigned int m, n, k;
	// left,right,result matrix pointer
	float *matrix_a, *matrix_b, *matrix_r;

// 2、Assignment of variables
	cout << "Please input the size of left and right matrix : (m n k)" << endl;
	while (n <= 0 || m <= 0 || k <= 0)
	{
		cin >> m >> n >> k;
		if (n <= 0 || m <= 0 || k <= 0)
		{
			cout << "Matrix size must be Positive number,please input again" << endl;
		}
	}

// 3、Generate matrix
	genMatrixValue(&matrix_a, &matrix_b, &matrix_r, m, n, k);

// 4、Matrix multiply
	// 4.1、matrix multiply on CPU
	matrixMulOnCPU(matrix_a, matrix_b, matrix_r, m, n, k);
	vector<float> CPU = arrayToVector(matrix_r, m*k);

	// 4.2、matrix multiply use Eigen
	matrixMulUseEigen(matrix_a, matrix_b, matrix_r, m, n, k);
	vector<float> Eigen = arrayToVector(matrix_r, m*k);

	// 4.3、matrix multiply on GPU
	matrixMulUseGPU(matrix_a, matrix_b, matrix_r, m, n, k);
	vector<float> GPU = arrayToVector(matrix_r, m*k);

	// 4.4、matrix multiply on GPU with Shared memory
	matrixMulUseGPUWithShared(matrix_a, matrix_b, matrix_r, m, n, k);
	vector<float> GPUShared = arrayToVector(matrix_r, m*k);

	// 4.5、matrix multiply on GPU with Cublas
	matrixMulUseGPUWithCublas(matrix_a, matrix_b, matrix_r, m, n, k);
	vector<float> GPUCublas = arrayToVector(matrix_r, m*k);


	cout << (GPUCublas == GPU) << endl;

	/*printMatrix(matrix_a, m, n);
	printMatrix(matrix_b, n, k);
	printMatrix(matrix_r, m, k);*/


	getchar();

	return 0;
}