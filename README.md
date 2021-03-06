# CUDA-Matirx-Multiplication

   矩阵乘法在计算机图形学中经常会使用到，当矩阵的维度较小时，普通的矩阵乘法运算可以直接循环实现，或者利用标准的第三方库来直接调用。但是当矩阵的维度达到很大的规模时，比如两个上千或者上万维度的矩阵进行乘法运算（这种维度在图形学中很常见）时，如果仍然采用传统的方法将会消耗大量的时间，如果在时间优先的任务中，则需要采用其他的方法来加速矩阵运算。接下来将介绍6种不同的矩阵乘法运算方法并比较他们的运算效率：

1. CPU端循环计算矩阵乘法
2. [Eigen库](http://eigen.tuxfamily.org/index.php?title=Main_Page)矩阵乘法
3. GPU并行矩阵乘法
4. GPU并行矩阵乘法优化--共享内存
5. GPU并行矩阵乘法优化--并行规约
6. GPU矩阵计算库--[cuBLAS](https://developer.nvidia.com/cublas)

## 1、CPU端循环计算矩阵乘法

<img src="https://github.com/fanghao6666/CUDA-Matirx-Multiplication/blob/master/image/matrix.PNG" width="70%" height="70%"/>

   矩阵乘法的规则如上图所示，矩阵A乘上矩阵B得到矩阵C，矩阵C的第i行第j列个元素的值为矩阵A的第i行向量与矩阵B的第j列向量的点积。则CPU端循环计算矩阵乘法即对结果矩阵进行循环计算，对每个元素进行一次点乘运算，我们假定A矩阵的大小为M*N,B矩阵的大小为N*K，结果矩阵C的大小为M*K，则总共需要进行M*K次的向量点乘。这样做的效率显然是非常低的。具体代码如下：
~~~cpp
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
~~~

## 2、[Eigen库](http://eigen.tuxfamily.org/index.php?title=Main_Page)矩阵乘法

Eigen是一个高层次的C ++库，有效支持线性代数，矩阵和矢量运算，数值分析及其相关的算法.Eigen中内建的有矩阵数据结构MatrixXd,我们将A和B矩阵转换为Eige的MatrixXd数据结构，然后直接进行矩阵乘法就可以得到结果矩阵，Eigen的运算也是在CPU端进行的，但是相比较CPU端的遍历循环的方法，Eigen对矩阵乘法做了一些优化，大大的提升了运算速度，但是毕竟在CPU上进行运算，所以当矩阵的维度达到一定大小时，消耗时间还是会大幅增加。具体代码如下：
~~~cpp
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
~~~

## 3、GPU并行矩阵乘法

   在CPU端进行矩阵乘法运算的限制在于对于结果矩阵C的每个值都需要进行一次向量点乘运算，但是我们观察到C中每个值的计算是相对独立的，不需要依赖其他的值，这也就意味这这些运算是可以并行处理的，于是我们便可以利用GPU的并行计算能力来进行矩阵乘法运算。如果你对CUDA的一些概念不是很理解的话，可以先看下[这篇文章](https://blog.csdn.net/hujingshuang/article/details/53097222),简单了解一下grid,block,thread等基本概念。然后介绍下在GPU上进行矩阵运算的基本思路：由于得到结果矩阵C我们需要进行M*K次的向量点乘运算，于是我们开辟M*k个线程，每个线程负责计算其中的一个值，那么当所有线程同时计算完成时也就得到了结果矩阵C，于是将M*K次运算简化为1次运算，理论上时间缩短了M*K倍，当然由于还需要CPU和GPU之间进行数据通信，所以实际上并没有M*K倍，但是速度提升的效果还是十分明显的，矩阵的维度越大时效果越明显。具体代码如下：
~~~cpp
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
	if (threadId >= m * k)
		return;

	int row = threadId / k;
	int col = threadId % k;

	m_r[threadId] = 0;
	for (size_t i = 0; i < n; ++i)
	{
		m_r[threadId] += m_a[row * n + i] * m_b[i * k + col];
	}
}
~~~   

## 4、GPU并行矩阵乘法优化--共享内存

<img src="https://github.com/fanghao6666/CUDA-Matirx-Multiplication/blob/master/image/GPU.png" width="70%" height="70%"/>

   上图为GPU CUDA的架构图，CUDA中的内存种类有多种，每个Grid内部有Constant Memory,Global Memory,和Texture Memory,每个Block内部有Shared Memory,每个Thread对应有Local Memory和Register。关于更加详细的内存介绍可以看[这篇文章](https://blog.csdn.net/qq_36387683/article/details/81125453),我们将采用shared Memory来对矩阵乘法进行加速。
   从上述的架构图和未优化的代码来看，我们的A和B矩阵是存储在Global Memory上，每个线程需要进行计算的时候都会到Global Memory上去取数据，然后再进行计算。在GPU中，传输数据所占用的时间远远要大于进行计算所花的时间。所以我们得想办法能不能减少数据的传输。利用Shared Memory就可以很好的解决这个问题。Shared Memory是位于每个Block内部的内存，所有的该Block内部的Thread共享这个内存，这样每个线程计算的时候就不需要每个线程都取Global Memory取数据，大大节省了数据传输的时间。由于每个Block的最大线程数有限（GTX1080 1024 线程/BLOCK），而且每个block的shared memory大小有限，所以并不能把A矩阵和B矩阵都导入到shared Memory里去，于是我们便对矩阵进行分块处理，如下图所示：
   
<img src="https://github.com/fanghao6666/CUDA-Matirx-Multiplication/blob/master/image/shared.PNG" width="70%" height="70%"/>

我们将结果矩阵C分为若干小块，每个小块的大小为32 * 32 （1024），每个小块作为一个Block,计算这个block内部的数据只需要A矩阵中的32 * N的子矩阵和B矩阵中的N * 32的子矩阵，于是我们将A和B的子矩阵一次性从Global Memory 导入到Block的shared Memory内部，然后再对Block内部的所有Thread进行计算。利用shared memory可以大大的减少线程对于Global Memory的数据传输，极大的减少了时间消耗。具体代码如下：
~~~cpp
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
~~~

## 5、GPU并行矩阵乘法优化--并行规约

<img src="https://github.com/fanghao6666/CUDA-Matirx-Multiplication/blob/master/image/reduction.png" width="70%" height="70%"/>

前面说到普通的GPU并行计算时间在于两个方面，一方面是从Global Memory读取数据，一方面是从进行向量点乘，利用shared memory可以解决读取数据问题，对于向量的点乘可以利用并行规约的方法来实现。并行规约采用的是分治思想，如上图所示，每次对一半的数进行规约，知道最终只有一个值，这种方法要求求和的向量维度要是2的幂，如果不是需要补零。这里我们采用的是第三方库[thrust](http://thrust.github.io/),具体代码如下：
~~~cpp
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
__host__ void matrixMulUseGPUWithReduction(float* m_a, float* m_b, float* m_r, unsigned int m, unsigned int n, unsigned int k)
{
	if (m <= 0 || n <= 0 || k <= 0)
	{
		cout << "Matrix size error !" << endl;
		return;
	}
	vector<vector<float>> a;
	vector<vector<float>> b;

	vector<float> a_tmp;
	vector<float> b_tmp;
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			a_tmp.push_back(m_a[i * n + j]);
		}
		a.push_back(a_tmp);
		a_tmp.clear();
	}
	for (int i = 0; i < k; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			b_tmp.push_back(m_b[j * k + i]);
		}
		b.push_back(b_tmp);
		b_tmp.clear();
	}
	clock_t start = clock();
	thrust::device_vector<float> result(n);
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			thrust::device_vector<float> row(a[i].begin(), a[i].end());
			thrust::device_vector<float> col(b[j].begin(), b[j].end());

			thrust::transform(row.begin(), row.end(), col.begin(), result.begin(), thrust::multiplies<float>());

			m_r[i * k + j] = thrust::reduce(result.begin(),result.end());
		}
	}
	clock_t end = clock();


	cout << endl << "*******************************************" << endl
		<< "Matrix A size : " << m << " * " << n << endl
		<< "Matrix B size : " << n << " * " << k << endl
		<< "GPUMultiTimeReduction : " << (end - start) << " ms" << endl
		<< "*******************************************"
		<< endl;

	return;
}
~~~
## 6、GPU矩阵计算库--[cuBLAS](https://developer.nvidia.com/cublas)

cuBLAS是CUDA专门用来矩阵运算的库，内部进行了很多的优化，效果表现也是目前用到的最好的，具体的使用可以看官方的[使用文档](https://docs.nvidia.com/cuda/index.html)。本文实现了一个样例，具体代码如下:
~~~cpp
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
~~~

## 速度对比总结

<img src="https://github.com/fanghao6666/CUDA-Matirx-Multiplication/blob/master/image/table1.PNG" width="70%" height="70%"/>

如上表所示，我们对比了各种矩阵乘法的运算速度，矩阵大小列中各个值代码矩阵的维度，A和B维度相同，即160代表M = N = K = 160。时间单位为ms。将上表用折线图表示如下：

<img src="https://github.com/fanghao6666/CUDA-Matirx-Multiplication/blob/master/image/graph1.PNG" width="70%" height="70%"/>

可以看到随着矩阵维度增加时，CPU端循环计算方法时间急剧增加，Eigen的变换比较平缓但是也是逐渐大于GPU端的各种算法。我们对GPU端的各种算法做了一个对比如下图：

<img src="https://github.com/fanghao6666/CUDA-Matirx-Multiplication/blob/master/image/graph2.PNG" width="70%" height="70%"/>

可以看到利用shared memory的效果要明显好于普通的GPU并行计算，但是效果最好的还是cuBLAS的矩阵运算，比shared memory方法要快十倍以上。具体怎么实现如此快速的计算，需要去官方文档寻找答案。

## 备注

完整部分的代码在 Kernel.cu文件，如有疑问，欢迎交流~
Zutterhao Nanjing University VISG


