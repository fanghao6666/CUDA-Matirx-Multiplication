# CUDA-Matirx-Multiplication

   矩阵乘法在计算机图形学中经常会使用到，当矩阵的维度较小时，普通的矩阵乘法运算可以直接循环实现，或者利用标准的第三方库来直接调用。但是当矩阵的维度达到很大的规模时，比如两个上千或者上万维度的矩阵进行乘法运算（这种维度在图形学中很常见）时，如果仍然采用传统的方法将会消耗大量的时间，如果在时间优先的任务中，则需要采用其他的方法来加速矩阵运算。接下来将介绍6种不同的矩阵乘法运算方法并比较他们的运算效率：

1. CPU端循环计算矩阵乘法
2. [Eigen库](http://eigen.tuxfamily.org/index.php?title=Main_Page)矩阵乘法
3. GPU并行矩阵乘法
4. GPU并行矩阵乘法优化--共享内存
5. GPU并行矩阵乘法优化--并行规约
6. GPU矩阵计算库--[cuBLAS](https://developer.nvidia.com/cublas)

## CPU端循环计算矩阵乘法

<img src="https://github.com/fanghao6666/CUDA-Matirx-Multiplication/blob/master/image/matrix.PNG" width="60%" height="60%"/>

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

## [Eigen库](http://eigen.tuxfamily.org/index.php?title=Main_Page)矩阵乘法

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

## GPU并行矩阵乘法

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

## GPU并行矩阵乘法优化--共享内存

<img src="https://github.com/fanghao6666/CUDA-Matirx-Multiplication/blob/master/image/GPU.png" width="60%" height="60%"/>

   上图为GPU CUDA的架构图，CUDA中的内存种类有多种，每个Grid内部有Constant Memory,Global Memory,和Texture Memory,每个Block内部有Shared Memory,每个Thread对应有Local Memory和Register。关于更加详细的内存介绍可以看[这篇文章](https://blog.csdn.net/qq_36387683/article/details/81125453),我们将采用shared Memory来对矩阵乘法进行加速。
   从上述的架构图和未优化的代码来看，我们的A和B矩阵是存储在Global Memory上，每个线程需要进行计算的时候都会到Global Memory上去取数据，然后再进行计算。在GPU中，传输数据所占用的时间远远要大于进行计算所花的时间。所以我们得想办法能不能减少数据的传输。利用Shared Memory就可以很好的解决这个问题。Shared Memory是位于每个Block内部的内存，所有的该Block内部的Thread共享这个内存，这样每个线程计算的时候就不需要每个线程都取Global Memory取数据，大大节省了数据传输的时间。由于每个Block的最大线程数有限（GTX1080 1024 线程/BLOCK），而且每个block的shared memory大小有限，所以并不能把A矩阵和B矩阵都导入到shared Memory里去，于是我们便对矩阵进行分块处理，如下图所示：
   
   
<img src="https://github.com/fanghao6666/CUDA-Matirx-Multiplication/blob/master/image/shared.PNG" width="60%" height="60%"/>
