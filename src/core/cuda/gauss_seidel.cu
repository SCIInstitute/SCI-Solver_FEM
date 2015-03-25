#include <smoothers/gauss_seidel.h>
#include <string>
#include <types.h>
#include <cutil.h>
#include <cusp/print.h>
#include <AggMIS_Types.h>

#define FUNC1(i)                                          \
{                                                         \
  Ajreg = Ajlocal[i];                                     \
  Axreg = Axlocal[i];                                     \
  if(Ajreg > 0)                                           \
  {                                                       \
    unsigned int idxi = Ajreg >> 16;                      \
    unsigned int idxj = Ajreg - (idxi << 16);             \
    atomicAdd(&s_Ax[idxi], Axreg * s_x[idxj]);            \
    atomicAdd(&s_Ax[idxj], Axreg * s_x[idxi]);            \
  }                                                       \
  else                                                    \
  {                                                       \
    goto gotolabel;                                       \
  }                                                       \
}                                                         \

#define FUNC2(i)                                          \
{                                                         \
  Ajreg = Ajlocal[i];                                     \
  Axreg = Axlocal[i];                                     \
  if(Ajreg > 0)                                           \
  {                                                       \
    unsigned int idxi = Ajreg >> 16;                      \
    unsigned int idxj = Ajreg - (idxi << 16);             \
    atomicAdd(&s_Ax[idxi], Axreg * s_x[idxj]);            \
    atomicAdd(&s_Ax[idxj], Axreg * s_x[idxi]);            \
  }                                                       \
  else                                                     \
  {                                                       \
    goto gotolabel2;                                      \
  }                                                       \
}                                                         \

#define LOOP10_FUNC1() { FUNC1(0) FUNC1(1) FUNC1(2) FUNC1(3) FUNC1(4) FUNC1(5) FUNC1(6) FUNC1(7) FUNC1(8) FUNC1(9)}
#define LOOP20_FUNC1() {LOOP10_FUNC1() FUNC1(10) FUNC1(11) FUNC1(12) FUNC1(13) FUNC1(14) FUNC1(15) FUNC1(16) FUNC1(17) FUNC1(18) FUNC1(19)}
#define LOOP30_FUNC1() {LOOP20_FUNC1() FUNC1(20) FUNC1(21) FUNC1(22) FUNC1(23) FUNC1(24) FUNC1(25) FUNC1(26) FUNC1(27) FUNC1(28) FUNC1(29)}
#define LOOP40_FUNC1() {LOOP30_FUNC1() FUNC1(30) FUNC1(31) FUNC1(32) FUNC1(33) FUNC1(34) FUNC1(35) FUNC1(36) FUNC1(37) FUNC1(38) FUNC1(39)}
#define LOOP50_FUNC1() {LOOP40_FUNC1() FUNC1(40) FUNC1(41) FUNC1(42) FUNC1(43) FUNC1(44) FUNC1(45) FUNC1(46) FUNC1(47) FUNC1(48) FUNC1(49)}
#define LOOP60_FUNC1() {LOOP50_FUNC1() FUNC1(50) FUNC1(51) FUNC1(52) FUNC1(53) FUNC1(54) FUNC1(55) FUNC1(56) FUNC1(57) FUNC1(58) FUNC1(59)}
#define LOOP70_FUNC1() {LOOP60_FUNC1() FUNC1(60) FUNC1(61) FUNC1(62) FUNC1(63) FUNC1(64) FUNC1(65) FUNC1(66) FUNC1(67) FUNC1(68) FUNC1(69)}

#define LOOP10_FUNC2() { FUNC2(0) FUNC2(1) FUNC2(2) FUNC2(3) FUNC2(4) FUNC2(5) FUNC2(6) FUNC2(7) FUNC2(8) FUNC2(9)}
#define LOOP20_FUNC2() {LOOP10_FUNC2() FUNC2(10) FUNC2(11) FUNC2(12) FUNC2(13) FUNC2(14) FUNC2(15) FUNC2(16) FUNC2(17) FUNC2(18) FUNC2(19)}
#define LOOP30_FUNC2() {LOOP20_FUNC2() FUNC2(20) FUNC2(21) FUNC2(22) FUNC2(23) FUNC2(24) FUNC2(25) FUNC2(26) FUNC2(27) FUNC2(28) FUNC2(29)}
#define LOOP40_FUNC2() {LOOP30_FUNC2() FUNC2(30) FUNC2(31) FUNC2(32) FUNC2(33) FUNC2(34) FUNC2(35) FUNC2(36) FUNC2(37) FUNC2(38) FUNC2(39)}
#define LOOP50_FUNC2() {LOOP40_FUNC2() FUNC2(40) FUNC2(41) FUNC2(42) FUNC2(43) FUNC2(44) FUNC2(45) FUNC2(46) FUNC2(47) FUNC2(48) FUNC2(49)}
#define LOOP60_FUNC2() {LOOP50_FUNC2() FUNC2(50) FUNC2(51) FUNC2(52) FUNC2(53) FUNC2(54) FUNC2(55) FUNC2(56) FUNC2(57) FUNC2(58) FUNC2(59)}
#define LOOP70_FUNC2() {LOOP60_FUNC2() FUNC2(60) FUNC2(61) FUNC2(62) FUNC2(63) FUNC2(64) FUNC2(65) FUNC2(66) FUNC2(67) FUNC2(68) FUNC2(69)}

/***************************************
 * Source Definitions
 ***************************************/
template <class Matrix, class Vector>
gauss_seidel<Matrix, Vector>::gauss_seidel(AMG_Config &cfg, const Matrix_d& Ainit)
{
  cusp::detail::extract_diagonal(Ainit, this->diag);
  post_relaxes = cfg.AMG_Config::getParameter<int>("post_relaxes");
  weight = cfg.AMG_Config::getParameter<double>("smoother_weight");
  nPreInnerIter = cfg.AMG_Config::getParameter<int>("PreINNER_iters");
  nPostInnerIter = cfg.AMG_Config::getParameter<int>("PostINNER_iters");
  max_threads_per_block_ = cfg.AMG_Config::getParameter<int>("max_threads_per_block");
}

template<>
void gauss_seidel<Matrix_d, Vector_d>::find_diag(const Matrix_ell_d& A)
{
  typedef typename Matrix_d::index_type IndexType;
  typedef typename Matrix_d::value_type ValueType;

  const size_t THREADS_PER_BLOCK = max_threads_per_block_;
  const size_t NUM_BLOCKS = min(65535, (int)ceil((double)A.num_rows / (double)THREADS_PER_BLOCK));
  diag.resize(A.num_rows);

  find_diag_kernel<IndexType, ValueType> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >
    (A.num_rows, A.num_cols, A.column_indices.num_cols, A.column_indices.pitch,
     thrust::raw_pointer_cast(&A.column_indices.values[0]),
     thrust::raw_pointer_cast(&A.values.values[0]),
     thrust::raw_pointer_cast(&diag[0]));
}

template<typename IndexType, typename ValueType>
__global__ void GS_smooth_kernel(const IndexType num_rows,
    const IndexType * Ap,
    const IndexType * Aj,
    const ValueType * Ax,
    const ValueType * diag,
    const ValueType * b,
    const double weight,
    ValueType * x)

{
  IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for(int ridx = tidx; ridx < num_rows; ridx += blockDim.x * gridDim.x)
  {
    IndexType row_start = Ap[ridx];
    IndexType row_end = Ap[ridx + 1];
    ValueType Axi = 0.0;
    for(int j = row_start; j < row_end; j++)
    {
      Axi += Ax[j] * x[Aj[j]];
    }
    ValueType tmp = x[ridx] + weight * (b[ridx] - Axi) / diag[ridx];
    x[ridx] = tmp;
  }
}

template<>
void gauss_seidel<Matrix_d, Vector_d>::smooth(const Matrix_d &A, const Vector_d &b, Vector_d &x)
{
  if(diag.empty()) find_diag(A);

  typedef typename Matrix_d::index_type IndexType;
  typedef typename Matrix_d::value_type ValueType;

  const size_t THREADS_PER_BLOCK = max_threads_per_block_;
  const size_t NUM_BLOCKS = min(65535, (int)ceil((double)A.num_rows / (double)THREADS_PER_BLOCK));
  GS_smooth_kernel<IndexType, ValueType> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >
    (A.num_rows,
     thrust::raw_pointer_cast(&A.row_offsets[0]),
     thrust::raw_pointer_cast(&A.column_indices[0]),
     thrust::raw_pointer_cast(&A.values[0]),
     thrust::raw_pointer_cast(&diag[0]),
     thrust::raw_pointer_cast(&b[0]),
     weight,
     thrust::raw_pointer_cast(&x[0]));
}

template<typename IndexType, typename ValueType>
__global__ void permutation_kernel1(const int n, const IndexType* permutation, ValueType* x, ValueType* xout)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = thread_id; i < n; i += blockDim.x * gridDim.x)
    xout[i] = x[permutation[i]];
}

template<typename IndexType, typename ValueType>
__global__ void permutation_kernel2(const int n, const IndexType* permutation, ValueType* x, ValueType* xout)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = thread_id; i < n; i += blockDim.x * gridDim.x)
    xout[permutation[i]] = x[i];
}

template<typename IndexType, typename ValueType, int NUMPERROW>
__global__ void preRRCsr_kernel(const IndexType num_rows,
    const IndexType* offsets,
    const IndexType* Aj,
    const ValueType* Ax,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* residual,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  short Ajlocal[NUMPERROW];


  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];

  IndexType row = thread_id + blockstart;
  IndexType rowstart = offsets[row];
  IndexType rowend = offsets[row + 1];
  IndexType num_cols_per_row = rowend - rowstart;

  __shared__ ValueType s_x[1024];

  ValueType brow, drow;


  if(row < blockend)
  {
    brow = b[row];
    drow = diag[row];
    s_x[thread_id] = weight * brow / drow;

  }


  __syncthreads();

  if(row < blockend)
  {

    //load in matrix A to registers
#pragma unroll
    for(int n = 0; n < NUMPERROW; n++)
    {
      if(n < num_cols_per_row)
      {
        Axlocal[n] = Ax[rowstart + n];
        Ajlocal[n] = (short)(Aj[rowstart + n] - blockstart);
      }
    }
  }

  ValueType sum;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      //compute Ax
      sum = 0.0;
#pragma unroll
      for(int n = 0; n < NUMPERROW; n++)
      {
        if(n < num_cols_per_row)
        {
          sum += Axlocal[n] * s_x[Ajlocal[n]];
        }
      }
      s_x[thread_id] = s_x[thread_id] + weight * (brow - sum) / drow;
    }
    __syncthreads();
  }


  if(row < blockend)
  {
    //update glocal mem x values
    x[row] = s_x[thread_id];

    //compute Ax for residual
    sum = 0.0;
#pragma unroll
    for(unsigned short n = 0; n < NUMPERROW; n++)
    {
      if(n < num_cols_per_row)
      {
        sum += Axlocal[n] * s_x[Ajlocal[n]];
      }
    }

    //use s_x to temperarily store the residual*P
    residual[row] = brow - sum;
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW>
__global__ void preRRCsrShared_kernel(const IndexType num_rows,
    const IndexType* offsets,
    const IndexType* Aj,
    const ValueType* Ax,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* residual,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  extern __shared__ char s_mem[];
  ValueType* s_x = (ValueType*)s_mem;
  ushort* s_Ajlocal = (ushort*) & s_x[blockDim.x];



  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  const IndexType colidxstart = offsets[blockstart];

  IndexType row = thread_id + blockstart;
  IndexType rowstart = offsets[row];
  IndexType rowend = offsets[row + 1];
  IndexType num_cols_per_row = rowend - rowstart;

  ValueType brow, drow;

  if(row < blockend)
  {
    //load in matrix Aj to shared mem
    for(int n = 0; n < num_cols_per_row; n++)
    {
      s_Ajlocal[rowstart + n - colidxstart] = (short)(Aj[rowstart + n] - blockstart);
    }

    brow = b[row];
    drow = diag[row];
    s_x[thread_id] = weight * brow / drow;

    //load in matrix Ax to registers
#pragma unroll
    for(int n = 0; n < NUMPERROW; n++)
    {
      if(n < num_cols_per_row)
      {
        Axlocal[n] = Ax[rowstart + n];
      }
    }
  }
  __syncthreads();

  ValueType sum;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      //compute Ax
      sum = 0.0;
#pragma unroll
      for(int n = 0; n < NUMPERROW; n++)
      {
        if(n < num_cols_per_row)
        {
          sum += Axlocal[n] * s_x[s_Ajlocal[rowstart + n - colidxstart]];
        }
      }
      s_x[thread_id] = s_x[thread_id] + weight * (brow - sum) / drow;
    }
    __syncthreads();
  }


  if(row < blockend)
  {
    //update glocal mem x values
    x[row] = s_x[thread_id];

    //compute Ax for residual
    sum = 0.0;
#pragma unroll
    for(unsigned short n = 0; n < NUMPERROW; n++)
    {
      if(n < num_cols_per_row)
      {
        sum += Axlocal[n] * s_x[s_Ajlocal[rowstart + n - colidxstart]];
      }
    }

    //use s_x to temperarily store the residual*P
    residual[row] = brow - sum;
  }
}

template<>
void gauss_seidel<Matrix_d, Vector_d>::preRRRFullCsr(const cusp::csr_matrix<IndexType, ValueType, MemorySpace>& AinCsr,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutCoo,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::hyb_matrix<IndexType, ValueType, MemorySpace>& restrictor,
    const cusp::array1d<IndexType, MemorySpace>& permutation,
    cusp::array1d<ValueType, MemorySpace>& b,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& bc,
    int level_id,
    int largestblksize,
    int largestnumentries,
    int largestnumperrow)
{
  typedef typename Matrix_d::index_type IndexType;
  typedef typename Matrix_d::value_type ValueType;

  const size_t THREADS_PER_BLOCK = std::min(max_threads_per_block_,largestblksize);
  const size_t NUM_BLOCKS = partitionIdx.size() - 1; //(int)ceil((double)AinEll.num_rows / (double)THREADS_PER_BLOCK);
  if(NUM_BLOCKS > 65535)
    cout << "Block number larger than 65535!!" << endl;

  const size_t SHAREDSIZE = THREADS_PER_BLOCK * sizeof(ValueType) + largestnumentries * sizeof(ushort);
  const bool useshared = (SHAREDSIZE <= 48 * 1024);
  const size_t NUMPERROW = largestnumperrow;
  cusp::array1d<ValueType, MemorySpace> residual(x.size(), 0.0);
  cusp::array1d<ValueType, MemorySpace> bout(b.size());

  if(level_id != 0)
  {
    permutation_kernel1<IndexType, ValueType> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(b.size(), thrust::raw_pointer_cast(&permutation[0]), thrust::raw_pointer_cast(&b[0]), thrust::raw_pointer_cast(&bout[0]));
    b.swap(bout);
  }

  if(SHAREDSIZE <= 16 * 1024)
  {
    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  }
  else if(SHAREDSIZE <= 48 * 1024)
  {
    cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
  }

  if(NUMPERROW < 10)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 9 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 9 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }


    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 15)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 14 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 14 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 20)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 19 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 19 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 25)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 24 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 24 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else if(NUMPERROW < 30)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 29 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 29 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 35)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 34 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 34 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 40)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 39 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 39 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 45)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 44 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 44 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 50)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 49 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 49 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 55)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 54 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 54 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else if(NUMPERROW < 60)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 59 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 59 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else if(NUMPERROW < 65)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 64 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 64 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 70)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 69 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 69 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else if(NUMPERROW < 76)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 75 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 75 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 80)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 79 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 79 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else if(NUMPERROW < 86)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 85 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 85 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 221)
  {
    if(useshared)
    {
      preRRCsrShared_kernel<IndexType, ValueType, 220 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    else
    {
      preRRCsr_kernel<IndexType, ValueType, 220 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
          thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
          thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
          thrust::raw_pointer_cast(&AinCsr.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    }
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else
  {
    cout << "preRRRFullCsr num_per_row is equal or larger than 221!!" << endl;
    exit(0);
  }

  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

  cusp::array1d<ValueType, MemorySpace> Ax_buffer(x.size());
  cusp::multiply(AoutCoo, x, Ax_buffer);
  cusp::blas::axpby(residual, Ax_buffer, residual, ValueType(1.0), ValueType(-1.0));

  cusp::multiply(restrictor, residual, bc);
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE, int NUMITERS>
__global__ void preRR_kernel(const IndexType num_rows,
    const IndexType num_cols,
    const IndexType num_cols_per_row,
    const IndexType pitch,
    const IndexType* Aj,
    const ValueType* Ax,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* residual,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  short Ajlocal[NUMPERROW];
  const short invalid_index = cusp::ell_matrix<short, ValueType, cusp::device_memory>::invalid_index;

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];

  IndexType row = thread_id + blockstart;

  __shared__ ValueType s_x[SHAREDSIZE];

  ValueType brow, drow;
  IndexType tmpidx;

  if(row < blockend)
  {
    brow = b[row];
    drow = diag[row];
    if(fabs(drow) < 1e-9)
      printf("drow is zero!!");

    s_x[thread_id] = weight * brow / drow;
  }

  __syncthreads();

  if(row < blockend)
  {

    IndexType offset = row;

    //load in matrix A to registers
#pragma unroll
    for(int n = 0; n < NUMPERROW; n++)
    {
      if(n < num_cols_per_row)
      {
        Axlocal[n] = Ax[offset];
        Ajlocal[n] = invalid_index;
        if((tmpidx = Aj[offset]) != (IndexType)invalid_index) Ajlocal[n] = tmpidx - blockstart;
        offset += pitch;
      }
    }
  }


  ValueType sum;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      //compute Ax
      sum = 0.0;
#pragma unroll
      for(int n = 0; n < NUMPERROW; n++)
      {
        if(n < num_cols_per_row)
        {
          if(Ajlocal[n] != invalid_index)
          {
            sum += Axlocal[n] * s_x[Ajlocal[n]];
          }
        }
      }
      s_x[thread_id] = s_x[thread_id] + weight * (brow - sum) / drow;
    }
    __syncthreads();
  }


  if(row < blockend)
  {
    //update glocal mem x values
    x[row] = s_x[thread_id];

    //compute Ax for residual
    sum = 0.0;
#pragma unroll
    for(unsigned short n = 0; n < NUMPERROW; n++)
    {
      if(n < num_cols_per_row)
      {
        if(Ajlocal[n] != invalid_index)
        {
          sum += Axlocal[n] * s_x[Ajlocal[n]];
        }
      }
    }

    //use s_x to temperarily store the residual*P
    residual[row] = brow - sum;
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW>
__global__ void preRRShared_kernel(const IndexType num_rows,
    const IndexType num_cols,
    const IndexType num_cols_per_row,
    const IndexType pitch,
    const IndexType* Aj,
    const ValueType* Ax,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* residual,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  extern __shared__ char s_mem[];
  ValueType* s_x = (ValueType*)s_mem;
  short* s_Ajlocal = (short*)&s_x[blockDim.x];
  const short invalid_index = cusp::ell_matrix<short, ValueType, cusp::device_memory>::invalid_index;

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];

  IndexType row = thread_id + blockstart;

  ValueType brow, drow;
  IndexType tmpidx;

  if(row < blockend)
  {
    brow = b[row];
    drow = diag[row];
    if(fabs(drow) < 1e-9)
      printf("drow is zero!!");
    s_x[thread_id] = weight * brow / drow;
  }

  __syncthreads();

  if(row < blockend)
  {

    IndexType offset = row;

    //load in matrix A to registers
#pragma unroll
    for(int n = 0; n < NUMPERROW; n++)
    {
      if(n < num_cols_per_row)
      {
        IndexType Ajidx = thread_id * num_cols_per_row + n;
        Axlocal[n] = Ax[offset];
        s_Ajlocal[Ajidx] = invalid_index;
        if((tmpidx = Aj[offset]) != (IndexType)invalid_index) s_Ajlocal[Ajidx] = (short)(tmpidx - blockstart);
        offset += pitch;
      }
    }
  }


  ValueType sum;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      //compute Ax
      sum = 0.0;
#pragma unroll
      for(int n = 0; n < NUMPERROW; n++)
      {
        if(n < num_cols_per_row)
        {
          IndexType Ajidx = thread_id * num_cols_per_row + n;
          if(s_Ajlocal[Ajidx] != invalid_index)
          {
            sum += Axlocal[n] * s_x[s_Ajlocal[Ajidx]];
          }
        }
      }
      s_x[thread_id] = s_x[thread_id] + weight * (brow - sum) / drow;
    }
    __syncthreads();
  }


  if(row < blockend)
  {
    //update glocal mem x values
    x[row] = s_x[thread_id];

    //compute Ax for residual
    sum = 0.0;
#pragma unroll
    for(unsigned short n = 0; n < NUMPERROW; n++)
    {
      if(n < num_cols_per_row)
      {
        IndexType Ajidx = thread_id * num_cols_per_row + n;
        if(s_Ajlocal[Ajidx] != invalid_index)
        {
          sum += Axlocal[n] * s_x[s_Ajlocal[Ajidx]];
        }
      }
    }

    //use s_x to temperarily store the residual*P
    residual[row] = brow - sum;
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE, int NUMITERS>
__global__ void preRRSym_kernel1(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* __restrict__ Aj,
    const ValueType* __restrict__ Ax,
    const ValueType* __restrict__ diag,
    const IndexType* __restrict__ aggregateIdx,
    const IndexType* __restrict__ partitionIdx,
    const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ AinBlockIdx,
    const ValueType* __restrict__ b,
    const double weight,
    ValueType* __restrict__ x,
    ValueType* __restrict__ residual,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0}; // assuming that 0 is not valid means (0,0) is not in this array
  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  const IndexType cooblockstart = AinBlockIdx[blockIdx.x];
  const IndexType cooblockend = AinBlockIdx[blockIdx.x + 1];

  IndexType row = thread_id + blockstart;

  __shared__ ValueType s_x[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];
  //  __shared__ ValueType s_b[SHAREDSIZE] = {0};

  ValueType brow, drow;

  if(row < blockend)
  {
    brow = b[row];
    drow = diag[row];
    s_x[thread_id] = weight * brow / drow;
  }
  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = cooblockstart + thread_id + i * blockDim.x;
    if(entryidx < cooblockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;


  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();
    LOOP10_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values
    x[row] = s_x[thread_id];
  }

  if(row < blockend)
  {
    s_Ax[thread_id] = 0.0;
  }
  __syncthreads();

  LOOP10_FUNC2();

gotolabel2:
  __syncthreads();

  if(row < blockend)
  {
    residual[row] = brow - s_Ax[thread_id] - drow * s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE, int NUMITERS>
__global__ void preRRSym_kernel2(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* Aj,
    const ValueType* Ax,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const IndexType* AinBlockIdx,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* residual,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0}; // assuming that 0 is not valid means (0,0) is not in this array
  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  const IndexType cooblockstart = AinBlockIdx[blockIdx.x];
  const IndexType cooblockend = AinBlockIdx[blockIdx.x + 1];

  IndexType row = thread_id + blockstart;

  __shared__ ValueType s_x[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];

  ValueType brow, drow;

  if(row < blockend)
  {
    brow = b[row];
    drow = diag[row];
    s_x[thread_id] = weight * brow / drow;
  }
  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = cooblockstart + thread_id + i * blockDim.x;
    if(entryidx < cooblockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;


  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();
    LOOP20_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values
    x[row] = s_x[thread_id];
  }

  if(row < blockend)
  {
    s_Ax[thread_id] = 0.0;
  }
  __syncthreads();
  LOOP20_FUNC2();

gotolabel2:
  __syncthreads();

  if(row < blockend)
  {
    residual[row] = brow - s_Ax[thread_id] - drow * s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE, int NUMITERS>
__global__ void preRRSym_kernel3(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* Aj,
    const ValueType* Ax,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const IndexType* AinBlockIdx,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* residual,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0}; // assuming that 0 is not valid means (0,0) is not in this array
  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  const IndexType cooblockstart = AinBlockIdx[blockIdx.x];
  const IndexType cooblockend = AinBlockIdx[blockIdx.x + 1];

  IndexType row = thread_id + blockstart;

  __shared__ ValueType s_x[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];

  ValueType brow, drow;

  if(row < blockend)
  {
    brow = b[row];
    drow = diag[row];
    s_x[thread_id] = weight * brow / drow;
  }
  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = cooblockstart + thread_id + i * blockDim.x;
    if(entryidx < cooblockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;


  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();
    LOOP30_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values
    x[row] = s_x[thread_id];
  }

  if(row < blockend)
  {
    s_Ax[thread_id] = 0.0;
  }
  __syncthreads();

  LOOP30_FUNC2();

gotolabel2:
  __syncthreads();

  if(row < blockend)
  {
    residual[row] = brow - s_Ax[thread_id] - drow * s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE, int NUMITERS>
__global__ void preRRSym_kernel4(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* Aj,
    const ValueType* Ax,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const IndexType* AinBlockIdx,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* residual,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0}; // assuming that 0 is not valid means (0,0) is not in this array
  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  const IndexType cooblockstart = AinBlockIdx[blockIdx.x];
  const IndexType cooblockend = AinBlockIdx[blockIdx.x + 1];

  IndexType row = thread_id + blockstart;

  __shared__ ValueType s_x[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];

  ValueType brow, drow;

  if(row < blockend)
  {
    brow = b[row];
    drow = diag[row];
    s_x[thread_id] = weight * brow / drow;
  }
  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = cooblockstart + thread_id + i * blockDim.x;
    if(entryidx < cooblockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;


  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();
    LOOP40_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values
    x[row] = s_x[thread_id];
  }

  if(row < blockend)
  {
    s_Ax[thread_id] = 0.0;
  }
  __syncthreads();

  LOOP40_FUNC2();

gotolabel2:
  __syncthreads();

  if(row < blockend)
  {
    residual[row] = brow - s_Ax[thread_id] - drow * s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE, int NUMITERS>
__global__ void preRRSym_kernel5(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* Aj,
    const ValueType* Ax,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const IndexType* AinBlockIdx,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* residual,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0}; // assuming that 0 is not valid means (0,0) is not in this array
  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  const IndexType cooblockstart = AinBlockIdx[blockIdx.x];
  const IndexType cooblockend = AinBlockIdx[blockIdx.x + 1];

  IndexType row = thread_id + blockstart;

  __shared__ ValueType s_x[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];

  ValueType brow, drow;

  if(row < blockend)
  {
    brow = b[row];
    drow = diag[row];
    s_x[thread_id] = weight * brow / drow;
  }
  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = cooblockstart + thread_id + i * blockDim.x;
    if(entryidx < cooblockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;


  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();
    //compute Ax
    LOOP50_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values
    x[row] = s_x[thread_id];
  }

  if(row < blockend)
  {
    s_Ax[thread_id] = 0.0;
  }
  __syncthreads();
  //compute Ax for residual

  LOOP50_FUNC2();

gotolabel2:
  __syncthreads();

  if(row < blockend)
  {
    residual[row] = brow - s_Ax[thread_id] - drow * s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE, int NUMITERS>
__global__ void preRRSym_kernel6(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* Aj,
    const ValueType* Ax,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const IndexType* AinBlockIdx,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* residual,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0}; // assuming that 0 is not valid means (0,0) is not in this array
  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  const IndexType cooblockstart = AinBlockIdx[blockIdx.x];
  const IndexType cooblockend = AinBlockIdx[blockIdx.x + 1];

  IndexType row = thread_id + blockstart;

  __shared__ ValueType s_x[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];

  ValueType brow, drow;

  if(row < blockend)
  {
    brow = b[row];
    drow = diag[row];
    s_x[thread_id] = weight * brow / drow;
  }
  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = cooblockstart + thread_id + i * blockDim.x;
    if(entryidx < cooblockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;


  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();
    //compute Ax
    LOOP60_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values
    x[row] = s_x[thread_id];
  }

  if(row < blockend)
  {
    s_Ax[thread_id] = 0.0;
  }
  __syncthreads();
  //compute Ax for residual

  LOOP60_FUNC2();

gotolabel2:
  __syncthreads();

  if(row < blockend)
  {
    residual[row] = brow - s_Ax[thread_id] - drow * s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE, int NUMITERS>
__global__ void preRRSym_kernel7(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* Aj,
    const ValueType* Ax,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const IndexType* AinBlockIdx,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* residual,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0}; // assuming that 0 is not valid means (0,0) is not in this array
  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  const IndexType cooblockstart = AinBlockIdx[blockIdx.x];
  const IndexType cooblockend = AinBlockIdx[blockIdx.x + 1];

  IndexType row = thread_id + blockstart;

  __shared__ ValueType s_x[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];

  ValueType brow, drow;

  if(row < blockend)
  {
    brow = b[row];
    drow = diag[row];
    s_x[thread_id] = weight * brow / drow;
  }
  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = cooblockstart + thread_id + i * blockDim.x;
    if(entryidx < cooblockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;


  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();
    //compute Ax
    LOOP70_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values
    x[row] = s_x[thread_id];
  }

  if(row < blockend)
  {
    s_Ax[thread_id] = 0.0;
  }
  __syncthreads();
  //compute Ax for residual

  LOOP70_FUNC2();

gotolabel2:
  __syncthreads();

  if(row < blockend)
  {
    residual[row] = brow - s_Ax[thread_id] - drow * s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE, int NUMITERS>
__global__ void preAout_kernel(const IndexType num_rows,
    const IndexType num_entries,
    const ValueType* __restrict__ x,
    ValueType* __restrict__ r,
    const IndexType* __restrict__ Aouti,
    const IndexType* __restrict__ Aoutj,
    const ValueType* __restrict__ Aoutv,
    const IndexType* __restrict__ AoutBlockIdx,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx)
{

  __shared__ ValueType s_r[SHAREDSIZE];
  IndexType tid = threadIdx.x;
  s_r[tid] = 0.0;
  __syncthreads();
  //compute AoutX
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  IndexType AoutBlockstart = AoutBlockIdx[blockIdx.x];
  IndexType AoutBlockend = AoutBlockIdx[blockIdx.x + 1];
  for(int i = AoutBlockstart + tid; i < AoutBlockend; i += blockDim.x)
  {
    IndexType idxi = Aouti[i];
    IndexType idxj = Aoutj[i];
    ValueType v = Aoutv[i];
    atomicAdd(&s_r[idxi - blockstart], v * x[idxj]); //assuming ValueType is float
  }
  __syncthreads();

  if(tid < blockend - blockstart)
  {
    r[blockstart + tid] -= s_r[tid];
  }

}

template<>
void gauss_seidel<Matrix_d, Vector_d>::preRRRFullSymmetric(const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AinSysCoo,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutSysCoo,
    const cusp::array1d<IndexType, MemorySpace>& AinBlockIdx,
    const cusp::array1d<IndexType, MemorySpace>& AoutBlockIdx,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::hyb_matrix<IndexType, ValueType, MemorySpace>& restrictor,
    const cusp::array1d<IndexType, MemorySpace>& permutation,
    cusp::array1d<ValueType, MemorySpace>& b,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& bc,
    int level_id,
    int largestblksize,
    int largestnumentries)
{
  typedef typename Matrix_d::index_type IndexType;
  typedef typename Matrix_d::value_type ValueType;


  const size_t THREADS_PER_BLOCK = std::min(max_threads_per_block_,largestblksize);
  const size_t NUM_BLOCKS = partitionIdx.size() - 1;
  if(NUM_BLOCKS > 65535)
    cout << "Block number larger than 65535!!" << endl;

  const size_t num_entries_per_thread = ceil((double)largestnumentries / THREADS_PER_BLOCK);
  cout << "In preRRRFullSymmetric : ";
  cout << "THREADS_PER_BLOCK = " << THREADS_PER_BLOCK;
  cout << ", NUM_BLOCKS = " << NUM_BLOCKS;
  cout << endl;

  cusp::array1d<ValueType, MemorySpace> residual(x.size(), 0.0);
  cusp::array1d<ValueType, MemorySpace> bout(b.size());

  if(level_id != 0)
  {
    permutation_kernel1<IndexType, ValueType> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(b.size(), thrust::raw_pointer_cast(&permutation[0]), thrust::raw_pointer_cast(&b[0]), thrust::raw_pointer_cast(&bout[0]));
    b.swap(bout);
  }

  const int shared_size = 1024;
  if(largestblksize > shared_size)
  {
    cout << "largest block size is larger than shared size!!!" << endl;
    exit(0);
  }

  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

  if(num_entries_per_thread < 11)
  {
    preRRSym_kernel1<IndexType, ValueType, 10, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(num_entries_per_thread < 21)
  {
    preRRSym_kernel2<IndexType, ValueType, 20, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }

  else if(num_entries_per_thread < 31)
  {
    preRRSym_kernel3<IndexType, ValueType, 30, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }

  else if(num_entries_per_thread < 41)
  {
    preRRSym_kernel4<IndexType, ValueType, 40, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }

  else if(num_entries_per_thread < 51)
  {
    preRRSym_kernel5<IndexType, ValueType, 50, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }

  else if(num_entries_per_thread < 61)
  {
    preRRSym_kernel6<IndexType, ValueType, 60, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }

  else if(num_entries_per_thread < 71)
  {
    preRRSym_kernel7<IndexType, ValueType, 70, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }
  else if(num_entries_per_thread < 81)
  {
    preRRSym_kernel7<IndexType, ValueType, 80, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }
  else if(num_entries_per_thread < 91)
  {
    preRRSym_kernel7<IndexType, ValueType, 90, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }
  else
  {
    cout << "preRRRFullSymmetric num_entries_per_thread is larger than 90!!" << endl;
    exit(0);
  }

  preAout_kernel<IndexType, ValueType, 90, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (AoutSysCoo.num_rows,
      AoutSysCoo.num_entries,
      thrust::raw_pointer_cast(&x[0]),
      thrust::raw_pointer_cast(&residual[0]),
      thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
      thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
      thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
      thrust::raw_pointer_cast(&AoutBlockIdx[0]),
      thrust::raw_pointer_cast(&aggregateIdx[0]),
      thrust::raw_pointer_cast(&partitionIdx[0]));

  cusp::multiply(restrictor, residual, bc);
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE, int NUMITERS>
__global__ void preRRSymSync_kernel(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* __restrict__ Aj,
    const ValueType* __restrict__ Ax,
    const ValueType* __restrict__ diag,
    const IndexType* __restrict__ aggregateIdx,
    const IndexType* __restrict__ partitionIdx,
    const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ AinBlockIdx,
    const ValueType* __restrict__ b,
    const IndexType* __restrict__ segSyncIdx,
    const IndexType* __restrict__ partSyncIdx,
    const double weight,
    ValueType* __restrict__ x,
    ValueType* __restrict__ residual,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0}; // assuming that 0 is not valid means (0,0) is not in this array
  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  const IndexType cooblockstart = AinBlockIdx[blockIdx.x];
  const IndexType cooblockend = AinBlockIdx[blockIdx.x + 1];

  IndexType row = thread_id + blockstart;

  __shared__ ValueType s_x[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];

  ValueType brow, drow;

  if(row < blockend)
  {
    brow = b[row];
    drow = diag[row];
    s_x[thread_id] = weight * brow / drow;
  }
  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = cooblockstart + thread_id + i * blockDim.x;
    if(entryidx < cooblockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;
  unsigned int idxi;
  unsigned int idxj;
  IndexType partSyncStart = partSyncIdx[blockIdx.x];
  IndexType partSyncEnd = partSyncIdx[blockIdx.x + 1];
  IndexType nseg = partSyncEnd - partSyncStart;

  IndexType cooidx;
  int n;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();
    //compute Ax
    n = 0;
#pragma unroll
    for(int segIdx = 0; segIdx < nseg; segIdx++)
    {
      IndexType segSyncStart = segSyncIdx[partSyncStart + segIdx];
      IndexType segSyncEnd = segSyncIdx[partSyncStart + segIdx + 1];
      bool inside = false;

      cooidx = cooblockstart + n * blockDim.x + threadIdx.x;
      inside = (cooidx >= segSyncStart && cooidx < segSyncEnd);
      if(inside)
      {
        Ajreg = Ajlocal[n];
        Axreg = Axlocal[n];
        idxi = Ajreg >> 16;
        idxj = Ajreg - (idxi << 16);
        s_Ax[idxi] += Axreg * s_x[idxj];
      }

      __syncthreads();

      if(inside)
      {
        s_Ax[idxj] += Axreg * s_x[idxi];
        n++;
      }
      __syncthreads();
    }

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  __syncthreads();

  if(row < blockend)
  {
    //update glocal mem x values
    x[row] = s_x[thread_id];
  }

  if(row < blockend)
  {
    s_Ax[thread_id] = 0.0;
  }
  __syncthreads();
  //compute Ax for residual
  n = 0;
#pragma unroll
  for(int segIdx = 0; segIdx < nseg; segIdx++)
  {
    IndexType segSyncStart = segSyncIdx[partSyncStart + segIdx];
    IndexType segSyncEnd = segSyncIdx[partSyncStart + segIdx + 1];
    bool inside = false;

    cooidx = cooblockstart + n * blockDim.x + threadIdx.x;
    inside = (cooidx >= segSyncStart && cooidx < segSyncEnd);
    if(inside)
    {
      Ajreg = Ajlocal[n];
      Axreg = Axlocal[n];
      idxi = Ajreg >> 16;
      idxj = Ajreg - (idxi << 16);
      s_Ax[idxi] += Axreg * s_x[idxj];
    }

    __syncthreads();

    if(inside)
    {
      s_Ax[idxj] += Axreg * s_x[idxi];
      n++;
    }
    __syncthreads();
  }

  if(row < blockend)
  {

    residual[row] = brow - s_Ax[thread_id] - drow * s_x[thread_id];
  }


}

template<>
void gauss_seidel<Matrix_d, Vector_d>::preRRRFullSymmetricSync(const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AinSysCoo,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutSysCoo,
    const cusp::array1d<IndexType, MemorySpace>& AinBlockIdx,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::hyb_matrix<IndexType, ValueType, MemorySpace>& restrictor,
    const cusp::array1d<IndexType, MemorySpace>& permutation,
    cusp::array1d<ValueType, MemorySpace>& b,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& bc,
    const cusp::array1d<IndexType, MemorySpace>& segSyncIdx,
    const cusp::array1d<IndexType, MemorySpace>& partSyncIdx,
    int level_id,
    int largestblksize,
    int largestnumentries)
{
  typedef typename Matrix_d::index_type IndexType;
  typedef typename Matrix_d::value_type ValueType;


  const size_t THREADS_PER_BLOCK = std::min(max_threads_per_block_,largestblksize);
  const size_t NUM_BLOCKS = partitionIdx.size() - 1;
  if(NUM_BLOCKS > 65535)
    cout << "Block number larger than 65535!!" << endl;

  const size_t num_entries_per_thread = ceil((double)largestnumentries / THREADS_PER_BLOCK);

  cusp::array1d<ValueType, MemorySpace> residual(x.size(), 0.0);
  cusp::array1d<ValueType, MemorySpace> bout(b.size());

  if(level_id != 0)
  {
    permutation_kernel1<IndexType, ValueType> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(b.size(), thrust::raw_pointer_cast(&permutation[0]), thrust::raw_pointer_cast(&b[0]), thrust::raw_pointer_cast(&bout[0]));
    b.swap(bout);
  }

  const int shared_size = 1024;
  if(largestblksize > shared_size)
  {
    cout << "largest block size is larger than shared size!!!" << endl;
    exit(0);
  }

  if(num_entries_per_thread < 11)
  {
    preRRSymSync_kernel<IndexType, ValueType, 10, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        thrust::raw_pointer_cast(&segSyncIdx[0]),
        thrust::raw_pointer_cast(&partSyncIdx[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(num_entries_per_thread < 21)
  {
    preRRSymSync_kernel<IndexType, ValueType, 20, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        thrust::raw_pointer_cast(&segSyncIdx[0]),
        thrust::raw_pointer_cast(&partSyncIdx[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }
  else if(num_entries_per_thread < 31)
  {
    preRRSymSync_kernel<IndexType, ValueType, 30, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        thrust::raw_pointer_cast(&segSyncIdx[0]),
        thrust::raw_pointer_cast(&partSyncIdx[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }
  else if(num_entries_per_thread < 41)
  {
    preRRSymSync_kernel<IndexType, ValueType, 40, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        thrust::raw_pointer_cast(&segSyncIdx[0]),
        thrust::raw_pointer_cast(&partSyncIdx[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }
  else if(num_entries_per_thread < 51)
  {
    preRRSymSync_kernel<IndexType, ValueType, 50, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        thrust::raw_pointer_cast(&segSyncIdx[0]),
        thrust::raw_pointer_cast(&partSyncIdx[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }
  else if(num_entries_per_thread < 61)
  {
    preRRSymSync_kernel<IndexType, ValueType, 60, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        thrust::raw_pointer_cast(&segSyncIdx[0]),
        thrust::raw_pointer_cast(&partSyncIdx[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }
  else if(num_entries_per_thread < 71)
  {
    preRRSymSync_kernel<IndexType, ValueType, 70, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
        AinSysCoo.num_entries,
        thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
        thrust::raw_pointer_cast(&AinSysCoo.values[0]),
        thrust::raw_pointer_cast(&diag[0]),
        thrust::raw_pointer_cast(&aggregateIdx[0]),
        thrust::raw_pointer_cast(&partitionIdx[0]),
        thrust::raw_pointer_cast(&permutation[0]),
        thrust::raw_pointer_cast(&AinBlockIdx[0]),
        thrust::raw_pointer_cast(&b[0]),
        thrust::raw_pointer_cast(&segSyncIdx[0]),
        thrust::raw_pointer_cast(&partSyncIdx[0]),
        weight,
        thrust::raw_pointer_cast(&x[0]),
        thrust::raw_pointer_cast(&residual[0]),
        nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

  }

  else
  {

    cout << "preRRRFullSymmetricSync num_entries_per_thread is larger than 70!!" << endl;
    exit(0);
  }

  cusp::array1d<ValueType, MemorySpace> Ax_buffer(x.size());
  cusp::multiply(AoutSysCoo, x, Ax_buffer);
  cusp::blas::axpby(residual, Ax_buffer, residual, ValueType(1.0), ValueType(-1.0));
  cusp::multiply(restrictor, residual, bc);

}

template<>
void gauss_seidel<Matrix_d, Vector_d>::preRRRFull(const cusp::ell_matrix<IndexType, ValueType, MemorySpace>& AinEll,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutCoo,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::hyb_matrix<IndexType, ValueType, MemorySpace>& restrictor,
    const cusp::array1d<IndexType, MemorySpace>& permutation,
    cusp::array1d<ValueType, MemorySpace>& b,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& bc,
    int level_id,
    int largestblksize)
{
  typedef typename Matrix_d::index_type IndexType;
  typedef typename Matrix_d::value_type ValueType;

  const size_t THREADS_PER_BLOCK = std::min(max_threads_per_block_,largestblksize);
  const size_t NUM_BLOCKS = partitionIdx.size() - 1; 
  if(NUM_BLOCKS > 65535)
    cout << "Block number larger than 65535!!" << endl;

  const size_t NUMPERROW = AinEll.column_indices.num_cols;
  const size_t SHAREDSIZE = THREADS_PER_BLOCK * sizeof(ValueType) + NUMPERROW * THREADS_PER_BLOCK * sizeof(short);
  bool useshared = (SHAREDSIZE <= 48 * 1024);
  if(SHAREDSIZE <= 16 * 1024)
  {
    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  }
  else if(SHAREDSIZE <= 48 * 1024)
  {
    cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
  }

  cusp::array1d<ValueType, MemorySpace> residual(x.size(), 0.0);
  cusp::array1d<ValueType, MemorySpace> bout(b.size());

  if(level_id != 0)
  {
    permutation_kernel1<IndexType, ValueType> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(b.size(), thrust::raw_pointer_cast(&permutation[0]), thrust::raw_pointer_cast(&b[0]), thrust::raw_pointer_cast(&bout[0]));
    b.swap(bout);
  }

  const int shared_size = 1024;
  if(largestblksize > shared_size)
  {
    cout << "largest block size is larger than shared size!!!" << endl;
    exit(0);
  }
  if(NUMPERROW < 10)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 9 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);

    else
      preRR_kernel<IndexType, ValueType, 9, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 15)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 14 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 14, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 20)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 19 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 19, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 25)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 24 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 24, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else if(NUMPERROW < 30)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 29 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 29, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 35)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 34 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 34, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 40)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 39 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 39, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 45)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 44 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 44, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 50)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 49 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 49, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 55)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 54 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 54, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else if(NUMPERROW < 60)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 59 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 59, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else if(NUMPERROW < 65)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 64 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 64, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 70)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 69 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 69, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else if(NUMPERROW < 76)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 75 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 75, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
  else if(NUMPERROW < 80)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 79 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 79, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else if(NUMPERROW < 86)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 85 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 85, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else if(NUMPERROW < 221)
  {
    if(useshared)
      preRRShared_kernel<IndexType, ValueType, 220 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    else
      preRR_kernel<IndexType, ValueType, 220, shared_size, 10 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
          AinEll.num_cols,
          AinEll.column_indices.num_cols,
          AinEll.column_indices.pitch,
          thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
          thrust::raw_pointer_cast(&AinEll.values.values[0]),
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&permutation[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&residual[0]),
          nPreInnerIter);
    AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }

  else
  {
    cout << "preRRRFull num_per_row is equal or larger than 221!!" << endl;
    exit(0);
  }
  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

  cusp::array1d<ValueType, MemorySpace> Ax_buffer(x.size());
  cusp::multiply(AoutCoo, x, Ax_buffer);

  cusp::blas::axpby(residual, Ax_buffer, residual, ValueType(1.0), ValueType(-1.0));
  cusp::multiply(restrictor, residual, bc);
}

template<typename IndexType, typename ValueType, int NUMPERROW, int NUMITERS, int SHAREDSIZE>
__global__ void postPC_kernel(const IndexType num_rows,
    const IndexType num_cols,
    const IndexType num_cols_per_row,
    const IndexType pitch,
    const IndexType* Aj,
    const ValueType* Ax,
    const IndexType* Aouti,
    const IndexType* Aoutj,
    const ValueType* Aoutv,
    const IndexType num_entries,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const ValueType* p,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* xc)
{
  const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::device_memory>::invalid_index;

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];

  __shared__ ValueType s_array[SHAREDSIZE];
  ValueType* s_x = &s_array[0];
  ValueType* s_p = &s_array[SHAREDSIZE / 2];
  IndexType row = thread_id + blockstart;

  if(row < blockend)
  {
    s_p[thread_id] = p[row];
    s_x[thread_id] = x[row];
  }
  __syncthreads();


  //correction
  unsigned short num_aggregates = aggrend - aggrstart;
  if(thread_id < num_aggregates)
  {
    unsigned short vstart = aggregateIdx[aggrstart + thread_id];
    unsigned short vend = aggregateIdx[aggrstart + thread_id + 1];
    ValueType xctmp = xc[aggrstart + thread_id];
    for(int i = vstart; i < vend; i++)
    {
      s_x[i - blockstart] += xctmp * s_p[i - blockstart];
    }

  }

  __syncthreads();

  //write out the corrected x
  if(row < blockend)
  {

    x[row] = s_x[thread_id];

  }
}

template<typename IndexType, typename ValueType, int NUMPERROW>
__global__ void postRelax_kernel(const IndexType num_rows,
    const IndexType num_cols,
    const IndexType num_cols_per_row,
    const IndexType pitch,
    const IndexType* Aj,
    const ValueType* Ax,
    const IndexType* Aouti,
    const IndexType* Aoutj,
    const ValueType* Aoutv,
    const IndexType num_entries,
    const IndexType* AoutBlockIdx,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* xout,
    int nInnerIter)
{

  ValueType Axlocal[NUMPERROW];
  short Ajlocal[NUMPERROW];
  const short invalid_index = cusp::ell_matrix<short, ValueType, cusp::device_memory>::invalid_index;

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];

  __shared__ ValueType s_array[1024];
  ValueType* s_b = &s_array[0];
  IndexType row = thread_id + blockstart;
  IndexType tmpIdx;

  if(row < blockend)
  {
    s_b[thread_id] = b[row];
  }

  __syncthreads();

  ValueType brow, drow;
  if(row < blockend)
  {
    brow = s_b[thread_id];
    drow = diag[row];
  }

  //load x to shared memory
  ValueType* s_x = &s_array[0];
  if(row < blockend)
    s_x[thread_id] = x[row];

  __syncthreads();




  if(row < blockend)
  {

    IndexType offset = row;

    //load in matrix A to registers
#pragma unroll
    for(int n = 0; n < NUMPERROW; n++)
    {
      if(n < num_cols_per_row)
      {
        Axlocal[n] = Ax[offset];
        Ajlocal[n] = invalid_index;
        if((tmpIdx = Aj[offset]) != (IndexType)invalid_index) Ajlocal[n] = (short)(tmpIdx - blockstart);
        offset += pitch;
      }
    }
  }


  ValueType sum;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      //compute Ax
      sum = 0.0;
#pragma unroll
      for(unsigned short n = 0; n < NUMPERROW; n++)
      {
        if(n < num_cols_per_row)
        {
          if(Ajlocal[n] != invalid_index)
          {
            sum += Axlocal[n] * s_x[Ajlocal[n]];
          }
        }
      }
      s_x[thread_id] = s_x[thread_id] + weight * (brow - sum) / drow;
    }
    __syncthreads();
  }


  if(row < blockend)
  {
    //update glocal mem x values

    xout[row] = s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW>
__global__ void postRelaxShared_kernel(const IndexType num_rows,
    const IndexType num_cols,
    const IndexType num_cols_per_row,
    const IndexType pitch,
    const IndexType* Aj,
    const ValueType* Ax,
    const IndexType* Aouti,
    const IndexType* Aoutj,
    const ValueType* Aoutv,
    const IndexType num_entries,
    const IndexType* AoutBlockIdx,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* xout,
    int nInnerIter)
{

  extern char s_mem[];
  ValueType Axlocal[NUMPERROW];
  const short invalid_index = cusp::ell_matrix<short, ValueType, cusp::device_memory>::invalid_index;

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];

  ValueType* s_b = (ValueType*)s_mem;
  IndexType row = thread_id + blockstart;
  IndexType tmpIdx;

  if(row < blockend)
  {
    s_b[thread_id] = b[row];
  }

  __syncthreads();

  ValueType brow, drow;
  if(row < blockend)
  {
    brow = s_b[thread_id];
    drow = diag[row];
  }

  //load x to shared memory
  ValueType* s_x = (ValueType*)s_mem;
  if(row < blockend)
    s_x[thread_id] = x[row];

  __syncthreads();


  short* s_Ajlocal = (short*)&s_x[blockDim.x];


  if(row < blockend)
  {

    IndexType offset = row;

    //load in matrix A to registers
#pragma unroll
    for(int n = 0; n < NUMPERROW; n++)
    {
      if(n < num_cols_per_row)
      {
        IndexType Ajidx = thread_id * num_cols_per_row + n;
        Axlocal[n] = Ax[offset];
        s_Ajlocal[Ajidx] = invalid_index;
        if((tmpIdx = Aj[offset]) != (IndexType)invalid_index) s_Ajlocal[Ajidx] = (short)(tmpIdx - blockstart);
        offset += pitch;
      }
    }
  }


  ValueType sum;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      //compute Ax
      sum = 0.0;
#pragma unroll
      for(unsigned short n = 0; n < NUMPERROW; n++)
      {
        if(n < num_cols_per_row)
        {
          IndexType Ajidx = thread_id * num_cols_per_row + n;
          if(s_Ajlocal[Ajidx] != invalid_index)
          {
            sum += Axlocal[n] * s_x[s_Ajlocal[Ajidx]];
          }
        }
      }
      s_x[thread_id] = s_x[thread_id] + weight * (brow - sum) / drow;
    }
    __syncthreads();
  }


  if(row < blockend)
  {
    //update glocal mem x values

    xout[row] = s_x[thread_id];
  }
}

template<>
void gauss_seidel<Matrix_d, Vector_d>::postPCR(const cusp::ell_matrix<IndexType, ValueType, MemorySpace>& AinEll,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutCoo,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::array1d<ValueType, MemorySpace>& P,
    const cusp::array1d<ValueType, MemorySpace>& b,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& xc)
{
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE>
__global__ void postRelaxSym_kernel1(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* __restrict__ Aj,
    const ValueType* __restrict__ Ax,
    const IndexType* __restrict__ Aouti,
    const IndexType* __restrict__ Aoutj,
    const ValueType* __restrict__ Aoutv,
    const IndexType Aout_num_entries,
    const ValueType* __restrict__ diag,
    const IndexType* __restrict__ aggregateIdx,
    const IndexType* __restrict__ partitionIdx,
    const IndexType* __restrict__ AinBlockIdx,
    const IndexType* __restrict__ AoutBlockIdx,
    const ValueType* __restrict__ b,
    const double weight,
    ValueType* __restrict__ x,
    ValueType* __restrict__ xout,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0};

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  IndexType AoutBlockstart = AoutBlockIdx[blockIdx.x];
  IndexType AoutBlockend = AoutBlockIdx[blockIdx.x + 1];
  IndexType AinBlockstart = AinBlockIdx[blockIdx.x];
  IndexType AinBlockend = AinBlockIdx[blockIdx.x + 1];

  __shared__ ValueType s_array[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];
  ValueType* s_b = &s_array[0];
  IndexType row = thread_id + blockstart;

  if(row < blockend)
  {
    s_b[thread_id] = b[row];
  }

  __syncthreads();
  //add values to b out of this block

  for(int i = AoutBlockstart + thread_id; i < AoutBlockend; i += blockDim.x)
  {
    IndexType idxi = Aouti[i];
    IndexType idxj = Aoutj[i];
    ValueType v = Aoutv[i];
    atomicAdd(&s_b[idxi - blockstart], -v * x[idxj]); //assuming ValueType is float
  }
  __syncthreads();

  ValueType brow, drow;
  if(row < blockend)
  {
    brow = s_b[thread_id];
    drow = diag[row];
  }

  //load x to shared memory
  ValueType* s_x = &s_array[0];
  if(row < blockend)
    s_x[thread_id] = x[row];

  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = AinBlockstart + thread_id + i * blockDim.x;
    if(entryidx < AinBlockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();

    LOOP10_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values

    xout[row] = s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE>
__global__ void postRelaxSym_kernel2(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* Aj,
    const ValueType* Ax,
    const IndexType* Aouti,
    const IndexType* Aoutj,
    const ValueType* Aoutv,
    const IndexType Aout_num_entries,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* AinBlockIdx,
    const IndexType* AoutBlockIdx,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* xout,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0};

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  IndexType AoutBlockstart = AoutBlockIdx[blockIdx.x];
  IndexType AoutBlockend = AoutBlockIdx[blockIdx.x + 1];
  IndexType AinBlockstart = AinBlockIdx[blockIdx.x];
  IndexType AinBlockend = AinBlockIdx[blockIdx.x + 1];

  __shared__ ValueType s_array[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];
  ValueType* s_b = &s_array[0];
  IndexType row = thread_id + blockstart;

  if(row < blockend)
  {
    s_b[thread_id] = b[row];
  }

  __syncthreads();
  //add values to b out of this block
  for(int i = AoutBlockstart + thread_id; i < AoutBlockend; i += blockDim.x)
  {
    IndexType idxi = Aouti[i];
    IndexType idxj = Aoutj[i];
    ValueType v = Aoutv[i];
    atomicAdd(&s_b[idxi - blockstart], -v * x[idxj]); //assuming ValueType is float
  }
  __syncthreads();

  ValueType brow, drow;
  if(row < blockend)
  {
    brow = s_b[thread_id];
    drow = diag[row];
  }

  //load x to shared memory
  ValueType* s_x = &s_array[0];
  if(row < blockend)
    s_x[thread_id] = x[row];

  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = AinBlockstart + thread_id + i * blockDim.x;
    if(entryidx < AinBlockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();

    LOOP20_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values

    xout[row] = s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE>
__global__ void postRelaxSym_kernel3(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* Aj,
    const ValueType* Ax,
    const IndexType* Aouti,
    const IndexType* Aoutj,
    const ValueType* Aoutv,
    const IndexType Aout_num_entries,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* AinBlockIdx,
    const IndexType* AoutBlockIdx,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* xout,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0};

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  IndexType AoutBlockstart = AoutBlockIdx[blockIdx.x];
  IndexType AoutBlockend = AoutBlockIdx[blockIdx.x + 1];
  IndexType AinBlockstart = AinBlockIdx[blockIdx.x];
  IndexType AinBlockend = AinBlockIdx[blockIdx.x + 1];

  __shared__ ValueType s_array[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];
  ValueType* s_b = &s_array[0];
  IndexType row = thread_id + blockstart;

  if(row < blockend)
  {
    s_b[thread_id] = b[row];
  }

  __syncthreads();
  //add values to b out of this block

  for(int i = AoutBlockstart + thread_id; i < AoutBlockend; i += blockDim.x)
  {
    IndexType idxi = Aouti[i];
    IndexType idxj = Aoutj[i];
    ValueType v = Aoutv[i];
    atomicAdd(&s_b[idxi - blockstart], -v * x[idxj]); //assuming ValueType is float
  }
  __syncthreads();

  ValueType brow, drow;
  if(row < blockend)
  {
    brow = s_b[thread_id];
    drow = diag[row];
  }

  //load x to shared memory
  ValueType* s_x = &s_array[0];
  if(row < blockend)
    s_x[thread_id] = x[row];

  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = AinBlockstart + thread_id + i * blockDim.x;
    if(entryidx < AinBlockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();

    LOOP30_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values

    xout[row] = s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE>
__global__ void postRelaxSym_kernel4(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* Aj,
    const ValueType* Ax,
    const IndexType* Aouti,
    const IndexType* Aoutj,
    const ValueType* Aoutv,
    const IndexType Aout_num_entries,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* AinBlockIdx,
    const IndexType* AoutBlockIdx,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* xout,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0};

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  IndexType AoutBlockstart = AoutBlockIdx[blockIdx.x];
  IndexType AoutBlockend = AoutBlockIdx[blockIdx.x + 1];
  IndexType AinBlockstart = AinBlockIdx[blockIdx.x];
  IndexType AinBlockend = AinBlockIdx[blockIdx.x + 1];

  __shared__ ValueType s_array[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];
  ValueType* s_b = &s_array[0];
  IndexType row = thread_id + blockstart;

  if(row < blockend)
  {
    s_b[thread_id] = b[row];
  }

  __syncthreads();
  //add values to b out of this block
  //
  for(int i = AoutBlockstart + thread_id; i < AoutBlockend; i += blockDim.x)
  {
    IndexType idxi = Aouti[i];
    IndexType idxj = Aoutj[i];
    ValueType v = Aoutv[i];
    atomicAdd(&s_b[idxi - blockstart], -v * x[idxj]); //assuming ValueType is float
  }
  __syncthreads();

  ValueType brow, drow;
  if(row < blockend)
  {
    brow = s_b[thread_id];
    drow = diag[row];
  }

  //load x to shared memory
  ValueType* s_x = &s_array[0];
  if(row < blockend)
    s_x[thread_id] = x[row];

  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = AinBlockstart + thread_id + i * blockDim.x;
    if(entryidx < AinBlockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();

    LOOP40_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values

    xout[row] = s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE>
__global__ void postRelaxSym_kernel5(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* Aj,
    const ValueType* Ax,
    const IndexType* Aouti,
    const IndexType* Aoutj,
    const ValueType* Aoutv,
    const IndexType Aout_num_entries,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* AinBlockIdx,
    const IndexType* AoutBlockIdx,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* xout,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0};

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  IndexType AoutBlockstart = AoutBlockIdx[blockIdx.x];
  IndexType AoutBlockend = AoutBlockIdx[blockIdx.x + 1];
  IndexType AinBlockstart = AinBlockIdx[blockIdx.x];
  IndexType AinBlockend = AinBlockIdx[blockIdx.x + 1];

  __shared__ ValueType s_array[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];
  ValueType* s_b = &s_array[0];
  IndexType row = thread_id + blockstart;

  if(row < blockend)
  {
    s_b[thread_id] = b[row];
  }

  __syncthreads();
  //add values to b out of this block

  for(int i = AoutBlockstart + thread_id; i < AoutBlockend; i += blockDim.x)
  {
    IndexType idxi = Aouti[i];
    IndexType idxj = Aoutj[i];
    ValueType v = Aoutv[i];
    atomicAdd(&s_b[idxi - blockstart], -v * x[idxj]); //assuming ValueType is float
  }
  __syncthreads();

  ValueType brow, drow;
  if(row < blockend)
  {
    brow = s_b[thread_id];
    drow = diag[row];
  }

  //load x to shared memory
  ValueType* s_x = &s_array[0];
  if(row < blockend)
    s_x[thread_id] = x[row];

  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = AinBlockstart + thread_id + i * blockDim.x;
    if(entryidx < AinBlockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();

    LOOP50_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values

    xout[row] = s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE>
__global__ void postRelaxSym_kernel6(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* Aj,
    const ValueType* Ax,
    const IndexType* Aouti,
    const IndexType* Aoutj,
    const ValueType* Aoutv,
    const IndexType Aout_num_entries,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* AinBlockIdx,
    const IndexType* AoutBlockIdx,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* xout,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0};

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  IndexType AoutBlockstart = AoutBlockIdx[blockIdx.x];
  IndexType AoutBlockend = AoutBlockIdx[blockIdx.x + 1];
  IndexType AinBlockstart = AinBlockIdx[blockIdx.x];
  IndexType AinBlockend = AinBlockIdx[blockIdx.x + 1];

  __shared__ ValueType s_array[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];
  ValueType* s_b = &s_array[0];
  IndexType row = thread_id + blockstart;

  if(row < blockend)
  {
    s_b[thread_id] = b[row];
  }

  __syncthreads();
  //add values to b out of this block
  for(int i = AoutBlockstart + thread_id; i < AoutBlockend; i += blockDim.x)
  {
    IndexType idxi = Aouti[i];
    IndexType idxj = Aoutj[i];
    ValueType v = Aoutv[i];
    atomicAdd(&s_b[idxi - blockstart], -v * x[idxj]); //assuming ValueType is float
  }
  __syncthreads();

  ValueType brow, drow;
  if(row < blockend)
  {
    brow = s_b[thread_id];
    drow = diag[row];
  }

  //load x to shared memory
  ValueType* s_x = &s_array[0];
  if(row < blockend)
    s_x[thread_id] = x[row];

  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = AinBlockstart + thread_id + i * blockDim.x;
    if(entryidx < AinBlockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();

    LOOP60_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values

    xout[row] = s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE>
__global__ void postRelaxSym_kernel7(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* Aj,
    const ValueType* Ax,
    const IndexType* Aouti,
    const IndexType* Aoutj,
    const ValueType* Aoutv,
    const IndexType Aout_num_entries,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* AinBlockIdx,
    const IndexType* AoutBlockIdx,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* xout,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0};

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  IndexType AoutBlockstart = AoutBlockIdx[blockIdx.x];
  IndexType AoutBlockend = AoutBlockIdx[blockIdx.x + 1];
  IndexType AinBlockstart = AinBlockIdx[blockIdx.x];
  IndexType AinBlockend = AinBlockIdx[blockIdx.x + 1];

  __shared__ ValueType s_array[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];
  ValueType* s_b = &s_array[0];
  IndexType row = thread_id + blockstart;

  if(row < blockend)
  {
    s_b[thread_id] = b[row];
  }

  __syncthreads();
  //add values to b out of this block
  //
  for(int i = AoutBlockstart + thread_id; i < AoutBlockend; i += blockDim.x)
  {
    IndexType idxi = Aouti[i];
    IndexType idxj = Aoutj[i];
    ValueType v = Aoutv[i];
    atomicAdd(&s_b[idxi - blockstart], -v * x[idxj]); //assuming ValueType is float
  }
  __syncthreads();

  ValueType brow, drow;
  if(row < blockend)
  {
    brow = s_b[thread_id];
    drow = diag[row];
  }

  //load x to shared memory
  ValueType* s_x = &s_array[0];
  if(row < blockend)
    s_x[thread_id] = x[row];

  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = AinBlockstart + thread_id + i * blockDim.x;
    if(entryidx < AinBlockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();

    LOOP70_FUNC1();

gotolabel:
    __syncthreads();

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  if(row < blockend)
  {
    //update glocal mem x values

    xout[row] = s_x[thread_id];
  }
}

template<>
void gauss_seidel<Matrix_d, Vector_d>::postPCRFullSymmetric(const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AinSysCoo,
    const cusp::array1d<IndexType, MemorySpace>& AinBlockIdx,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutSysCoo,
    const cusp::array1d<IndexType, MemorySpace>& AoutBlockIdx,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::hyb_matrix<IndexType, ValueType, MemorySpace>& prolongator,
    const cusp::array1d<IndexType, MemorySpace>& permutation,
    const cusp::array1d<ValueType, MemorySpace>& b,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& xc,
    int level_id,
    int largestblksize,
    int largestnumentries)
{

  Vector_d deltax(x.size());
  cusp::multiply(prolongator, xc, deltax); // e = P * x
  cusp::blas::axpby(x, deltax, x, ValueType(1.0), ValueType(1.0)); // x = x + e

  const size_t THREADS_PER_BLOCK = std::min(max_threads_per_block_,largestblksize);
  const size_t NUM_BLOCKS = partitionIdx.size() - 1;
  const size_t num_entries_per_thread = ceil((double)largestnumentries / THREADS_PER_BLOCK);
  if(NUM_BLOCKS > 65535)
    cout << "Block number larger than 65535!!" << endl;

  const size_t shared_size = 1024;

  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

  Vector_d xout(x.size());
  for(int i = 0; i < post_relaxes; i++)
  {
    if(num_entries_per_thread < 11)
    {
      postRelaxSym_kernel1<IndexType, ValueType, 10, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(num_entries_per_thread < 21)
    {
      postRelaxSym_kernel2<IndexType, ValueType, 20, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);

      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(num_entries_per_thread < 31)
    {
      postRelaxSym_kernel3<IndexType, ValueType, 30, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }

    else if(num_entries_per_thread < 41)
    {
      postRelaxSym_kernel4<IndexType, ValueType, 40, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(num_entries_per_thread < 51)
    {
      postRelaxSym_kernel5<IndexType, ValueType, 50, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(num_entries_per_thread < 61)
    {
      postRelaxSym_kernel6<IndexType, ValueType, 60, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }

    else if(num_entries_per_thread < 71)
    {
      postRelaxSym_kernel7<IndexType, ValueType, 70, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }

    else if(num_entries_per_thread < 81)
    {
      postRelaxSym_kernel7<IndexType, ValueType, 80, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }

    else if(num_entries_per_thread < 91)
    {
      postRelaxSym_kernel7<IndexType, ValueType, 90, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }

    else
    {
      cout << "In posePCRFull num_per_row larger than 90!!" << endl;
      exit(0);
    }
    x.swap(xout);

  }

  if(level_id != 0)
  {

    permutation_kernel2<IndexType, ValueType> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(x.size(), thrust::raw_pointer_cast(&permutation[0]), thrust::raw_pointer_cast(&x[0]), thrust::raw_pointer_cast(&xout[0]));
    x.swap(xout);
  }

}

template<typename IndexType, typename ValueType, int NUMPERROW, int SHAREDSIZE>
__global__ void postRelaxSymSync_kernel(const IndexType num_rows,
    const IndexType num_entries,
    const IndexType* __restrict__ Aj,
    const ValueType* __restrict__ Ax,
    const IndexType* __restrict__ Aouti,
    const IndexType* __restrict__ Aoutj,
    const ValueType* __restrict__ Aoutv,
    const IndexType Aout_num_entries,
    const ValueType* __restrict__ diag,
    const IndexType* __restrict__ aggregateIdx,
    const IndexType* __restrict__ partitionIdx,
    const IndexType* __restrict__ AinBlockIdx,
    const IndexType* __restrict__ AoutBlockIdx,
    const ValueType* __restrict__ b,
    const IndexType* __restrict__ segSyncIdx,
    const IndexType* __restrict__ partSyncIdx,
    const double weight,
    ValueType* __restrict__ x,
    ValueType* __restrict__ xout,
    int nInnerIter)
{
  ValueType Axlocal[NUMPERROW];
  unsigned int Ajlocal[NUMPERROW] = {0};

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];
  IndexType AinBlockstart = AinBlockIdx[blockIdx.x];
  IndexType AinBlockend = AinBlockIdx[blockIdx.x + 1];

  __shared__ ValueType s_array[SHAREDSIZE];
  __shared__ ValueType s_Ax[SHAREDSIZE];
  ValueType* s_b = &s_array[0];
  IndexType row = thread_id + blockstart;

  if(row < blockend)
  {
    s_b[thread_id] = b[row];
  }

  __syncthreads();

  ValueType brow, drow;
  if(row < blockend)
  {
    brow = s_b[thread_id];
    drow = diag[row];
  }

  //load x to shared memory
  ValueType* s_x = &s_array[0];
  if(row < blockend)
    s_x[thread_id] = x[row];

  __syncthreads();

#pragma unroll
  for(int i = 0; i < NUMPERROW; i++)
  {
    int entryidx = AinBlockstart + thread_id + i * blockDim.x;
    if(entryidx < AinBlockend)
    {
      Ajlocal[i] = Aj[entryidx];
      Axlocal[i] = Ax[entryidx];
    }
  }

  unsigned int Ajreg;
  ValueType Axreg;
  unsigned int idxi;
  unsigned int idxj;
  IndexType partSyncStart = partSyncIdx[blockIdx.x];
  IndexType partSyncEnd = partSyncIdx[blockIdx.x + 1];
  IndexType nseg = partSyncEnd - partSyncStart;
  IndexType cooidx;
  int n;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      s_Ax[thread_id] = 0.0;
    }
    __syncthreads();
    //compute Ax
    n = 0;
#pragma unroll
    for(int segIdx = 0; segIdx < nseg; segIdx++)
    {
      IndexType segSyncStart = segSyncIdx[partSyncStart + segIdx];
      IndexType segSyncEnd = segSyncIdx[partSyncStart + segIdx + 1];
      bool inside = false;
      cooidx = AinBlockstart + n * blockDim.x + threadIdx.x;
      inside = (cooidx >= segSyncStart && cooidx < segSyncEnd);
      if(inside)
      {
        Ajreg = Ajlocal[n];
        Axreg = Axlocal[n];
        idxi = Ajreg >> 16;
        idxj = Ajreg - (idxi << 16);
        s_Ax[idxi] += Axreg * s_x[idxj];
      }

      __syncthreads();

      if(inside)
      {
        s_Ax[idxj] += Axreg * s_x[idxi];
        n++;
      }
      __syncthreads();

    }

    if(row < blockend)
    {
      s_x[thread_id] += weight * (brow - s_Ax[thread_id] - drow * s_x[thread_id]) / drow;
    }
  }

  __syncthreads();

  if(row < blockend)
  {
    //update glocal mem x values

    xout[row] = s_x[thread_id];
  }
}

template<>
void gauss_seidel<Matrix_d, Vector_d>::postPCRFullSymmetricSync(const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AinSysCoo,
    const cusp::array1d<IndexType, MemorySpace>& AinBlockIdx,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutSysCoo,
    const cusp::array1d<IndexType, MemorySpace>& AoutBlockIdx,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::hyb_matrix<IndexType, ValueType, MemorySpace>& prolongator,
    const cusp::array1d<IndexType, MemorySpace>& permutation,
    const cusp::array1d<ValueType, MemorySpace>& origb,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& xc,
    const cusp::array1d<IndexType, MemorySpace>& segSyncIdx,
    const cusp::array1d<IndexType, MemorySpace>& partSyncIdx,
    int level_id,
    int largestblksize,
    int largestnumentries)
{
  Vector_d deltax(x.size());
  cusp::multiply(prolongator, xc, deltax);
  cusp::blas::axpby(x, deltax, x, ValueType(1.0), ValueType(1.0));
  cusp::multiply(AoutSysCoo, x, deltax); // b' = Aout * x
  cusp::array1d<ValueType, MemorySpace> b(x.size());
  cusp::blas::axpby(origb, deltax, b, ValueType(1.0), ValueType(-1.0)); // b = b - b'

  const size_t THREADS_PER_BLOCK = std::min(max_threads_per_block_,largestblksize);
  const size_t NUM_BLOCKS = partitionIdx.size() - 1;
  const size_t num_entries_per_thread = ceil((double)largestnumentries / THREADS_PER_BLOCK);
  if(NUM_BLOCKS > 65535)
    cout << "Block number larger than 65535!!" << endl;

  const size_t shared_size = 1024;

  Vector_d xout(x.size());
  for(int i = 0; i < post_relaxes; i++)
  {
    if(num_entries_per_thread < 11)
    {
      postRelaxSymSync_kernel<IndexType, ValueType, 10, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          thrust::raw_pointer_cast(&segSyncIdx[0]),
          thrust::raw_pointer_cast(&partSyncIdx[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(num_entries_per_thread < 21)
    {
      postRelaxSymSync_kernel<IndexType, ValueType, 20, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          thrust::raw_pointer_cast(&segSyncIdx[0]),
          thrust::raw_pointer_cast(&partSyncIdx[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

    }
    else if(num_entries_per_thread < 31)
    {
      postRelaxSymSync_kernel<IndexType, ValueType, 30, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          thrust::raw_pointer_cast(&segSyncIdx[0]),
          thrust::raw_pointer_cast(&partSyncIdx[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

    }
    else if(num_entries_per_thread < 41)
    {
      postRelaxSymSync_kernel<IndexType, ValueType, 40, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          thrust::raw_pointer_cast(&segSyncIdx[0]),
          thrust::raw_pointer_cast(&partSyncIdx[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

    }
    else if(num_entries_per_thread < 51)
    {
      postRelaxSymSync_kernel<IndexType, ValueType, 50, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          thrust::raw_pointer_cast(&segSyncIdx[0]),
          thrust::raw_pointer_cast(&partSyncIdx[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

    }
    else if(num_entries_per_thread < 61)
    {
      postRelaxSymSync_kernel<IndexType, ValueType, 60, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          thrust::raw_pointer_cast(&segSyncIdx[0]),
          thrust::raw_pointer_cast(&partSyncIdx[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

    }
    else if(num_entries_per_thread < 71)
    {
      postRelaxSymSync_kernel<IndexType, ValueType, 70, shared_size> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinSysCoo.num_rows,
          AinSysCoo.num_entries,
          thrust::raw_pointer_cast(&AinSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AinSysCoo.values[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.row_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.column_indices[0]),
          thrust::raw_pointer_cast(&AoutSysCoo.values[0]),
          AoutSysCoo.num_entries,
          thrust::raw_pointer_cast(&diag[0]),
          thrust::raw_pointer_cast(&aggregateIdx[0]),
          thrust::raw_pointer_cast(&partitionIdx[0]),
          thrust::raw_pointer_cast(&AinBlockIdx[0]),
          thrust::raw_pointer_cast(&AoutBlockIdx[0]),
          thrust::raw_pointer_cast(&b[0]),
          thrust::raw_pointer_cast(&segSyncIdx[0]),
          thrust::raw_pointer_cast(&partSyncIdx[0]),
          weight,
          thrust::raw_pointer_cast(&x[0]),
          thrust::raw_pointer_cast(&xout[0]),
          nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

    }
    else
    {
      cout << "In posePCRFullSymmetricSync num_per_row larger than 70!!" << endl;
      exit(0);
    }


    x.swap(xout);
  }

  if(level_id != 0)
  {

    permutation_kernel2<IndexType, ValueType> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(x.size(), thrust::raw_pointer_cast(&permutation[0]), thrust::raw_pointer_cast(&x[0]), thrust::raw_pointer_cast(&xout[0]));
    x.swap(xout);
  }



}

template<>
void gauss_seidel<Matrix_d, Vector_d>::postPCRFull(const cusp::ell_matrix<IndexType, ValueType, MemorySpace>& AinEll,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutCoo,
    const cusp::array1d<IndexType, MemorySpace>& AoutBlockIdx,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::hyb_matrix<IndexType, ValueType, MemorySpace>& prolongator,
    const cusp::array1d<IndexType, MemorySpace>& permutation,
    const cusp::array1d<ValueType, MemorySpace>& origb,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& xc,
    int level_id,
    int largestblksize)
{
  Vector_d deltax(x.size());
  cusp::multiply(prolongator, xc, deltax);
  cusp::blas::axpby(x, deltax, x, ValueType(1.0), ValueType(1.0));

  cusp::multiply(AoutCoo, x, deltax); // b' = Aout * x
  cusp::array1d<ValueType, MemorySpace> b(x.size());
  cusp::blas::axpby(origb, deltax, b, ValueType(1.0), ValueType(-1.0)); // b = b - b'

  const size_t THREADS_PER_BLOCK = std::min(max_threads_per_block_,largestblksize);
  const size_t NUM_BLOCKS = partitionIdx.size() - 1; 
  if(NUM_BLOCKS > 65535)
    cout << "Block number larger than 65535!!" << endl;

  const size_t NUMPERROW = AinEll.column_indices.num_cols;
  const size_t SHAREDSIZE = THREADS_PER_BLOCK * sizeof(ValueType) + NUMPERROW * THREADS_PER_BLOCK * sizeof(short);
  bool useshared = (SHAREDSIZE <= 48 * 1024);
  if(SHAREDSIZE <= 16 * 1024)
  {
    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  }
  else if(SHAREDSIZE <= 48 * 1024)
  {
    cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
  }



  Vector_d xout(x.size());

  for(int i = 0; i < post_relaxes; i++)
  {
    if(NUMPERROW < 10)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 9 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 9 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 15)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 14 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 14 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 20)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 19 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 19 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 25)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 24 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 24 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 30)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 29 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 29 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 35)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 34 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 34 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 40)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 39 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 39 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 45)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 44 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 44 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 50)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 49 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 49 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 55)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 54 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 54 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 60)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 59 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 59 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 65)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 64 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 64 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 70)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 69 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 69 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }

    else if(NUMPERROW < 76)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 75 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 75 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 80)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 79 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 79 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 86)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 85 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 85 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 221)
    {
      if(useshared)
        postRelaxShared_kernel<IndexType, ValueType, 220 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      else
        postRelax_kernel<IndexType, ValueType, 220 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinEll.num_rows,
            AinEll.num_cols,
            AinEll.column_indices.num_cols,
            AinEll.column_indices.pitch,
            thrust::raw_pointer_cast(&AinEll.column_indices.values[0]),
            thrust::raw_pointer_cast(&AinEll.values.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else
    {
      cout << "In posePCRFull num_per_row equal or larger than 221!!" << endl;
      exit(0);
    }


    x.swap(xout);
  }

  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

  if(level_id != 0)
  {

    permutation_kernel2<IndexType, ValueType> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(x.size(), thrust::raw_pointer_cast(&permutation[0]), thrust::raw_pointer_cast(&x[0]), thrust::raw_pointer_cast(&xout[0]));
    x.swap(xout);
  }

}

template<typename IndexType, typename ValueType, int NUMPERROW>
__global__ void postRelaxCsr_kernel(const IndexType num_rows,
    const IndexType* offsets,
    const IndexType* Aj,
    const ValueType* Ax,
    const IndexType* Aouti,
    const IndexType* Aoutj,
    const ValueType* Aoutv,
    const IndexType num_entries,
    const IndexType* AoutBlockIdx,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* xout,
    int nInnerIter)
{

  ValueType Axlocal[NUMPERROW];
  short Ajlocal[NUMPERROW];

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];

  __shared__ ValueType s_array[1024];
  ValueType* s_b = &s_array[0];
  IndexType row = thread_id + blockstart;

  if(row < blockend)
  {
    s_b[thread_id] = b[row];
  }

  __syncthreads();

  IndexType rowstart = offsets[row];
  IndexType rowend = offsets[row + 1];
  IndexType num_cols_per_row = rowend - rowstart;
  if(row < blockend)
  {
    //load in matrix A to registers
#pragma unroll
    for(int n = 0; n < NUMPERROW; n++)
    {
      if(n < num_cols_per_row)
      {
        Axlocal[n] = Ax[rowstart + n];
        Ajlocal[n] = (short)(Aj[rowstart + n] - blockstart);
      }
    }
  }



  ValueType brow, drow;
  if(row < blockend)
  {
    brow = s_b[thread_id];
    drow = diag[row];
  }

  //load x to shared memory
  ValueType* s_x = &s_array[0];
  if(row < blockend)
    s_x[thread_id] = x[row];

  __syncthreads();

  ValueType sum;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      //compute Ax
      sum = 0.0;
#pragma unroll
      for(unsigned short n = 0; n < NUMPERROW; n++)
      {
        if(n < num_cols_per_row)
        {
          sum += Axlocal[n] * s_x[Ajlocal[n]];
        }
      }
      s_x[thread_id] = s_x[thread_id] + weight * (brow - sum) / drow;
    }
    __syncthreads();
  }


  if(row < blockend)
  {
    //update glocal mem x values

    xout[row] = s_x[thread_id];
  }
}

template<typename IndexType, typename ValueType, int NUMPERROW>
__global__ void postRelaxCsrShared_kernel(const IndexType num_rows,
    const IndexType* offsets,
    const IndexType* Aj,
    const ValueType* Ax,
    const IndexType* Aouti,
    const IndexType* Aoutj,
    const ValueType* Aoutv,
    const IndexType num_entries,
    const IndexType* AoutBlockIdx,
    const ValueType* diag,
    const IndexType* aggregateIdx,
    const IndexType* partitionIdx,
    const IndexType* permutation,
    const ValueType* b,
    const double weight,
    ValueType* x,
    ValueType* xout,
    int nInnerIter)
{
  extern __shared__ char s_mem[];
  ValueType Axlocal[NUMPERROW];
  ushort* s_Ajlocal;

  const IndexType thread_id = threadIdx.x;
  IndexType aggrstart = partitionIdx[blockIdx.x];
  IndexType aggrend = partitionIdx[blockIdx.x + 1];
  const IndexType blockstart = aggregateIdx[aggrstart];
  const IndexType blockend = aggregateIdx[aggrend];


  ValueType* s_b = (ValueType*)s_mem;
  IndexType row = thread_id + blockstart;

  //
  if(row < blockend)
  {
    s_b[thread_id] = b[row];
  }

  __syncthreads();

  ValueType brow, drow;
  if(row < blockend)
  {
    brow = s_b[thread_id];
    drow = diag[row];
  }

  //load x to shared memory
  ValueType* s_x = (ValueType*)s_mem;
  if(row < blockend)
    s_x[thread_id] = x[row];

  __syncthreads();

  s_Ajlocal = (ushort*) & s_x[blockDim.x];
  IndexType rowstart = offsets[row];
  IndexType rowend = offsets[row + 1];
  IndexType num_cols_per_row = rowend - rowstart;
  const IndexType colidxstart = offsets[blockstart];

  if(row < blockend)
  {
    //load in matrix Aj to shared mem
    for(int n = 0; n < num_cols_per_row; n++)
    {
      s_Ajlocal[rowstart + n - colidxstart] = (short)(Aj[rowstart + n] - blockstart);
    }

    //load in matrix Ax to registers
#pragma unroll
    for(int n = 0; n < NUMPERROW; n++)
    {
      if(n < num_cols_per_row)
      {
        Axlocal[n] = Ax[rowstart + n];
      }
    }
  }

  ValueType sum;

  //inner iteration
#pragma unroll
  for(int iter = 0; iter < nInnerIter; iter++)
  {
    if(row < blockend)
    {
      //compute Ax
      sum = 0.0;
#pragma unroll
      for(unsigned short n = 0; n < NUMPERROW; n++)
      {
        if(n < num_cols_per_row)
        {
          sum += Axlocal[n] * s_x[s_Ajlocal[rowstart + n - colidxstart]];
        }
      }
      s_x[thread_id] = s_x[thread_id] + weight * (brow - sum) / drow;
    }
    __syncthreads();
  }


  if(row < blockend)
  {
    //update glocal mem x values

    xout[row] = s_x[thread_id];
  }
}

template<>
void gauss_seidel<Matrix_d, Vector_d>::postPCRFullCsr(const cusp::csr_matrix<IndexType, ValueType, MemorySpace>& AinCsr,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutCoo,
    const cusp::array1d<IndexType, MemorySpace>& AoutBlockIdx,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::hyb_matrix<IndexType, ValueType, MemorySpace>& prolongator,
    const cusp::array1d<IndexType, MemorySpace>& permutation,
    const cusp::array1d<ValueType, MemorySpace>& origb,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& xc,
    int level_id,
    int largestblksize,
    int largestnumentries,
    int largestnumperrow)
{
  Vector_d deltax(x.size());
  cusp::multiply(prolongator, xc, deltax);
  cusp::blas::axpby(x, deltax, x, ValueType(1.0), ValueType(1.0));

  cusp::multiply(AoutCoo, x, deltax); // b' = Aout * x
  cusp::array1d<ValueType, MemorySpace> b(x.size());
  cusp::blas::axpby(origb, deltax, b, ValueType(1.0), ValueType(-1.0)); // b = b - b'

  const size_t THREADS_PER_BLOCK = std::min(max_threads_per_block_,largestblksize);
  const size_t NUM_BLOCKS = partitionIdx.size() - 1;
  if(NUM_BLOCKS > 65535)
    cout << "Block number larger than 65535!!" << endl;

  const size_t SHAREDSIZE = THREADS_PER_BLOCK * sizeof(ValueType) + largestnumentries * sizeof(ushort);
  const size_t NUMPERROW = largestnumperrow;
  const bool useshared = (SHAREDSIZE <= 48 * 1024);
  if(SHAREDSIZE <= 16 * 1024)
  {
    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  }
  else if(SHAREDSIZE <= 48 * 1024)
  {
    cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
  }

  Vector_d xout(x.size());

  for(int i = 0; i < post_relaxes; i++)
  {
    if(NUMPERROW < 10)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 9 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 9 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 15)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 14 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 14 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 20)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 19 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 19 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 25)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 24 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 24 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 30)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 29 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 29 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 35)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 34 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 34 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 40)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 39 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 39 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 45)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 44 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 44 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 50)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 49 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 49 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 55)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 54 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 54 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 60)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 59 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 59 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 65)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 64 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 64 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 70)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 69 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 69 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }

    else if(NUMPERROW < 76)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 75 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 75 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 80)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 79 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 79 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 86)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 85 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 85 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else if(NUMPERROW < 221)
    {
      if(useshared)
      {
        postRelaxCsrShared_kernel<IndexType, ValueType, 220 > << <NUM_BLOCKS, THREADS_PER_BLOCK, SHAREDSIZE >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);

      }
      else
      {
        postRelaxCsr_kernel<IndexType, ValueType, 220 > << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(AinCsr.num_rows,
            thrust::raw_pointer_cast(&AinCsr.row_offsets[0]),
            thrust::raw_pointer_cast(&AinCsr.column_indices[0]),
            thrust::raw_pointer_cast(&AinCsr.values[0]),
            thrust::raw_pointer_cast(&AoutCoo.row_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.column_indices[0]),
            thrust::raw_pointer_cast(&AoutCoo.values[0]),
            AoutCoo.num_entries,
            thrust::raw_pointer_cast(&AoutBlockIdx[0]),
            thrust::raw_pointer_cast(&diag[0]),
            thrust::raw_pointer_cast(&aggregateIdx[0]),
            thrust::raw_pointer_cast(&partitionIdx[0]),
            thrust::raw_pointer_cast(&permutation[0]),
            thrust::raw_pointer_cast(&b[0]),
            weight,
            thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&xout[0]),
            nPostInnerIter);
      }
      AggMIS::CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    else
    {
      cout << "In posePCRFull num_per_row equal or larger than 221!!" << endl;
      exit(0);
    }

    x.swap(xout);
  }

  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

  if(level_id != 0)
  {
    permutation_kernel2<IndexType, ValueType> << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(x.size(), thrust::raw_pointer_cast(&permutation[0]), thrust::raw_pointer_cast(&x[0]), thrust::raw_pointer_cast(&xout[0]));
    x.swap(xout);
  }


}

/****************************************
 * Explict instantiations
 ***************************************/
template class gauss_seidel<Matrix_d, Vector_d>;
