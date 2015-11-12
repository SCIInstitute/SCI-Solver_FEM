/*
 * File:   perform_element_loop_2D.h
 * Author: zhisong
 *
 * Created on November 5, 2011, 1:48 PM
 */

#ifndef __PERFORM_ELEMENT_LOOP_2D_H__
#define __PERFORM_ELEMENT_LOOP_2D_H__

#define DEGREE 6

#include <types.h>
#include <cutil.h>
#include <error.h>
#include <cusp/print.h>

__device__ __constant__ CGType c_z_x[DEGREE];
__device__ __constant__ CGType c_z_y[DEGREE];
__device__ __constant__ CGType c_w_x[DEGREE];
__device__ __constant__ CGType c_w_y[DEGREE];

template<typename ValueType>
__device__ __host__ ValueType forceFunction(ValueType x, ValueType y)
{
  return 0.0;
}

template<typename IndexType, typename ValueType>
__device__ void compute_stiffness_matrix(const ValueType* __restrict__ linearBaseCoeff, ValueType TArea, ValueType* __restrict__ stiffMat)
{
  ValueType a1, a2, b1, b2;
  int cnt = 0;
#pragma unroll
  for (int k = 0; k < 3; k++)
  {
#pragma unroll
    for (int g = k; g < 3; g++)
    {
      a1 = linearBaseCoeff[3 * k + 0];
      b1 = linearBaseCoeff[3 * k + 1];
      a2 = linearBaseCoeff[3 * g + 0];
      b2 = linearBaseCoeff[3 * g + 1];

      stiffMat[cnt++] = (a1 * a2 + b1 * b2) * TArea;
    }
  }
}

template<typename ValueType>
__device__ ValueType Integration_Quadrilateral(ValueType(*fx)[DEGREE])
{
  ValueType integral = 0;
  ValueType tmp_y;

#pragma unroll
  for (int i = 0; i < DEGREE; i++)
  {
    tmp_y = 0.0;
#pragma unroll
    for (int j = 0; j < DEGREE; j++)
    {
      tmp_y += fx[i][j] * c_w_y[j];
    }
    integral += tmp_y * c_w_x[i];
  }

  return integral;
}

template<typename IndexType, typename ValueType>
__device__ void compute_massmatrix_vector(ValueType* __restrict__ vertX, ValueType* __restrict__ vertY,
    ValueType* __restrict__ linearBaseCoeff, ValueType* __restrict__ massMat, ValueType* __restrict__ ele_b)
{
  ValueType x[DEGREE][DEGREE];
  ValueType y[DEGREE][DEGREE];
#pragma unroll
  for (int m = 0; m < DEGREE; m++)
  {
#pragma unroll
    for (int j = 0; j < DEGREE; j++)
    {
      x[m][j] = vertX[0] *(1 - c_z_x[m]) * 0.5 * (1 - c_z_y[j])*0.5 + vertX[1]* (1 + c_z_x[m])*0.5 * (1 - c_z_y[j])*0.5 + vertX[2] * (1 + c_z_y[j])*0.5;
      y[m][j] = vertY[0] *(1 - c_z_x[m]) * 0.5 * (1 - c_z_y[j])*0.5 + vertY[1]* (1 + c_z_x[m])*0.5 * (1 - c_z_y[j])*0.5 + vertY[2] * (1 + c_z_y[j])*0.5;
    }
  }

  ValueType a1, b1, c1, a2, b2, c2;
  ValueType integrandMass[DEGREE][DEGREE];

  int Cnt = 0;
  ValueType jacobi = (vertX[0] * vertY[1] - vertX[1] * vertY[0] - vertX[0] * vertY[2] + vertX[2] * vertY[0] + vertX[1] * vertY[2] - vertX[2] * vertY[1]) / 8;

#pragma unroll
  for (int k = 0; k < 3; k++)
  {
#pragma unroll
    for (int g = k; g < 3; g++)
    {
      a1 = linearBaseCoeff[3 * k + 0];
      b1 = linearBaseCoeff[3 * k + 1];
      c1 = linearBaseCoeff[3 * k + 2];
      a2 = linearBaseCoeff[3 * g + 0];
      b2 = linearBaseCoeff[3 * g + 1];
      c2 = linearBaseCoeff[3 * g + 2];

#pragma unroll
      for (int p = 0; p < DEGREE; p++)
      {
#pragma unroll
        for (int q = 0; q < DEGREE; q++)
        {
          integrandMass[p][q] = (a1 * x[p][q] + b1 * y[p][q] + c1)*(a2 * x[p][q] + b2 * y[p][q] + c2) * jacobi;
        }
      }

      ValueType integralMass = Integration_Quadrilateral<ValueType > (integrandMass);
      massMat[Cnt++] = integralMass;
    }
  }

  ValueType(*integrandForce)[DEGREE];
  integrandForce = integrandMass;
  Cnt = 0;

#pragma unroll
  for (int k = 0; k < 3; k++)
  {
    a1 = linearBaseCoeff[3 * k + 0];
    b1 = linearBaseCoeff[3 * k + 1];
    c1 = linearBaseCoeff[3 * k + 2];

#pragma unroll
    for (int p = 0; p < DEGREE; p++)
    {
#pragma unroll
      for (int q = 0; q < DEGREE; q++)
      {
        ValueType f = forceFunction<ValueType > (x[p][q], y[p][q]);
        integrandForce[p][q] = f * (a1 * x[p][q] + b1 * y[p][q] + c1) * jacobi;
      }
    }

    ValueType integralForce = Integration_Quadrilateral<ValueType > (integrandForce);
    ele_b[Cnt++] = integralForce;
  }
}

template <typename IndexType>
__device__ int binarySearch(IndexType *indices, IndexType low, IndexType high, IndexType _val, const IndexType pitch)
{
  IndexType retval = -1;

  intuint<IndexType> val;
  val.ival = _val;

  //printf("blockIdx: %d, threadIdx: %d, searching for val: %d\n",blockIdx.x,threadIdx.x,_val);
  while (high >= low)
  {
    IndexType mid = low + (high - low) / 2;
    intuint<IndexType> mval;
    mval.ival = indices[pitch * mid];
    if (mval.uval > val.uval)
      high = mid - 1;
    else if (mval.uval < val.uval)
      low = mid + 1;
    else
    {
      retval = mid;
      break;
    }
  }
  //printf("blockIdx: %d, threadIdx: %d, loc: %d\n",blockIdx.x,threadIdx.x,retval);
  return retval;
}

__device__ double atomicAdd(double* address, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  }
  while (assumed != old);
  return __longlong_as_double(old);
}

template<typename IndexType, typename ValueType>
__device__ void sum_into_global_linear_system_cuda(IndexType* __restrict__ ids, ValueType* __restrict__ stiffMat, ValueType* __restrict__ massMat,
    ValueType* __restrict__ ele_b,
    ValueType* __restrict__ d_ellvalues, IndexType* __restrict__ d_ellcolidx, size_t nrow, size_t num_col_per_row, size_t pitch,
    ValueType* __restrict__ d_b)
{
  IndexType idxi = ids[0];
  IndexType idxj = ids[1];
  IndexType* mat_row_cols = &d_ellcolidx[idxi];
  ValueType* mat_row_coefs = &d_ellvalues[idxi];
  ValueType lambda = 1.0;
  ValueType coef = stiffMat[1] + lambda * massMat[1];
  IndexType loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch); // first one is diagonal
  if (loc >= 0)
  {
    atomicAdd(&mat_row_coefs[pitch * loc], coef);
    //    mat_row_coefs[pitch * loc] += coef;
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch); // first one is diagonal
  if (loc >= 0)
  {
    atomicAdd(&mat_row_coefs[pitch * loc], coef);
    //    mat_row_coefs[pitch * loc] += coef;
  }

  idxi = ids[0];
  idxj = ids[2];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[2] + lambda * massMat[2];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch); // first one is diagonal
  if (loc >= 0)
  {
    atomicAdd(&mat_row_coefs[pitch * loc], coef);
    //    mat_row_coefs[pitch * loc] += coef;
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch); // first one is diagonal
  if (loc >= 0)
  {
    atomicAdd(&mat_row_coefs[pitch * loc], coef);
    //    mat_row_coefs[pitch * loc] += coef;
  }

  idxi = ids[1];
  idxj = ids[2];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[4] + lambda * massMat[4];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch); // first one is diagonal
  if (loc >= 0)
  {
    atomicAdd(&mat_row_coefs[pitch * loc], coef);
    //    mat_row_coefs[pitch * loc] += coef;
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch); // first one is diagonal
  if (loc >= 0)
  {
    atomicAdd(&mat_row_coefs[pitch * loc], coef);
    //    mat_row_coefs[pitch * loc] += coef;
  }

  idxi = ids[0];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[0] + lambda * massMat[0];
  atomicAdd(&mat_row_coefs[0], coef);
  //  mat_row_coefs[0] += coef;

  idxi = ids[1];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[3] + lambda * massMat[3];
  atomicAdd(&mat_row_coefs[0], coef);
  //  mat_row_coefs[0] += coef;

  idxi = ids[2];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[5] + lambda * massMat[5];
  atomicAdd(&mat_row_coefs[0], coef);
  //  mat_row_coefs[0] += coef;

  //  sum_into_vector
  atomicAdd(&d_b[ids[0]], ele_b[0]);
  atomicAdd(&d_b[ids[1]], ele_b[1]);
  atomicAdd(&d_b[ids[2]], ele_b[2]);
}

template<typename IndexType, typename ValueType>
__global__ void element_loop_kernel(size_t nv, ValueType *d_nx, 
  ValueType *d_ny, size_t ne, IndexType *d_tri0, IndexType *d_tri1, 
  IndexType *d_tri2,ValueType *d_ellvalues, IndexType *d_ellcolidx,
  size_t nrow, size_t num_col_per_row, size_t pitch, ValueType *d_b)
{
  ValueType coeffs[9];
  ValueType stiffMat[6];
  ValueType massMat[6];
  ValueType ele_b[3];
  IndexType ids[3];
  ValueType x[3];
  ValueType y[3];

  for (int eleidx = blockIdx.x * blockDim.x + threadIdx.x; 
    eleidx < ne; eleidx += blockDim.x * gridDim.x)
  {
    ValueType Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz;
    ValueType AB0, AB1, AB2, AC0, AC1, AC2, r0, r1, r2, a, b, c;

    ids[0] = d_tri0[eleidx];
    ids[1] = d_tri1[eleidx];
    ids[2] = d_tri2[eleidx];

    x[0] = d_nx[ids[0]];
    x[1] = d_nx[ids[1]];
    x[2] = d_nx[ids[2]];

    y[0] = d_ny[ids[0]];
    y[1] = d_ny[ids[1]];
    y[2] = d_ny[ids[2]];

    ValueType TArea = fabs(x[0] * y[2] - x[0] * y[1] +
      x[1] * y[0] - x[1] * y[2] + x[2] * y[1] - x[2] * y[0]) / 2.0;

#pragma unroll
    for (int i = 0; i < 3; i++)
    {
      Ax = x[i % 3];
      Ay = y[i % 3];
      Az = 0.0;
      Bx = x[(i + 1) % 3];
      By = y[(i + 1) % 3];
      Bz = 0.0;
      Cx = x[(i + 2) % 3];
      Cy = y[(i + 2) % 3];
      Cz = 1.0;

      //compute AB cross AC
      AB0 = Bx - Ax;
      AB1 = By - Ay;
      AB2 = Bz - Az;

      AC0 = Cx - Ax;
      AC1 = Cy - Ay;
      AC2 = Cz - Az;

      r0 = AB1 * AC2 - AB2*AC1;
      r1 = AB2 * AC0 - AB0*AC2;
      r2 = AB0 * AC1 - AB1*AC0;

      if (r2 == 0.0) { 
        //TODO
        printf("r2 == 0 : %d, %d\n", eleidx, i);
      }

      a = -r0 / r2;
      b = -r1 / r2;
      c = (r0 * Bx + r1 * By) / r2;

      coeffs[i * 3 + 0] = a;
      coeffs[i * 3 + 1] = b;
      coeffs[i * 3 + 2] = c;

    }

    //compute element stiffness matrix
    compute_stiffness_matrix<IndexType, ValueType > (coeffs, TArea, stiffMat);

    //compte element mass matrix and vector
    compute_massmatrix_vector<IndexType, ValueType > (x, y, coeffs, massMat, ele_b);

    sum_into_global_linear_system_cuda<IndexType, ValueType > (ids, stiffMat, massMat, ele_b,
        d_ellvalues, d_ellcolidx, nrow, num_col_per_row, pitch,
        d_b);



  }

}

template<typename IndexType, typename ValueType>
__global__ void element_loop_coo_kernel(size_t nv, ValueType *d_nx, ValueType *d_ny, size_t ne, IndexType *d_tri0, IndexType *d_tri1, IndexType *d_tri2,
    IndexType *coorowidx, IndexType *coocolidx, ValueType *coovalues,
    ValueType *d_b)
{
}

void perform_element_loop_2d(Vector_d_CG &nx, Vector_d_CG &ny, IdxVector_d &tri0, IdxVector_d &tri1, IdxVector_d &tri2, Matrix_ell_d_CG &A, Vector_d_CG &b,
    Vector_h_CG &z_x, Vector_h_CG &z_y, Vector_h_CG &weight_x, Vector_h_CG &weight_y)
{
  typedef typename Matrix_ell_d_CG::index_type IndexType;
  typedef typename Matrix_ell_d_CG::value_type ValueType;
  int nv = nx.size();
  int ne = tri0.size();

  ValueType *d_b = thrust::raw_pointer_cast(&b[0]);
  ValueType *d_nx = thrust::raw_pointer_cast(&nx[0]);
  ValueType *d_ny = thrust::raw_pointer_cast(&ny[0]);

  IndexType *d_tri0 = thrust::raw_pointer_cast(&tri0[0]);
  IndexType *d_tri1 = thrust::raw_pointer_cast(&tri1[0]);
  IndexType *d_tri2 = thrust::raw_pointer_cast(&tri2[0]);

  ValueType *d_ellvalues = thrust::raw_pointer_cast(&A.values.values[0]);
  IndexType *d_ellcolidx = thrust::raw_pointer_cast(&A.column_indices.values[0]);

  ValueType *zx = thrust::raw_pointer_cast(&z_x[0]);
  ValueType *zy = thrust::raw_pointer_cast(&z_y[0]);
  ValueType *wx = thrust::raw_pointer_cast(&weight_x[0]);
  ValueType *wy = thrust::raw_pointer_cast(&weight_y[0]);

  size_t num_col_per_row = A.column_indices.num_cols;
  size_t pitch = A.column_indices.pitch;
  size_t nrow = A.num_rows;


  cudaSafeCall(cudaMemcpyToSymbol(c_z_x, zx, sizeof (ValueType) * z_x.size(), 0, cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(c_z_y, zy, sizeof (ValueType) * z_y.size(), 0, cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(c_w_x, wx, sizeof (ValueType) * weight_x.size(), 0, cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(c_w_y, wy, sizeof (ValueType) * weight_y.size(), 0, cudaMemcpyHostToDevice));


  int threads = 256;
  int num_blocks = std::min((int)ceil((double)ne / threads), 65535); //32 blocks per SM
  cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
  //cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  //Now do the actual finite-element assembly loop:
  element_loop_kernel<IndexType, ValueType> << <num_blocks, threads >> >(nv, d_nx, d_ny, ne, d_tri0, d_tri1, d_tri2,
      d_ellvalues, d_ellcolidx, nrow, num_col_per_row, pitch,
      d_b);
}

template<typename IndexType, typename ValueType>
__global__ void assemble2csr_kernel(const IndexType* __restrict__ column_indices, const ValueType* __restrict__ values, const IndexType* __restrict__ vert_indices,
    const IndexType* __restrict__ csr_row_offsets, ValueType* __restrict__ csr_values, int nv)
{
  for (int vidx = blockIdx.x * blockDim.x + threadIdx.x; vidx < nv; vidx += gridDim.x * blockDim.x)
  {
    int start = vert_indices[vidx];
    int end = vert_indices[vidx + 1];
    int rowstart = csr_row_offsets[vidx];

    int cnt = 0;
    csr_values[rowstart] += values[start];
    for (int i = start + 1; i < end; i++)
    {

      ValueType v = values[i];
      if (column_indices[i] == column_indices[i - 1])
      {
        csr_values[rowstart + cnt] += v;
      }
      else
      {
        cnt++;
        csr_values[rowstart + cnt] += v;
      }
    }
  }
}

void perform_element_loop_2d_coo(Vector_d_CG &nx, Vector_d_CG &ny, IdxVector_d &tri0, IdxVector_d &tri1, IdxVector_d &tri2, Matrix_d_CG &A, Vector_d_CG &b,
    Vector_h_CG &z_x, Vector_h_CG &z_y, Vector_h_CG &weight_x, Vector_h_CG & weight_y)
{

  typedef typename Matrix_d_CG::index_type IndexType;
  typedef typename Matrix_d_CG::value_type ValueType;
  int nv = nx.size();
  int ne = tri0.size();

  Matrix_coo_d_CG Aout(nv, nv, 6 * ne);

  ValueType *d_b = thrust::raw_pointer_cast(&b[0]);
  ValueType *d_nx = thrust::raw_pointer_cast(&nx[0]);
  ValueType *d_ny = thrust::raw_pointer_cast(&ny[0]);

  IndexType *d_tri0 = thrust::raw_pointer_cast(&tri0[0]);
  IndexType *d_tri1 = thrust::raw_pointer_cast(&tri1[0]);
  IndexType *d_tri2 = thrust::raw_pointer_cast(&tri2[0]);


  IndexType *d_coorowidx = thrust::raw_pointer_cast(&Aout.row_indices[0]);
  IndexType *d_coocolidx = thrust::raw_pointer_cast(&Aout.column_indices[0]);
  ValueType *d_coovalues = thrust::raw_pointer_cast(&Aout.values[0]);

  ValueType *zx = thrust::raw_pointer_cast(&z_x[0]);
  ValueType *zy = thrust::raw_pointer_cast(&z_y[0]);
  ValueType *wx = thrust::raw_pointer_cast(&weight_x[0]);
  ValueType *wy = thrust::raw_pointer_cast(&weight_y[0]);

  cudaSafeCall(cudaMemcpyToSymbol(c_z_x, zx, sizeof (ValueType) * z_x.size(), 0, cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(c_z_y, zy, sizeof (ValueType) * z_y.size(), 0, cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(c_w_x, wx, sizeof (ValueType) * weight_x.size(), 0, cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(c_w_y, wy, sizeof (ValueType) * weight_y.size(), 0, cudaMemcpyHostToDevice));


  int threads = 256;
  int num_blocks = std::min((int)ceil((double)ne / threads), 65535); //32 blocks per SM
  //  cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  //Now do the actual finite-element assembly loop:
  element_loop_coo_kernel<IndexType, ValueType> << <num_blocks, threads >> >(nv, d_nx, d_ny, ne, d_tri0, d_tri1, d_tri2,
      d_coorowidx, d_coocolidx, d_coovalues,
      d_b);

  Aout.sort_by_row_and_column();

  //  cusp::print(Aout);

  cusp::array1d<int, cusp::device_memory> flags(6 * ne, 1);
  cusp::array1d<int, cusp::device_memory> keyoutput(nv+1);
  cusp::array1d<int, cusp::device_memory> valoutput(nv);

  int* flagtmp = thrust::raw_pointer_cast(&flags[0]);
  int* keytmp = thrust::raw_pointer_cast(&keyoutput[0]);
  int* valtmp = thrust::raw_pointer_cast(&valoutput[0]);
  int* rtmp = thrust::raw_pointer_cast(&Aout.row_indices[0]);
  thrust::reduce_by_key(Aout.row_indices.begin(), Aout.row_indices.end(), flags.begin(), keyoutput.begin(), valoutput.begin());
  keyoutput.resize(nv + 1);

  keyoutput[0] = 0;
  thrust::inclusive_scan(valoutput.begin(), valoutput.end(), keyoutput.begin() + 1);

  num_blocks = std::min((int)ceil((double)nv / threads), 65535);
  assemble2csr_kernel<IndexType, ValueType> << <num_blocks, threads >> >(thrust::raw_pointer_cast(&Aout.column_indices[0]), thrust::raw_pointer_cast(&Aout.values[0]), thrust::raw_pointer_cast(&keyoutput[0]),
      thrust::raw_pointer_cast(&A.row_offsets[0]), thrust::raw_pointer_cast(&A.values[0]), nv);

  Aout.resize(0, 0, 0);
  flags.resize(0);
  keyoutput.resize(0);
  valoutput.resize(0);
}
#endif
