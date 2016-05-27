/*
 * File:   perform_element_loop_2D.h
 * Author: zhisong
 *
 * Created on November 5, 2011, 1:48 PM
 */

#ifndef __PERFORM_ELEMENT_LOOP_3D_H__
#define __PERFORM_ELEMENT_LOOP_3D_H__

#include <types.h>
#include <cutil.h>
#include <error.h>
#include <cusp/print.h>
#include <my_timer.h>

__device__ __constant__ CGType c_w_x_3d[DEGREE];
__device__ __constant__ CGType c_w_y_3d[DEGREE];
__device__ __constant__ CGType c_w_z_3d[DEGREE];
__device__ __constant__ CGType c_phi[DEGREE*DEGREE*DEGREE * 4];

template<typename ValueType>
__device__ __host__ ValueType forceFunction_3d(ValueType x, ValueType y)
{
  return 0.0;
}

template<typename IndexType, typename ValueType>
__device__ __host__ void compute_stiffness_matrix_3d(const ValueType* __restrict__ linearBaseCoeff, ValueType Tvol, ValueType* __restrict__ stiffMat, ValueType co)
{
  ValueType a1, a2, b1, b2, c1, c2;
  int cnt = 0;
#pragma unroll
  for (int k = 0; k < 4; k++)
  {
#pragma unroll
    for (int g = k; g < 4; g++)
    {
      a1 = linearBaseCoeff[4 * k + 0];
      b1 = linearBaseCoeff[4 * k + 1];
      c1 = linearBaseCoeff[4 * k + 2];
      a2 = linearBaseCoeff[4 * g + 0];
      b2 = linearBaseCoeff[4 * g + 1];
      c2 = linearBaseCoeff[4 * g + 2];
      stiffMat[cnt++] = (a1 * a2 + b1 * b2 + c1 * c2) * Tvol * co;
    }
  }
}

template<typename ValueType >
__device__ __host__ ValueType Integration_Quadrilateral_3d(ValueType(*fx)[DEGREE][DEGREE], ValueType* w_x, ValueType* w_y, ValueType* w_z)
{
  ValueType integral = 0;
  ValueType tmp_y, tmp_z;

  for (int i = 0; i < DEGREE; i++)
  {
    tmp_y = 0.0;
    for (int j = 0; j < DEGREE; j++)
    {
      tmp_z = 0.0;
      for (int k = 0; k < DEGREE; k++)
      {
        tmp_z += fx[i][j][k] * w_z[k];
      }
      tmp_y += tmp_z * w_y[j];
    }
    integral += tmp_y * w_x[i];
  }

  return integral;
}

template<typename IndexType, typename ValueType >
__device__ __host__ void compute_massmatrix_vector_3d(ValueType* vertX, ValueType* vertY, ValueType* vertZ,
    ValueType* __restrict__ linearBaseCoeff, ValueType* __restrict__ massMat, ValueType * __restrict__ ele_b,
    ValueType* w_x_3d, ValueType* w_y_3d, ValueType* w_z_3d, ValueType* phi, ValueType* integrand)
{
  ValueType x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4;

  x1 = vertX[0];
  y1 = vertY[0];
  z1 = vertZ[0];
  x2 = vertX[1];
  y2 = vertY[1];
  z2 = vertZ[1];
  x3 = vertX[2];
  y3 = vertY[2];
  z3 = vertZ[2];
  x4 = vertX[3];
  y4 = vertY[3];
  z4 = vertZ[3];


  ValueType det = 0.125 * ((-x1 + x2) * (-y1 + y3) * (-z1 + z4) +
      (-y1 + y2) * (-z1 + z3) * (-x1 + x4) +
      (-z1 + z2) * (-x1 + x3) * (-y1 + y4) -
      (-x1 + x2) * (-z1 + z3) * (-y1 + y4) -
      (-z1 + z2) * (-y1 + y3) * (-x1 + x4) -
      (-y1 + y2) * (-x1 + x3) * (-z1 + z4));

  ValueType jacobi = fabs(det);

  int Cnt = 0;

#pragma unroll
  for (int k = 0; k < 4; k++)
  {
#pragma unroll
    for (int g = k; g < 4; g++)
    {
      massMat[Cnt] = integrand[Cnt] * jacobi;
      Cnt++;
    }
  }
}

template <typename IndexType >
__device__ __host__ int binarySearch(IndexType *indices, IndexType low, IndexType high, IndexType _val, const IndexType pitch)
{
  IndexType retval = -1;

  intuint<IndexType> val;
  val.ival = _val;

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
  return retval;
}

__device__ double atomicAdd_3d(double* address, double val)
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

__device__ float atomicAdd_3d(float* address, float val)
{
  return atomicAdd(address, val);
}

template<typename IndexType, typename ValueType >
__device__ void sum_into_global_linear_system_cuda_3d(IndexType* __restrict__ ids, ValueType* __restrict__ stiffMat, ValueType* __restrict__ massMat,
    ValueType* __restrict__ ele_b,
    ValueType* __restrict__ d_ellvalues, IndexType* __restrict__ d_ellcolidx, size_t nrow, size_t num_col_per_row, size_t pitch,
    ValueType * __restrict__ d_b)
{
  IndexType idxi = ids[0];
  IndexType idxj = ids[1];
  IndexType* mat_row_cols = &d_ellcolidx[idxi];
  __syncthreads();
  ValueType* mat_row_coefs = &d_ellvalues[idxi];
  ValueType lambda = 1.0;
  ValueType coef = stiffMat[1] + lambda * massMat[1];

  IndexType loc;
  // first one is diagonal
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch);
  if (loc >= 0)
  {
    atomicAdd_3d(&mat_row_coefs[pitch * loc], coef);
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch); // first one is diagonal
  if (loc >= 0)
  {
    atomicAdd_3d(&mat_row_coefs[pitch * loc], coef);
  }

  idxi = ids[0];
  idxj = ids[2];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[2] + lambda * massMat[2];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch); // first one is diagonal
  if (loc >= 0)
  {
    atomicAdd_3d(&mat_row_coefs[pitch * loc], coef);

  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch); // first one is diagonal
  if (loc >= 0)
  {
    atomicAdd_3d(&mat_row_coefs[pitch * loc], coef);
  }

  idxi = ids[0];
  idxj = ids[3];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[3] + lambda * massMat[3];
  // first one is diagonal
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch);
  if (loc >= 0)
  {
    atomicAdd_3d(&mat_row_coefs[pitch * loc], coef);
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch); // first one is diagonal
  if (loc >= 0)
  {
    atomicAdd_3d(&mat_row_coefs[pitch * loc], coef);
  }

  idxi = ids[1];
  idxj = ids[2];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[5] + lambda * massMat[5];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch); // first one is diagonal
  if (loc >= 0)
  {
    atomicAdd_3d(&mat_row_coefs[pitch * loc], coef);
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  //first one is diagonal
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch);
  if (loc >= 0)
  {
    atomicAdd_3d(&mat_row_coefs[pitch * loc], coef);
  }

  idxi = ids[1];
  idxj = ids[3];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[6] + lambda * massMat[6];
  //first one is diagonal
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch);
  if (loc >= 0)
  {
    atomicAdd_3d(&mat_row_coefs[pitch * loc], coef);
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  //first one is diagonal
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch);
  if (loc >= 0)
  {
    atomicAdd_3d(&mat_row_coefs[pitch * loc], coef);
  }

  idxi = ids[2];
  idxj = ids[3];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[8] + lambda * massMat[8];
  // first one is diagonal
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch);
  if (loc >= 0)
  {
    atomicAdd_3d(&mat_row_coefs[pitch * loc], coef);
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  // first one is diagonal
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch);
  if (loc >= 0)
  {
    atomicAdd_3d(&mat_row_coefs[pitch * loc], coef);
  }

  idxi = ids[0];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[0] + lambda * massMat[0];
  atomicAdd_3d(&mat_row_coefs[0], coef);

  idxi = ids[1];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[4] + lambda * massMat[4];
  atomicAdd_3d(&mat_row_coefs[0], coef);

  idxi = ids[2];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[7] + lambda * massMat[7];
  atomicAdd_3d(&mat_row_coefs[0], coef);

  idxi = ids[3];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[9] + lambda * massMat[9];
  atomicAdd_3d(&mat_row_coefs[0], coef);
}

template<typename IndexType, typename ValueType >
void sum_into_global_linear_system_3d_host(IndexType* __restrict__ ids, ValueType* __restrict__ stiffMat, ValueType* __restrict__ massMat,
    ValueType* __restrict__ ele_b,
    ValueType* __restrict__ d_ellvalues, IndexType* __restrict__ d_ellcolidx, size_t nrow, size_t num_col_per_row, size_t pitch,
    ValueType * __restrict__ d_b)
{
  IndexType idxi = ids[0];
  IndexType idxj = ids[1];
  IndexType* mat_row_cols = &d_ellcolidx[idxi];
  ValueType* mat_row_coefs = &d_ellvalues[idxi];
  ValueType lambda = 1.0;
  ValueType coef = stiffMat[1] + lambda * massMat[1];
  // first one is diagonal
  IndexType loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch);
  if (loc >= 0)
  {
    mat_row_coefs[pitch * loc] += coef;
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch);
  if (loc >= 0)
  {
    mat_row_coefs[pitch * loc] += coef;
  }

  idxi = ids[0];
  idxj = ids[2];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[2] + lambda * massMat[2];
  // first one is diagonal
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch);
  if (loc >= 0)
  {
    mat_row_coefs[pitch * loc] += coef;
  }
  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch);
  if (loc >= 0)
  {
    mat_row_coefs[pitch * loc] += coef;
  }

  idxi = ids[0];
  idxj = ids[3];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[3] + lambda * massMat[3];
  // first one is diagonal
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch);
  if (loc >= 0)
  {
    mat_row_coefs[pitch * loc] += coef;
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch);
  if (loc >= 0)
  {
    mat_row_coefs[pitch * loc] += coef;
  }

  idxi = ids[1];
  idxj = ids[2];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[5] + lambda * massMat[5];
  // first one is diagonal
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch);
  if (loc >= 0)
  {
    mat_row_coefs[pitch * loc] += coef;
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch);
  if (loc >= 0)
  {
    mat_row_coefs[pitch * loc] += coef;
  }

  idxi = ids[1];
  idxj = ids[3];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[6] + lambda * massMat[6];
  // first one is diagonal
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch);
  if (loc >= 0)
  {
    mat_row_coefs[pitch * loc] += coef;
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch);
  if (loc >= 0)
  {
    mat_row_coefs[pitch * loc] += coef;
  }

  idxi = ids[2];
  idxj = ids[3];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[8] + lambda * massMat[8];
  // first one is diagonal
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxj, pitch);
  if (loc >= 0)
  {
    mat_row_coefs[pitch * loc] += coef;
  }

  mat_row_cols = &d_ellcolidx[idxj];
  mat_row_coefs = &d_ellvalues[idxj];
  loc = binarySearch<IndexType > (mat_row_cols, 1, num_col_per_row - 1, idxi, pitch);
  if (loc >= 0)
  {
    mat_row_coefs[pitch * loc] += coef;
  }

  idxi = ids[0];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[0] + lambda * massMat[0];
  // first one is diagonal
  mat_row_coefs[0] += coef;

  idxi = ids[1];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[4] + lambda * massMat[4];
  // first one is diagonal
  mat_row_coefs[0] += coef;

  idxi = ids[2];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[7] + lambda * massMat[7];
  // first one is diagonal
  mat_row_coefs[0] += coef;

  idxi = ids[3];
  mat_row_cols = &d_ellcolidx[idxi];
  mat_row_coefs = &d_ellvalues[idxi];
  coef = stiffMat[9] + lambda * massMat[9];
  // first one is diagonal
  mat_row_coefs[0] += coef;

  d_b[ids[0]] += ele_b[0];
  d_b[ids[1]] += ele_b[1];
  d_b[ids[2]] += ele_b[2];
  d_b[ids[3]] += ele_b[3];
}

template<typename IndexType, typename ValueType >
__global__ void element_loop_3d_kernel(size_t nv, ValueType *d_nx, ValueType *d_ny, ValueType *d_nz, size_t ne, IndexType *d_tri0, IndexType *d_tri1, IndexType *d_tri2, IndexType *d_tri3,
    ValueType *d_ellvalues, IndexType *d_ellcolidx, size_t nrow, size_t num_col_per_row, size_t pitch,
    ValueType * d_b, IndexType* matlabels, ValueType* matvalues, ValueType* integrand)
{
  ValueType coeffs[16];
  ValueType stiffMat[10];
  ValueType massMat[10] = {0};
  ValueType ele_b[4];
  IndexType ids[4];
  ValueType x[4];
  ValueType y[4];
  ValueType z[4];

  IndexType matlabel;
  ValueType co;

  for (int eleidx = blockIdx.x * blockDim.x + threadIdx.x; eleidx < ne; eleidx += blockDim.x * gridDim.x)
  {
    ids[0] = d_tri0[eleidx];
    ids[1] = d_tri1[eleidx];
    ids[2] = d_tri2[eleidx];
    ids[3] = d_tri3[eleidx];

    x[0] = d_nx[ids[0]];
    x[1] = d_nx[ids[1]];
    x[2] = d_nx[ids[2]];
    x[3] = d_nx[ids[3]];

    y[0] = d_ny[ids[0]];
    y[1] = d_ny[ids[1]];
    y[2] = d_ny[ids[2]];
    y[3] = d_ny[ids[3]];

    z[0] = d_nz[ids[0]];
    z[1] = d_nz[ids[1]];
    z[2] = d_nz[ids[2]];
    z[3] = d_nz[ids[3]];
    ValueType a1 = x[1] - x[3], a2 = y[1] - y[3], a3 = z[1] - z[3];
    ValueType b1 = x[2] - x[3], b2 = y[2] - y[3], b3 = z[2] - z[3];
    ValueType c1 = x[0] - x[3], c2 = y[0] - y[3], c3 = z[0] - z[3];

    ValueType Tvol = fabs(
      c1 * (a2 * b3 - a3 * b2) +
      c2 * (a3 * b1 - a1 * b3) +
      c3 * (a1 * b2 - a2 * b1)) / 6.0;

    //compute inverse of 4 by 4 matrix
    ValueType a11 = x[0], a12 = y[0], a13 = z[0], a14 = 1.0,
      a21 = x[1], a22 = y[1], a23 = z[1], a24 = 1.0,
      a31 = x[2], a32 = y[2], a33 = z[2], a34 = 1.0,
      a41 = x[3], a42 = y[3], a43 = z[3], a44 = 1.0;

    ValueType det =
      a11 * a22 * a33 * a44 + a11 * a23 * a34 * a42 + a11 * a24 * a32 * a43
      + a12 * a21 * a34 * a43 + a12 * a23 * a31 * a44 + a12 * a24 * a33 * a41
      + a13 * a21 * a32 * a44 + a13 * a22 * a34 * a41 + a13 * a24 * a31 * a42
      + a14 * a21 * a33 * a42 + a14 * a22 * a31 * a43 + a14 * a23 * a32 * a41
      - a11 * a22 * a34 * a43 - a11 * a23 * a32 * a44 - a11 * a24 * a33 * a42
      - a12 * a21 * a33 * a44 - a12 * a23 * a34 * a41 - a12 * a24 * a31 * a43
      - a13 * a21 * a34 * a42 - a13 * a22 * a31 * a44 - a13 * a24 * a32 * a41
      - a14 * a21 * a32 * a43 - a14 * a22 * a33 * a41 - a14 * a23 * a31 * a42;

    ValueType b11 = a22 * a33 * a44 + a23 * a34 * a42 + a24 * a32 * a43 - a22 * a34 * a43 - a23 * a32 * a44 - a24 * a33 * a42;
    ValueType b12 = a12 * a34 * a43 + a13 * a32 * a44 + a14 * a33 * a42 - a12 * a33 * a44 - a13 * a34 * a42 - a14 * a32 * a43;
    ValueType b13 = a12 * a23 * a44 + a13 * a24 * a42 + a14 * a22 * a43 - a12 * a24 * a43 - a13 * a22 * a44 - a14 * a23 * a42;
    ValueType b14 = a12 * a24 * a33 + a13 * a22 * a34 + a14 * a23 * a32 - a12 * a23 * a34 - a13 * a24 * a32 - a14 * a22 * a33;

    ValueType b21 = a21 * a34 * a43 + a23 * a31 * a44 + a24 * a33 * a41 - a21 * a33 * a44 - a23 * a34 * a41 - a24 * a31 * a43;
    ValueType b22 = a11 * a33 * a44 + a13 * a34 * a41 + a14 * a31 * a43 - a11 * a34 * a43 - a13 * a31 * a44 - a14 * a33 * a41;
    ValueType b23 = a11 * a24 * a43 + a13 * a21 * a44 + a14 * a23 * a41 - a11 * a23 * a44 - a13 * a24 * a41 - a14 * a21 * a43;
    ValueType b24 = a11 * a23 * a34 + a13 * a24 * a31 + a14 * a21 * a33 - a11 * a24 * a33 - a13 * a21 * a34 - a14 * a23 * a31;


    ValueType b31 = a21 * a32 * a44 + a22 * a34 * a41 + a24 * a31 * a42 - a21 * a34 * a42 - a22 * a31 * a44 - a24 * a32 * a41;
    ValueType b32 = a11 * a34 * a42 + a12 * a31 * a44 + a14 * a32 * a41 - a11 * a32 * a44 - a12 * a34 * a41 - a14 * a31 * a42;
    ValueType b33 = a11 * a22 * a44 + a12 * a24 * a41 + a14 * a21 * a42 - a11 * a24 * a42 - a12 * a21 * a44 - a14 * a22 * a41;
    ValueType b34 = a11 * a24 * a32 + a12 * a21 * a34 + a14 * a22 * a31 - a11 * a22 * a34 - a12 * a24 * a31 - a14 * a21 * a32;

    ValueType b41 = a21 * a33 * a42 + a22 * a31 * a43 + a23 * a32 * a41 - a21 * a32 * a43 - a22 * a33 * a41 - a23 * a31 * a42;
    ValueType b42 = a11 * a32 * a43 + a12 * a33 * a41 + a13 * a31 * a42 - a11 * a33 * a42 - a12 * a31 * a43 - a13 * a32 * a41;
    ValueType b43 = a11 * a23 * a42 + a12 * a21 * a43 + a13 * a22 * a41 - a11 * a22 * a43 - a12 * a23 * a41 - a13 * a21 * a42;
    ValueType b44 = a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a11 * a23 * a32 - a12 * a21 * a33 - a13 * a22 * a31;

    coeffs[0] = b11 / det;
    coeffs[1] = b21 / det;
    coeffs[2] = b31 / det;
    coeffs[3] = b41 / det;

    coeffs[4] = b12 / det;
    coeffs[5] = b22 / det;
    coeffs[6] = b32 / det;
    coeffs[7] = b42 / det;

    coeffs[8] = b13 / det;
    coeffs[9] = b23 / det;
    coeffs[10] = b33 / det;
    coeffs[11] = b43 / det;

    coeffs[12] = b14 / det;
    coeffs[13] = b24 / det;
    coeffs[14] = b34 / det;
    coeffs[15] = b44 / det;

    //compute element stiffness matrix
    matlabel = matlabels[eleidx];
    co = matvalues[matlabel];

    compute_stiffness_matrix_3d<IndexType, ValueType > (coeffs, Tvol, stiffMat, co);
    //    if(threadIdx.x < 0)
    {

      //compte element mass matrix and vector
      compute_massmatrix_vector_3d<IndexType, ValueType > (x, y, z, coeffs, massMat, ele_b, c_w_x_3d, c_w_y_3d, c_w_z_3d, c_phi, integrand);


      sum_into_global_linear_system_cuda_3d<IndexType, ValueType > (ids, stiffMat, massMat, ele_b,
          d_ellvalues, d_ellcolidx, nrow, num_col_per_row, pitch,
          d_b);
    }


  }
}

template<typename IndexType, typename ValueType >
void element_loop_3d_host(Vector_h_CG &nx, Vector_h_CG &ny, Vector_h_CG &nz, IdxVector_h &tri0, IdxVector_h &tri1, IdxVector_h &tri2, IdxVector_h &tri3, Matrix_ell_h_CG &A, Vector_h_CG &b,
    Vector_h_CG & phi, Vector_h_CG &weight_x, Vector_h_CG &weight_y, Vector_h_CG & weight_z, IdxVector_h &matlabels, Vector_h_CG &matvalues, Vector_h_CG &integrand)
{
  ValueType coeffs[16];
  ValueType stiffMat[10];
  ValueType massMat[10] = {0};
  ValueType ele_b[4];
  IndexType ids[4];
  ValueType x[4];
  ValueType y[4];
  ValueType z[4];

  IndexType matlabel;
  ValueType co;
  int ne = tri0.size();

  ValueType *integrand_ptr = thrust::raw_pointer_cast(&integrand[0]);
  ValueType *wx_ptr = thrust::raw_pointer_cast(&weight_x[0]);
  ValueType *wy_ptr = thrust::raw_pointer_cast(&weight_y[0]);
  ValueType *wz_ptr = thrust::raw_pointer_cast(&weight_z[0]);
  ValueType *phi_ptr = thrust::raw_pointer_cast(&phi[0]);
  ValueType *d_ellvalues = thrust::raw_pointer_cast(&A.values.values[0]);
  IndexType *d_ellcolidx = thrust::raw_pointer_cast(&A.column_indices.values[0]);
  size_t num_col_per_row = A.column_indices.num_cols;
  size_t pitch = A.column_indices.pitch;
  size_t nrow = A.num_rows;
  ValueType *d_b = thrust::raw_pointer_cast(&b[0]);

  for (int eleidx = 0; eleidx < ne; eleidx++)
  {
    ids[0] = tri0[eleidx];
    ids[1] = tri1[eleidx];
    ids[2] = tri2[eleidx];
    ids[3] = tri3[eleidx];

    x[0] = nx[ids[0]];
    x[1] = nx[ids[1]];
    x[2] = nx[ids[2]];
    x[3] = nx[ids[3]];

    y[0] = ny[ids[0]];
    y[1] = ny[ids[1]];
    y[2] = ny[ids[2]];
    y[3] = ny[ids[3]];

    z[0] = nz[ids[0]];
    z[1] = nz[ids[1]];
    z[2] = nz[ids[2]];
    z[3] = nz[ids[3]];

    ValueType a1 = x[1] - x[3], a2 = y[1] - y[3], a3 = z[1] - z[3];
    ValueType b1 = x[2] - x[3], b2 = y[2] - y[3], b3 = z[2] - z[3];
    ValueType c1 = x[0] - x[3], c2 = y[0] - y[3], c3 = z[0] - z[3];

    ValueType Tvol = fabs(c1 * (a2 * b3 - a3 * b2) + c2 * (a3 * b1 - a1 * b3) + c3 * (a1 * b2 - a2 * b1)) / 6.0;

    //compute inverse of 4 by 4 matrix
    ValueType a11 = x[0], a12 = y[0], a13 = z[0], a14 = 1.0, a21 = x[1], a22 = y[1], a23 = z[1], a24 = 1.0, a31 = x[2], a32 = y[2], a33 = z[2], a34 = 1.0, a41 = x[3], a42 = y[3], a43 = z[3], a44 = 1.0;

    ValueType det =
      a11 * a22 * a33 * a44 + a11 * a23 * a34 * a42 + a11 * a24 * a32 * a43
      + a12 * a21 * a34 * a43 + a12 * a23 * a31 * a44 + a12 * a24 * a33 * a41
      + a13 * a21 * a32 * a44 + a13 * a22 * a34 * a41 + a13 * a24 * a31 * a42
      + a14 * a21 * a33 * a42 + a14 * a22 * a31 * a43 + a14 * a23 * a32 * a41
      - a11 * a22 * a34 * a43 - a11 * a23 * a32 * a44 - a11 * a24 * a33 * a42
      - a12 * a21 * a33 * a44 - a12 * a23 * a34 * a41 - a12 * a24 * a31 * a43
      - a13 * a21 * a34 * a42 - a13 * a22 * a31 * a44 - a13 * a24 * a32 * a41
      - a14 * a21 * a32 * a43 - a14 * a22 * a33 * a41 - a14 * a23 * a31 * a42;

    ValueType b11 = a22 * a33 * a44 + a23 * a34 * a42 + a24 * a32 * a43 - a22 * a34 * a43 - a23 * a32 * a44 - a24 * a33 * a42;
    ValueType b12 = a12 * a34 * a43 + a13 * a32 * a44 + a14 * a33 * a42 - a12 * a33 * a44 - a13 * a34 * a42 - a14 * a32 * a43;
    ValueType b13 = a12 * a23 * a44 + a13 * a24 * a42 + a14 * a22 * a43 - a12 * a24 * a43 - a13 * a22 * a44 - a14 * a23 * a42;
    ValueType b14 = a12 * a24 * a33 + a13 * a22 * a34 + a14 * a23 * a32 - a12 * a23 * a34 - a13 * a24 * a32 - a14 * a22 * a33;

    ValueType b21 = a21 * a34 * a43 + a23 * a31 * a44 + a24 * a33 * a41 - a21 * a33 * a44 - a23 * a34 * a41 - a24 * a31 * a43;
    ValueType b22 = a11 * a33 * a44 + a13 * a34 * a41 + a14 * a31 * a43 - a11 * a34 * a43 - a13 * a31 * a44 - a14 * a33 * a41;
    ValueType b23 = a11 * a24 * a43 + a13 * a21 * a44 + a14 * a23 * a41 - a11 * a23 * a44 - a13 * a24 * a41 - a14 * a21 * a43;
    ValueType b24 = a11 * a23 * a34 + a13 * a24 * a31 + a14 * a21 * a33 - a11 * a24 * a33 - a13 * a21 * a34 - a14 * a23 * a31;


    ValueType b31 = a21 * a32 * a44 + a22 * a34 * a41 + a24 * a31 * a42 - a21 * a34 * a42 - a22 * a31 * a44 - a24 * a32 * a41;
    ValueType b32 = a11 * a34 * a42 + a12 * a31 * a44 + a14 * a32 * a41 - a11 * a32 * a44 - a12 * a34 * a41 - a14 * a31 * a42;
    ValueType b33 = a11 * a22 * a44 + a12 * a24 * a41 + a14 * a21 * a42 - a11 * a24 * a42 - a12 * a21 * a44 - a14 * a22 * a41;
    ValueType b34 = a11 * a24 * a32 + a12 * a21 * a34 + a14 * a22 * a31 - a11 * a22 * a34 - a12 * a24 * a31 - a14 * a21 * a32;

    ValueType b41 = a21 * a33 * a42 + a22 * a31 * a43 + a23 * a32 * a41 - a21 * a32 * a43 - a22 * a33 * a41 - a23 * a31 * a42;
    ValueType b42 = a11 * a32 * a43 + a12 * a33 * a41 + a13 * a31 * a42 - a11 * a33 * a42 - a12 * a31 * a43 - a13 * a32 * a41;
    ValueType b43 = a11 * a23 * a42 + a12 * a21 * a43 + a13 * a22 * a41 - a11 * a22 * a43 - a12 * a23 * a41 - a13 * a21 * a42;
    ValueType b44 = a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a11 * a23 * a32 - a12 * a21 * a33 - a13 * a22 * a31;

    coeffs[0] = b11 / det;
    coeffs[1] = b21 / det;
    coeffs[2] = b31 / det;
    coeffs[3] = b41 / det;

    coeffs[4] = b12 / det;
    coeffs[5] = b22 / det;
    coeffs[6] = b32 / det;
    coeffs[7] = b42 / det;

    coeffs[8] = b13 / det;
    coeffs[9] = b23 / det;
    coeffs[10] = b33 / det;
    coeffs[11] = b43 / det;

    coeffs[12] = b14 / det;
    coeffs[13] = b24 / det;
    coeffs[14] = b34 / det;
    coeffs[15] = b44 / det;

    //compute element stiffness matrix
    matlabel = matlabels[eleidx];
    co = matvalues[matlabel];

    compute_stiffness_matrix_3d<IndexType, ValueType > (coeffs, Tvol, stiffMat, co);

    //compte element mass matrix and vector
    compute_massmatrix_vector_3d<IndexType, ValueType > (x, y, z, coeffs, massMat, ele_b, wx_ptr, wy_ptr, wz_ptr, phi_ptr, integrand_ptr);

    sum_into_global_linear_system_3d_host<IndexType, ValueType > (ids, stiffMat, massMat, ele_b,
        d_ellvalues, d_ellcolidx, nrow, num_col_per_row, pitch,
        d_b);
  }
}

void perform_element_loop_3d(Vector_d_CG &nx, Vector_d_CG &ny, Vector_d_CG &nz, IdxVector_d &tri0, IdxVector_d &tri1, IdxVector_d &tri2, IdxVector_d &tri3, Matrix_ell_d_CG &A, Vector_d_CG &b,
    Vector_h_CG & phi, Vector_h_CG &weight_x, Vector_h_CG &weight_y, Vector_h_CG & weight_z, IdxVector_d &matlabels, Vector_d_CG &matvalues, Vector_d_CG &integrand, bool isdevice)
{
  typedef typename Matrix_ell_d_CG::index_type IndexType;
  typedef typename Matrix_ell_d_CG::value_type ValueType;
  int nv = nx.size();
  int ne = tri0.size();
  double start, stop;
  if (isdevice)
  {
    ValueType *d_b = thrust::raw_pointer_cast(&b[0]);
    ValueType *d_nx = thrust::raw_pointer_cast(&nx[0]);
    ValueType *d_ny = thrust::raw_pointer_cast(&ny[0]);
    ValueType *d_nz = thrust::raw_pointer_cast(&nz[0]);

    IndexType *d_tri0 = thrust::raw_pointer_cast(&tri0[0]);
    IndexType *d_tri1 = thrust::raw_pointer_cast(&tri1[0]);
    IndexType *d_tri2 = thrust::raw_pointer_cast(&tri2[0]);
    IndexType *d_tri3 = thrust::raw_pointer_cast(&tri3[0]);

    IndexType *d_matlabels = thrust::raw_pointer_cast(&matlabels[0]);
    ValueType *d_matvalues = thrust::raw_pointer_cast(&matvalues[0]);

    ValueType *d_ellvalues = thrust::raw_pointer_cast(&A.values.values[0]);
    IndexType *d_ellcolidx = thrust::raw_pointer_cast(&A.column_indices.values[0]);

    ValueType *wx = thrust::raw_pointer_cast(&weight_x[0]);
    ValueType *wy = thrust::raw_pointer_cast(&weight_y[0]);
    ValueType *wz = thrust::raw_pointer_cast(&weight_z[0]);
    ValueType *integrand_d = thrust::raw_pointer_cast(&integrand[0]);

    ValueType *h_phi = thrust::raw_pointer_cast(&phi[0]);

    size_t num_col_per_row = A.column_indices.num_cols;
    size_t pitch = A.column_indices.pitch;
    size_t nrow = A.num_rows;

    cudaMemcpyToSymbol(c_w_x_3d, wx, sizeof (ValueType) *
        weight_x.size(), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_w_y_3d, wy, sizeof (ValueType) *
        weight_y.size(), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_w_z_3d, wz, sizeof (ValueType) *
        weight_z.size(), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_phi, h_phi, sizeof (ValueType) *
        phi.size(), 0, cudaMemcpyHostToDevice);

    int threads = 256;
    int num_blocks = std::min((int)ceil((double)ne / threads), 65535); //32 blocks per SM
    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
    //Now do the actual finite-element assembly loop:
    element_loop_3d_kernel<IndexType, ValueType>
      << <num_blocks, threads >> >(
          nv, d_nx, d_ny, d_nz, ne, d_tri0, d_tri1, d_tri2, d_tri3,
          d_ellvalues, d_ellcolidx, nrow, num_col_per_row, pitch,
          d_b, d_matlabels, d_matvalues, integrand_d);

  }
  else
  {
    Vector_h_CG h_b = b;
    Vector_h_CG h_nx = nx;
    Vector_h_CG h_ny = ny;
    Vector_h_CG h_nz = nz;

    IdxVector_h h_tri0 = tri0;
    IdxVector_h h_tri1 = tri1;
    IdxVector_h h_tri2 = tri2;
    IdxVector_h h_tri3 = tri3;
    Vector_h_CG integrand_h = integrand;

    IdxVector_h h_matlabels = matlabels;
    Vector_h_CG h_matvalues = matvalues;

    start = CLOCK();

    Matrix_ell_h_CG h_Aell = A;

    cudaThreadSynchronize();
    stop = CLOCK();

    double copy1 = stop - start;


    element_loop_3d_host<IndexType, ValueType >
      (h_nx, h_ny, h_nz, h_tri0, h_tri1, h_tri2, h_tri3,
       h_Aell, h_b, phi, weight_x, weight_y, weight_z, h_matlabels, h_matvalues, integrand_h);

    start = CLOCK();

    A = h_Aell;

    cudaThreadSynchronize();
    stop = CLOCK();

    double copy2 = stop - start;
    printf("data transfer time in host assemble is: %f\n", copy1 + copy2);
  }
}
#endif
