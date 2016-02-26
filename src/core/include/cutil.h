#ifndef __CUTIL_H__
#define __CUTIL_H__

#include <TriMesh.h>
#include <tetmesh.h>



/**********************************************************
 * Checks for a cuda error and if one exists prints it,
 * the stack trace, and exits
 *********************************************************/
#define cudaCheckError() {                              \
  cudaError_t e=cudaGetLastError();                                 \
  char error_str[100];                                              \
  if(e!=cudaSuccess) {                                              \
    sprintf(error_str,"Cuda failure: '%s'",cudaGetErrorString(e));  \
    FatalError(error_str);                                          \
  }                                                                 \
}

#define cudaSafeCall(x) {(x); cudaCheckError()}

template <class Matrix, class Vector>
void computeResidual(const Matrix& A, const Vector& x, const Vector& b, Vector& r);

template<typename IndexType, typename ValueType>
__global__ void find_diag_kernel(const IndexType num_rows, const IndexType num_cols, const IndexType num_cols_per_row, const IndexType pitch,
                                 const IndexType * Aj,
                                 const ValueType* Ax,
                                 ValueType* diag)
{
  const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  const IndexType grid_size = gridDim.x * blockDim.x;

  for (IndexType row = thread_id; row < num_rows; row += grid_size)
  {
    IndexType offset = row;

    for (IndexType n = 0; n < num_cols_per_row; n++)
    {
      const IndexType col = Aj[offset];

      if (col == row)
      {
        const ValueType A_ij = Ax[offset];
        diag[row] = A_ij;
      }

      offset += pitch;
    }
  }
}

/**************************************************
 * structs for converting between signed and unsigned values without 
 * type casting.
 * ************************************************/

/*****************************
 * Generic converter for unsigned types.
 * This becomes a no op
 *****************************/
template <class GlobalOrdinal>
struct intuint
{

  union
  {
    GlobalOrdinal ival;
    GlobalOrdinal uval;
  };
};

/***************************
 * char converter
 **************************/
template <>
struct intuint<char>
{

  union
  {
    char ival;
    unsigned char uval;
  };
};

/***************************
 * Short converter
 **************************/
template <>
struct intuint<short>
{

  union
  {
    short ival;
    unsigned short uval;
  };
};

/***************************
 * Integer converter
 **************************/
template <>
struct intuint<int>
{

  union
  {
    int ival;
    unsigned int uval;
  };
};

/***************************
 * long converter
 **************************/
template <>
struct intuint<long>
{

  union
  {
    long ival;
    unsigned long uval;
  };
};

struct metisinput
{
  int nn;
  int* xadj;
  int* adjncy;
};

struct cudaCSRGraph
{
  int nn;
  int* xadj;
  int* adjncy;
};

template<class Matrix>
void trimesh2ell(TriMesh* meshPtr, Matrix &A);

template<class Matrix>
void trimesh2csr(TriMesh* meshPtr, Matrix &A);

template<class Matrix>
void tetmesh2ell(TetMesh* meshPtr, Matrix &A, bool verbose);

void convertSym2gen(Matrix_d_CG &Acsr, Matrix_coo_d_CG &Aout);
#endif
