#ifndef __TYPES_H__
#define __TYPES_H__
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/array1d.h>

typedef double CGType;
//typedef float  CGType;
typedef float AMGType;
typedef double AssembleType;

template<typename IndexType, typename ValueType, class MemorySpace>
class myEll
{
  IndexType num_rows;
  IndexType num_entries;
  cusp::array1d<IndexType, MemorySpace> column_indices;
	cusp::array1d<ValueType, MemorySpace> values;
};

typedef myEll<int, CGType, cusp::host_memory> myEll_h_CG;
typedef myEll<int, CGType, cusp::device_memory> myEll_d_CG;


typedef cusp::csr_matrix<int, CGType, cusp::host_memory> Matrix_h_CG;
typedef cusp::csr_matrix<int, CGType, cusp::device_memory> Matrix_d_CG;

typedef cusp::array1d<CGType, cusp::host_memory> Vector_h_CG;
typedef cusp::array1d<CGType, cusp::device_memory> Vector_d_CG;


typedef cusp::ell_matrix<int, CGType, cusp::device_memory> Matrix_ell_d_CG;
typedef cusp::ell_matrix<int, CGType, cusp::host_memory> Matrix_ell_h_CG;

typedef cusp::coo_matrix<int, CGType, cusp::device_memory> Matrix_coo_d_CG;
typedef cusp::coo_matrix<int, CGType, cusp::host_memory> Matrix_coo_h_CG;

typedef cusp::hyb_matrix<int, CGType, cusp::device_memory> Matrix_hyb_d_CG;
typedef cusp::hyb_matrix<int, CGType, cusp::host_memory> Matrix_hyb_h_CG;

typedef cusp::csr_matrix<int, AMGType, cusp::host_memory> Matrix_h;
typedef cusp::csr_matrix<int, AMGType, cusp::device_memory> Matrix_d;

typedef cusp::array1d<AMGType, cusp::host_memory> Vector_h;
typedef cusp::array1d<AMGType, cusp::device_memory> Vector_d;

typedef cusp::array1d<int, cusp::host_memory> IdxVector_h;
typedef cusp::array1d<int, cusp::device_memory> IdxVector_d;

typedef cusp::ell_matrix<int, AMGType, cusp::device_memory> Matrix_ell_d;
typedef cusp::ell_matrix<int, AMGType, cusp::host_memory> Matrix_ell_h;

typedef cusp::coo_matrix<int, AMGType, cusp::device_memory> Matrix_coo_d;
typedef cusp::coo_matrix<int, AMGType, cusp::host_memory> Matrix_coo_h;

typedef cusp::hyb_matrix<int, AMGType, cusp::device_memory> Matrix_hyb_d;
typedef cusp::hyb_matrix<int, AMGType, cusp::host_memory> Matrix_hyb_h;

#endif
