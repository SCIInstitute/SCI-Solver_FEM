#include <cusp/multiply.h>
#include <cusp/blas.h>
#include <types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <thrust/scan.h>
#include <error.h>

template<typename IndexType, typename ValueType>
void __global__ neighbor_count_kernel(IndexType* tri0, IndexType* tri1, IndexType* tri2, IndexType ne, IndexType* nbcount)
{
  for(int eidx = threadIdx.x; eidx < ne; eidx += gridDim.x * blockDim.x)
  {
    IndexType i = tri0[eidx];
    IndexType j = tri1[eidx];
    IndexType k = tri2[eidx];

    atomicInc((unsigned *)nbcount + i, INT_MAX);
    atomicInc((unsigned *)nbcount + j, INT_MAX);
    atomicInc((unsigned *)nbcount + k, INT_MAX);

  }

}

template<typename IndexType, typename ValueType>
void __global__ compute_nb_indices_kernel(IndexType* rowoffsets, IndexType* ele_indices, IndexType *tri0, IndexType* tri1, IndexType* tri2, IndexType nv, IndexType* column_indices, size_t num_cols, size_t pitch)
{

  for(int nidx = threadIdx.x; nidx < nv; nidx += gridDim.x * blockDim.x)
  {
    for(int i = 0; i < num_cols; i++)
    {
      column_indices[pitch * i + nidx] = -1;
    }

    int nedges = 0;
    for(int j = rowoffsets[nidx]; j < rowoffsets[nidx + 1]; j++)
    {
      IndexType jj = ele_indices[j];
      IndexType node0 = tri0[jj];
      IndexType node1 = tri1[jj];
      IndexType node2 = tri2[jj];
      if(node0 != nidx)
      {
        column_indices[pitch * nedges + nidx] = node0;
        nedges++;
      }

    }
  }

}

template<typename IndexType, typename ValueType>
void __global__ compute_ele_indices_kernel(IndexType* tri0, IndexType* tri1, IndexType* tri2, IndexType ne, IndexType* rowoffsets, IndexType* ele_indices)
{
  for(int eidx = threadIdx.x; eidx < ne; eidx += gridDim.x * blockDim.x)
  {
    IndexType i = tri0[eidx];
    IndexType j = tri1[eidx];
    IndexType k = tri2[eidx];
    IndexType starti = rowoffsets[i];
    IndexType startj = rowoffsets[j];
    IndexType startk = rowoffsets[k];
    IndexType endi = rowoffsets[i + 1];
    IndexType endj = rowoffsets[j + 1];
    IndexType endk = rowoffsets[k + 1];
    for(int n = starti; n < endi; n++)
    {
      atomicCAS(ele_indices + n, -1, eidx);
      break;
    }

    for(int n = startj; n < endj; n++)
    {
      atomicCAS(ele_indices + n, -1, eidx);
      break;
    }

    for(int n = startk; n < endk; n++)
    {
      atomicCAS(ele_indices + n, -1, eidx);
    }


  }

}

//template<>
//void trimesh2ell<Matrix_ell_d>(TriMesh* meshPtr, Matrix_ell_d &A)
//{
//  typedef typename Matrix_ell_d::value_type ValueType;
//  typedef typename Matrix_ell_d::index_type IndexType;
//  int nv = meshPtr->vertices.size();
//  int ne = meshPtr->faces.size();
//  ValueType* x = new ValueType[nv];
//  ValueType* y = new ValueType[nv];
//  IndexType* tri0 = new IndexType[ne];
//  IndexType* tri1 = new IndexType[ne];
//  IndexType* tri2 = new IndexType[ne];
//
//  for(int i = 0; i < nv; i++)
//  {
//    x[i] = meshPtr->vertices[i][0];
//    y[i] = meshPtr->vertices[i][1];
//  }
//
//  for(int i = 0; i < ne; i++)
//  {
//    tri0[i] = meshPtr->faces[i][0];
//    tri1[i] = meshPtr->faces[i][1];
//    tri2[i] = meshPtr->faces[i][2];
//  }
//
//  IndexType* d_tri0;
//  IndexType* d_tri1;
//  IndexType* d_tri2;
//  IndexType* d_nbcount;
//  //	IdxVector_d d_rowoffsets(nv+1, 0);
//  IndexType* d_rowoffsets;
//  cudaSafeCall(cudaMalloc(&d_tri0, ne * sizeof(IndexType)));
//  cudaSafeCall(cudaMalloc(&d_tri1, ne * sizeof(IndexType)));
//  cudaSafeCall(cudaMalloc(&d_tri2, ne * sizeof(IndexType)));
//  cudaSafeCall(cudaMalloc(&d_nbcount, nv * sizeof(IndexType)));
//  cudaSafeCall(cudaMalloc(&d_rowoffsets, (nv + 1) * sizeof(IndexType)));
//
//  cudaSafeCall(cudaMemset(d_nbcount, 0, nv));
//
//  size_t threads = 256;
//  size_t blocks = min((int)ceil(ne / threads), (int)65535);
//  neighbor_count_kernel<IndexType, ValueType> << <blocks, threads >> >(d_tri0, d_tri1, d_tri2, ne, d_nbcount);
//  cudaCheckError();
//
//  cudaSafeCall(cudaMemset(d_rowoffsets, 0, 1));
//  thrust::inclusive_scan(d_nbcount, d_nbcount + nv, d_rowoffsets + 1); // out-place scan
//
//  IndexType total;
//  cudaSafeCall(cudaMemcpy(&total, d_rowoffsets + nv, 1, cudaMemcpyDeviceToHost));
//
//  //  IndexType* d_ele_indices; //stores adjacent elements for each node 
//  IdxVector_d d_ele_indices(total, -1);
//  //  cudaSafeCall(cudaMalloc(&ele_indices, total * sizeof(IndexType)));
//  //  cudaSafeCall(cudaMemset(ele_indices, -1, total * sizeof(IndexType)));
//  compute_ele_indices_kernel<IndexType, ValueType> << <blocks, threads >> >(d_tri0, d_tri1, d_tri2, ne, d_rowoffsets, thrust::raw_pointer_cast(&d_ele_indices[0]));
//  cudaCheckError();
//
//  IndexType* tmpvector = thrust::raw_pointer_cast(&d_ele_indices[0]);
//
//  IndexType maxnumnb = thrust::reduce(d_nbcount, d_nbcount + nv, -1, thrust::maximum<IndexType > ()) * 2;
//
//  A.column_indices.resize(nv, maxnumnb, nv);
//  A.values.resize(nv, maxnumnb, nv);
//  A.num_cols = nv;
//  A.num_rows = nv;
//
//  threads = 256;
//  blocks = min((int)ceil(nv / threads), (int)65535);
//
//  compute_nb_indices_kernel<IndexType, ValueType> << <blocks, threads >> >(thrust::raw_pointer_cast(&d_rowoffsets[0]), thrust::raw_pointer_cast(&d_ele_indices[0]), d_tri0, d_tri1, d_tri2, nv, thrust::raw_pointer_cast(&A.column_indices.values[0]), A.column_indices.num_cols, A.column_indices.pitch);
//
//  int meiyongde = 0;
//  int a = meiyongde + 1;
//}

template<typename MeshType>
void populateStiffnessMatrix(MeshType mesh, Matrix_ell_d_CG &stiffnessMatrix, int numVerts, bool verbose)
{
  int maxsize = 0;
  int num_entries = 0;
  const int X = Matrix_ell_d_CG::invalid_index;

  for(int i = 0; i < numVerts; i++)
  {
    num_entries += (int)mesh->neighbors[i].size();
    maxsize = std::max(maxsize, (int)mesh->neighbors[i].size());
  }
  num_entries += numVerts;
  maxsize += 1; // should include itself

  if( verbose )
    std::cout << "Constructing Matrix_ell_h_CG A";
  Matrix_ell_h_CG A(numVerts, numVerts, num_entries, maxsize, 32);
  if( verbose )
    std::cout << "Adding values to matrix A";
  for(int i = 0; i < numVerts; i++)
  {
    A.column_indices(i, 0) = i;
    for(int j = 1; j < maxsize; j++)
    {
      A.values(i, j) = 0.0;
      if(j < mesh->neighbors[i].size() + 1)
      {
        A.column_indices(i, j) = mesh->neighbors[i][j - 1];
      }
      else
      {
        A.column_indices(i, j) = X;
      }
    }
  }
  if( verbose )
    std::cout << "Copying A to device";
  //A_d = Matrix_ell_d_CG(A);
  stiffnessMatrix = A;
}

template<>
void tetmesh2ell<Matrix_ell_d_CG>(TetMesh* meshPtr, Matrix_ell_d_CG &A_d,
		                          bool generateStiffnessMatrix, bool verbose)
{
  int nv = meshPtr->vertices.size();

  meshPtr->need_neighbors();
  for(int i = 0; i < nv; i++)
  {
    std::sort(meshPtr->neighbors[i].begin(), meshPtr->neighbors[i].end());
  }

  if( generateStiffnessMatrix )
  {
    populateStiffnessMatrix<TetMesh*>(meshPtr, A_d, nv, verbose);
  }
}

template<>
void trimesh2csr<Matrix_d_CG>(TriMesh* meshPtr, Matrix_d_CG &A_d)
{
  typedef typename Matrix_d_CG::value_type ValueType;
  typedef typename Matrix_d_CG::index_type IndexType;

  int nv = meshPtr->vertices.size();
  int ne = meshPtr->faces.size();



  meshPtr->need_neighbors();
  for(int i = 0; i < nv; i++)
  {
    std::sort(meshPtr->neighbors[i].begin(), meshPtr->neighbors[i].end());
  }

  int maxsize = 0;
  int num_entries = 0;
  for(int i = 0; i < nv; i++)
  {
    num_entries += (int)meshPtr->neighbors[i].size();
    maxsize = std::max(maxsize, (int)meshPtr->neighbors[i].size());
  }
  num_entries = num_entries / 2 + nv;
  maxsize += 1; // should include itself

  vector<IndexType> rowoffsets;
  vector<IndexType> idxj;
  rowoffsets.reserve(nv + 1);
  idxj.reserve(num_entries);

  rowoffsets.push_back(0);
  int count;
  for(int i = 0; i < nv; i++)
  {
    count = 0;
    idxj.push_back(i);
    for(int j = 0; j < meshPtr->neighbors[i].size(); j++)
    {
      if(meshPtr->neighbors[i][j] > i)
      {
        count++;
        idxj.push_back(meshPtr->neighbors[i][j]);
      }
    }
    rowoffsets.push_back(rowoffsets[i] + count + 1);

  }

  int realsz = idxj.size();
  vector<ValueType> values(realsz, 0.0);


  Matrix_h_CG A(nv, nv, realsz);
  A.row_offsets = rowoffsets;
  A.column_indices = idxj;
  A.values = values;

  A_d = A;

  A.resize(0, 0, 0);

}

template<>
void trimesh2ell<Matrix_ell_d_CG>(TriMesh* meshPtr, Matrix_ell_d_CG &A_d,
		                          bool generateStiffnessMatrix, bool verbose)
{
  int nv = meshPtr->vertices.size();

  meshPtr->need_neighbors();
  for(int i = 0; i < nv; i++)
  {
    std::sort(meshPtr->neighbors[i].begin(), meshPtr->neighbors[i].end());
  }

  if( generateStiffnessMatrix )
  {
    populateStiffnessMatrix<TriMesh*>(meshPtr, A_d, nv, verbose);
  }
}

template<typename IndexType, typename ValueType>
__global__ void convert_kernel(IndexType* rowoff1, IndexType* colidx1, ValueType* values1, IndexType* rowidx2, IndexType* colidx2, ValueType* values2, int num_rows)
{
  for(int ridx = blockIdx.x * blockDim.x + threadIdx.x; ridx < num_rows; ridx++)
  {
    IndexType start1 = rowoff1[ridx];
    IndexType end1 = rowoff1[ridx + 1];
    IndexType start2 = start1 * 2 - ridx;

    rowidx2[start2] = ridx;
    colidx2[start2] = ridx;
    values2[start2] = values1[start1];
    for(int i = start1 + 1; i < end1; i++)
    {
      ValueType v = values1[i];
      IndexType col = colidx1[i];
      IndexType loc = start2 + 1 + 2 * (i - start1 - 1);
      rowidx2[loc] = ridx;
      colidx2[loc] = col;
      values2[loc] = v;
      rowidx2[loc + 1] = col;
      colidx2[loc + 1] = ridx;
      values2[loc + 1] = v;
    }
  }

}

void convertSym2gen(Matrix_d_CG &Acsr, Matrix_coo_d_CG &Aout)
{
  typedef typename Matrix_d_CG::value_type ValueType;
  typedef typename Matrix_d_CG::index_type IndexType;

  int num_entries = Acsr.num_entries;
  int num_rows = Acsr.num_rows;
  int num_cols = Acsr.num_cols;
  Aout.resize(num_rows, num_cols, 2 * num_entries - num_rows);
  int threads = 256;
  int blocks = std::min((int)ceil((double)num_rows / threads), 65535);
  IndexType* rowoff1 = thrust::raw_pointer_cast(&Acsr.row_offsets[0]);
  IndexType* colidx1 = thrust::raw_pointer_cast(&Acsr.column_indices[0]);
  ValueType* values1 = thrust::raw_pointer_cast(&Acsr.values[0]);
  IndexType* rowidx2 = thrust::raw_pointer_cast(&Aout.row_indices[0]);
  IndexType* colidx2 = thrust::raw_pointer_cast(&Aout.column_indices[0]);
  ValueType* values2 = thrust::raw_pointer_cast(&Aout.values[0]);


  convert_kernel<IndexType, ValueType> << <blocks, threads >> >(rowoff1, colidx1, values1, rowidx2, colidx2, values2, num_rows);


}

template <class Matrix, class Vector>
void computeResidual(const Matrix& A, const Vector& x, const Vector& b, Vector& r)
{
  cusp::multiply(A, x, r);
  cusp::blas::axpby(r, b, r, -1, 1);
}

//template void trimesh2csr<int,float,cusp::device_memory>(const TriMesh* meshPtr, struct cudaCSRGraph& csrgraph);
//template void tetmesh2csr<int,float,cusp::device_memory>(const TetMesh* meshPtr, struct cudaCSRGraph& csrgraph);
//template void trimesh2csr<int,float,cusp::host_memory>(const TriMesh* meshPtr, struct cudaCSRGraph& csrgraph);
//template void tetmesh2csr<int,float,cusp::host_memory>(const TetMesh* meshPtr, struct cudaCSRGraph& csrgraph);

template void computeResidual<Matrix_ell_h, Vector_h>(const Matrix_ell_h& A, const Vector_h& x, const Vector_h& b, Vector_h& r);
template void computeResidual<Matrix_hyb_d, Vector_d>(const Matrix_hyb_d& A, const Vector_d& x, const Vector_d& b, Vector_d& r);
