
#include <smoothedMG/smoothedMG_amg_level.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <amg_level.h>

#include<types.h>
#include<cutil.h>
#include <vector>

using namespace std;

template <class Matrix, class Vector>
SmoothedMG_AMG_Level<Matrix, Vector>::SmoothedMG_AMG_Level(AMG<Matrix, Vector> *amg) : AMG_Level<Matrix, Vector>(amg)
{
  //  strength=Strength<Matrix,Vector>::allocate(amg->cfg);
  prosmoothomega = amg->cfg.AMG_Config::getParameter<double> ("pro_omega");
  aggregator = Aggregator<Matrix, Vector>::allocate(amg->cfg); // DHL
  DS_type = amg->cfg.AMG_Config::getParameter<int> ("DS_type");
  metis_size = amg->cfg.AMG_Config::getParameter<int> ("metis_size");
  mesh_type = amg->cfg.AMG_Config::getParameter<int> ("mesh_type");
  part_max_size = amg->cfg.AMG_Config::getParameter<int> ("part_max_size");
  //  interpolator=Interpolator<Matrix,Vector>::allocate(amg->cfg);
}

template <class Matrix, class Vector>
SmoothedMG_AMG_Level<Matrix, Vector>::~SmoothedMG_AMG_Level()
{
  //  delete strength;
  //  delete aggregator;
  //  delete interpolator;
}

__global__ void packcoo_kernel(int num_entries,
                               int* row_indices,
                               int* column_indices,
                               int* aggridx,
                               int* partidx,
                               int* partlabel)
{
  int entryidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(entryidx < num_entries)
  {
    int row = row_indices[entryidx];
    int col = column_indices[entryidx];
    int l = partlabel[row];
    int partstart = aggridx[partidx[l]];
    unsigned int newindex = row - partstart;
    newindex <<= 16;
    newindex += col - partstart;
    row_indices[entryidx] = newindex;
  }
}

__global__ void matrixpermute_kernel(int np, int num_entries,
                                     int* row_indices,
                                     int* column_indices,
                                     AMGType* values,
                                     int* entrypartlabel,
                                     int* permutation,
                                     int* partitionlabel)
{
  int entryidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(entryidx < num_entries)
  {
    int oldrow = row_indices[entryidx];
    int oldcol = column_indices[entryidx];
    int newrow = permutation[oldrow];
    int newcol = permutation[oldcol];
    row_indices[entryidx] = newrow;
    column_indices[entryidx] = newcol;
    int rowpartition = partitionlabel[newrow];
    int colpartition = partitionlabel[newcol];

    if(rowpartition == colpartition) //inside point
    {
      if(newcol > newrow)
      {
        entrypartlabel[entryidx] = rowpartition;
      }
      else
        entrypartlabel[entryidx] = INT_MAX;
    }
    else
    {
      entrypartlabel[entryidx] = rowpartition + np;

    }
  }
}

__global__ void matrixpermute_csr_kernel(int np, int num_entries,
                                     int* row_indices,
                                     int* column_indices,
                                     AMGType* values,
                                     int* entrypartlabel,
                                     int* permutation,
                                     int* partitionlabel)
{
  int entryidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(entryidx < num_entries)
  {
    int oldrow = row_indices[entryidx];
    int oldcol = column_indices[entryidx];
    int newrow = permutation[oldrow];
    int newcol = permutation[oldcol];
    row_indices[entryidx] = newrow;
    column_indices[entryidx] = newcol;
    int rowpartition = partitionlabel[newrow];
    int colpartition = partitionlabel[newcol];

    if(rowpartition == colpartition) //inside point
    {
        entrypartlabel[entryidx] = rowpartition;
    }
    else
    {
      entrypartlabel[entryidx] = rowpartition + np;
    }
  }
}

template <>
void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateMatrixCsr(IdxVector_d &permutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel)
{

	int nnode = A_d.num_rows;
  int numpart = partitionIdx.size() - 1;
  int numentries = A_d.num_entries;
  //  Matrix_coo_d Atmpcoo_d = A;
  Acoo_d = A_d; //Matrix_coo_d(A.num_rows, A.num_cols, A.num_entries);
  cusp::array1d<int, cusp::device_memory> entrypartlabel(numentries, -1);

  //  printf("before permute: \n");
  //  cusp::print(Acoo_d);

  size_t blocksize = 256;
  size_t blocknum = ceil((float)numentries / (float)blocksize);
  if(blocknum > 65535) printf("too many blocks!!\n");

  matrixpermute_csr_kernel << <blocknum, blocksize >> >(numpart, A_d.num_entries,
                                                    thrust::raw_pointer_cast(&Acoo_d.row_indices[0]),
                                                    thrust::raw_pointer_cast(&Acoo_d.column_indices[0]),
                                                    thrust::raw_pointer_cast(&Acoo_d.values[0]),
                                                    thrust::raw_pointer_cast(&entrypartlabel[0]),
                                                    thrust::raw_pointer_cast(&permutation[0]),
                                                    thrust::raw_pointer_cast(&partitionlabel[0]));
	cudaCheckError();

  typedef IdxVector_d::iterator IntIterator;
  typedef Vector_d::iterator FloatIterator;
  typedef thrust::tuple<IntIterator, IntIterator, FloatIterator> IteratorTuple;

  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
  ZipIterator iter(thrust::make_tuple(Acoo_d.row_indices.begin(), Acoo_d.column_indices.begin(), Acoo_d.values.begin()));

  thrust::sort_by_key(entrypartlabel.begin(), entrypartlabel.end(), iter);

	IdxVector_d redoutput(numentries);
  IdxVector_d redvalue(numentries, 1);
  IdxVector_d redoutputkey(2 * numpart);
  IdxVector_d redoutputvalue(2 * numpart);

  thrust::reduce_by_key(entrypartlabel.begin(), entrypartlabel.end(), redvalue.begin(), redoutputkey.begin(), redoutputvalue.begin());
  //	cusp::print(redoutputkey);
  //	cusp::print(redoutputvalue);
  int innum = thrust::reduce(redoutputvalue.begin(), redoutputvalue.begin() + numpart);
  int outnum = numentries - innum;
  //  int outnum = redoutputvalue[0];
  //  int innum = numentries - outnum - redoutputvalue[numpart + 1];
  //  IntIterator res = thrust::max_element(redoutputvalue.begin() + 1, redoutputvalue.end() - 1);
  IntIterator res = thrust::max_element(redoutputvalue.begin(), redoutputvalue.begin() + numpart);
  largest_num_entries = *res;

  printf("CSR largest_num_entries is %d\n", largest_num_entries);
  printf("CSR inside number is %d, outside number is %d\n", innum, outnum);

  IdxVector_d AinBlockIdx(numpart + 1);
  thrust::copy(redoutputvalue.begin(), redoutputvalue.begin() + numpart, AinBlockIdx.begin());
  thrust::exclusive_scan(AinBlockIdx.begin(), AinBlockIdx.end(), AinBlockIdx.begin());
  AinBlockIdx[numpart] = innum;

  IdxVector_d AoutBlockIdx(numpart + 1);
  thrust::copy(redoutputvalue.begin() + numpart, redoutputvalue.begin() + 2 * numpart, AoutBlockIdx.begin());
  thrust::exclusive_scan(AoutBlockIdx.begin(), AoutBlockIdx.end(), AoutBlockIdx.begin());
  AoutBlockIdx[numpart] = outnum;

  //    printf("AinBlockIdx after scan: \n");
  //    cusp::print(AinBlockIdx);

  AinBlockIdx_d = AinBlockIdx;
  AoutBlockIdx_d = AoutBlockIdx;

	AinCSR_d = Matrix_d(A_d.num_rows, A_d.num_cols, innum);
  Aout_d = Matrix_coo_d(A_d.num_rows, A_d.num_cols, outnum);
	Matrix_coo_d AinCOO_tmp(A_d.num_rows, A_d.num_cols, innum);

  thrust::copy_n(Acoo_d.row_indices.begin(), innum, AinCOO_tmp.row_indices.begin());
  thrust::copy_n(Acoo_d.column_indices.begin(), innum, AinCOO_tmp.column_indices.begin());
  thrust::copy_n(Acoo_d.values.begin(), innum, AinCOO_tmp.values.begin());
  AinCOO_tmp.sort_by_row_and_column();
	
	AinCSR_d.column_indices = AinCOO_tmp.column_indices;
	AinCSR_d.values = AinCOO_tmp.values;
	cusp::detail::indices_to_offsets(AinCOO_tmp.row_indices, AinCSR_d.row_offsets);
	thrust::reduce_by_key(AinCOO_tmp.row_indices.begin(), AinCOO_tmp.row_indices.end(), redvalue.begin(), redoutputkey.begin(), redoutputvalue.begin());
	IntIterator res2 = thrust::max_element(redoutputvalue.begin(), redoutputvalue.begin() + AinCOO_tmp.num_rows);
	largest_num_per_row = *res2;

  thrust::copy_n(Acoo_d.row_indices.begin() + innum, outnum, Aout_d.row_indices.begin());
  thrust::copy_n(Acoo_d.column_indices.begin() + innum, outnum, Aout_d.column_indices.begin());
  thrust::copy_n(Acoo_d.values.begin() + innum, outnum, Aout_d.values.begin());
  Aout_d.sort_by_row_and_column();

  Acoo_d.sort_by_row_and_column();
  A_d = Acoo_d;

  printf("Finished generateMatrixCSR_d!!\n");
}

template <>
void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateMatrixSymmetric_d(IdxVector_d &permutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel)
{
//  cudaDeviceSynchronize();
//  cudaCheckError();
  
  int nnode = A_d.num_rows;
  int numpart = partitionIdx.size() - 1;
  int numentries = A_d.num_entries;
  Acoo_d = A_d; //Matrix_coo_d(A.num_rows, A.num_cols, A.num_entries);
  cusp::array1d<int, cusp::device_memory> entrypartlabel(numentries, -1);

    printf("before permute: \n");
  //  cusp::print(Acoo_d);

  size_t blocksize = 256;
  size_t blocknum = ceil((float)numentries / (float)blocksize);
  if(blocknum > 65535) printf("too many blocks!!\n");
  
//  printf("Before call to matrixpermute_kernel:\n");
//  printf("\t numentries = %d blocknum = %d \n", numentries, blocknum);
//  int t;
//  cin>>t;
//  
//  cudaDeviceSynchronize();
//  cudaCheckError();
  
  matrixpermute_kernel << <blocknum, blocksize >> >(numpart, A_d.num_entries,
                                                    thrust::raw_pointer_cast(&Acoo_d.row_indices[0]),
                                                    thrust::raw_pointer_cast(&Acoo_d.column_indices[0]),
                                                    thrust::raw_pointer_cast(&Acoo_d.values[0]),
                                                    thrust::raw_pointer_cast(&entrypartlabel[0]),
                                                    thrust::raw_pointer_cast(&permutation[0]),
                                                    thrust::raw_pointer_cast(&partitionlabel[0]));
//  cudaCheckError(); // DHL

  typedef IdxVector_d::iterator IntIterator;
  typedef Vector_d::iterator FloatIterator;
  typedef thrust::tuple<IntIterator, IntIterator, FloatIterator> IteratorTuple;

  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
  ZipIterator iter(thrust::make_tuple(Acoo_d.row_indices.begin(), Acoo_d.column_indices.begin(), Acoo_d.values.begin()));

//  printf("before sort: \n");
  thrust::sort_by_key(entrypartlabel.begin(), entrypartlabel.end(), iter);
  printf("partition number is %d\n", numpart);

  IdxVector_d redoutput(numentries);
  IdxVector_d redvalue(numentries, 1);
  IdxVector_d redoutputkey(2 * numpart + 1);
  IdxVector_d redoutputvalue(2 * numpart + 1);

  thrust::reduce_by_key(entrypartlabel.begin(), entrypartlabel.end(), redvalue.begin(), redoutputkey.begin(), redoutputvalue.begin());
  //	cusp::print(redoutputkey);
  //	cusp::print(redoutputvalue);
  int innum = thrust::reduce(redoutputvalue.begin(), redoutputvalue.begin() + numpart);
  int outnum = numentries - innum - redoutputvalue[numpart * 2];
  //  int outnum = redoutputvalue[0];
  //  int innum = numentries - outnum - redoutputvalue[numpart + 1];
  //  IntIterator res = thrust::max_element(redoutputvalue.begin() + 1, redoutputvalue.end() - 1);
  IntIterator res = thrust::max_element(redoutputvalue.begin(), redoutputvalue.begin() + numpart);
  largest_num_entries = *res;

  printf("largest_num_entries is %d\n", largest_num_entries);
  printf("inside number is %d, outside number is %d\n", innum, outnum);

  IdxVector_d AinBlockIdx(numpart + 1);
  thrust::copy(redoutputvalue.begin(), redoutputvalue.begin() + numpart, AinBlockIdx.begin());
  thrust::exclusive_scan(AinBlockIdx.begin(), AinBlockIdx.end(), AinBlockIdx.begin());
  AinBlockIdx[numpart] = innum;

  IdxVector_d AoutBlockIdx(numpart + 1);
  thrust::copy(redoutputvalue.begin() + numpart, redoutputvalue.begin() + 2 * numpart, AoutBlockIdx.begin());
  thrust::exclusive_scan(AoutBlockIdx.begin(), AoutBlockIdx.end(), AoutBlockIdx.begin());
  AoutBlockIdx[numpart] = outnum;

  AinBlockIdx_d = AinBlockIdx;
  AoutBlockIdx_d = AoutBlockIdx;

  AinSysCoo_d = Matrix_coo_d(A_d.num_rows, A_d.num_cols, innum);
  AoutSys_d = Matrix_coo_d(A_d.num_rows, A_d.num_cols, outnum);

  thrust::copy_n(Acoo_d.row_indices.begin(), innum, AinSysCoo_d.row_indices.begin());
  thrust::copy_n(Acoo_d.column_indices.begin(), innum, AinSysCoo_d.column_indices.begin());
  thrust::copy_n(Acoo_d.values.begin(), innum, AinSysCoo_d.values.begin());
  AinSysCoo_d.sort_by_row_and_column();

  thrust::copy_n(Acoo_d.row_indices.begin() + innum, outnum, AoutSys_d.row_indices.begin());
  thrust::copy_n(Acoo_d.column_indices.begin() + innum, outnum, AoutSys_d.column_indices.begin());
  thrust::copy_n(Acoo_d.values.begin() + innum, outnum, AoutSys_d.values.begin());
  AoutSys_d.sort_by_row_and_column();

  //pack column of Ain to row
  blocknum = ceil((float)innum / (float)blocksize);
  if(blocknum > 65535) printf("too many blocks!!\n");

  packcoo_kernel << <blocknum, blocksize >> >(innum,
                                              thrust::raw_pointer_cast(&AinSysCoo_d.row_indices[0]),
                                              thrust::raw_pointer_cast(&AinSysCoo_d.column_indices[0]),
                                              thrust::raw_pointer_cast(&aggregateIdx[0]),
                                              thrust::raw_pointer_cast(&partitionIdx[0]),
                                              thrust::raw_pointer_cast(&partitionlabel[0]));
//  cudaCheckError(); // DHL
  Acoo_d.sort_by_row_and_column();
  A_d = Acoo_d;

  printf("Finished generateMatrixSymmetric_d!!\n");

}

//template <>
//void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateMatrixSymmetric(int* permutation, int* aggregateIdx, int* partitionIdx, int* partitionlabel)
//{
//  int nnode = A.num_rows;
//
//  Acoo = Matrix_coo_h(A.num_rows, A.num_cols, A.num_entries);
//  vector<int> idxiin;
//  vector<int> idxjin;
//  vector<AMGType> valuein;
//
//  vector<int> idxiout;
//  vector<int> idxjout;
//  vector<AMGType> valueout;
//
//  int k = 0;
//  int maxvalue1 = 0;
//  for(int i = 0; i < nn; i++)
//  {
//    for(int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
//    {
//      Acoo.row_indices[k] = permutation[i];
//      Acoo.column_indices[k] = permutation[A.column_indices[j]];
//      Acoo.values[k] = A.values[j];
//      k++;
//    }
//  }
//
//  Acoo.sort_by_row_and_column();
//
//  int previousBlockout = 0;
//  int previousBlockin = 0;
//  vector<int> AoutBlockIdx;
//  vector<int> AinBlockIdx;
//  AoutBlockIdx.push_back(0);
//  AinBlockIdx.push_back(0);
//
//  for(int i = 0; i < Acoo.num_entries; i++)
//  {
//    int idxi = Acoo.row_indices[i];
//    int idxj = Acoo.column_indices[i];
//
//    int blkidxi = partitionlabel[idxi];
//    int blkidxj = partitionlabel[idxj];
//    if(blkidxi != blkidxj)//boudary points
//    {
//      if(blkidxi != previousBlockout)
//      {
//        AoutBlockIdx.push_back(idxiout.size());
//        previousBlockout = blkidxi;
//      }
//      idxiout.push_back(idxi);
//      idxjout.push_back(idxj);
//      valueout.push_back((AMGType)Acoo.values[i]);
//    }
//    else //inside points
//    {
//      if(idxj > idxi)
//      {
//        if(blkidxi != previousBlockin)
//        {
//          AinBlockIdx.push_back(idxiin.size());
//          previousBlockin = blkidxi;
//        }
//        int blockstart = aggregateIdx[partitionIdx[blkidxi]];
//        int blockend = aggregateIdx[partitionIdx[blkidxi + 1]];
//        if(idxi < blockstart || idxi >= blockend || idxj < blockstart || idxj >= blockend)
//          cout << "index out of range!!" << endl;
//
//        idxiin.push_back(idxi - blockstart);
//        idxjin.push_back(idxj - blockstart);
//        valuein.push_back((AMGType)Acoo.values[i]);
//      }
//    }
//  }
//
//  AinSysCoo.row_indices.resize(idxiin.size()); // row_indices includes both row and col indices packed into 1 int
//  for(int i = 0; i < idxiin.size(); i++)
//  {
//    unsigned int tmpidx = idxiin[i];
//    tmpidx <<= 16;
//    tmpidx += idxjin[i];
//    AinSysCoo.row_indices[i] = tmpidx;
//
//  }
//  //  AinSysCoo.row_indices = idxiin;
//  //  AinSysCoo.column_indices = idxjin;
//  AinBlockIdx.push_back(idxiin.size());
//  AinSysCoo.values = valuein;
//  AinSysCoo.num_entries = idxiin.size();
//  AinSysCoo.num_rows = nnode;
//
//  largest_num_entries = 0;
//  for(int i = 0; i < AinBlockIdx.size(); i++)
//  {
//    int num = AinBlockIdx[i + 1] - AinBlockIdx[i];
//
//    if(num > largest_num_entries)
//      largest_num_entries = num;
//  }
//  cout << "Largest number of entries for AinSysCoo is " << largest_num_entries << endl;
//
//  AoutBlockIdx.push_back(idxiout.size());
//  AoutSys.row_indices = idxiout;
//  AoutSys.column_indices = idxjout;
//  AoutSys.num_entries = idxiout.size();
//  AoutSys.values = valueout;
//  AoutSys.num_rows = nnode;
//  IndexType* tmpidx0 = thrust::raw_pointer_cast(&AoutSys.column_indices[0]);
//  IndexType* tmpidx1 = thrust::raw_pointer_cast(&AoutSys.row_indices[0]);
//  //	AinSysCoo_d = AinSysCoo;
//
//  IdxVector_h AoutBlockIdx_h = AoutBlockIdx;
//  IndexType* tmpidx2 = thrust::raw_pointer_cast(&AoutBlockIdx_h[0]);
//  AoutBlockIdx_d = AoutBlockIdx_h;
//
//  IdxVector_h AinBlockIdx_h = AinBlockIdx;
//  IndexType* tmpidx3 = thrust::raw_pointer_cast(&AinBlockIdx_h[0]);
//  AinBlockIdx_d = AinBlockIdx_h;
//  A = Acoo;
//  A_d = Acoo;
//
//}
//
////template <>
////void SmoothedMG_AMG_Level<Matrix_d, Vector_d>::generateMatrix(int* permutation, int* aggregateIdx, int* partitionIdx, int* partitionlabel)
////{
////
////}
//
//template <>
//void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateSmoothingMatrix(Matrix_h& S, Vector_h D, ValueType omega)
//{
//
//  for(int i = 0; i < S.num_rows; i++)
//  {
//    for(int j = S.row_offsets[i]; j < S.row_offsets[i + 1]; j++)
//    {
//      if(S.column_indices[j] == i)//diagonal
//        S.values[j] = 1 - omega;
//      else
//      {
//
//        ValueType dia = D[i];
//        S.values[j] = -S.values[j] / dia;
//
//      }
//
//    }
//  }
//}

//template <>
//void SmoothedMG_AMG_Level<Matrix_d, Vector_d>::generateSmoothingMatrix(Matrix_d& S, Vector_d D, ValueType omega)
//{
//
//}
//
//template <>
//void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateProlongator(int* aggregateIdx, int* partitionIdx)
//{
//  Vector_h D(this->nn);
//  cusp::detail::extract_diagonal(A, D);
//
//  Matrix_h S = A;
//  generateSmoothingMatrix(S, D, prosmoothomega);
//
//  int num_aggregates = this->nnout;
//
//  prolongator.resize(this->nn);
//  for(int i = 0; i < num_aggregates; i++)
//  {
//    Vector_h tentativePi(this->nn, 0.0);
//    Vector_h Pi(this->nn, 0.0);
//    for(int j = aggregateIdx[i]; j < aggregateIdx[i + 1]; j++)
//      tentativePi[j] = 1.0;
//
//    cusp::multiply(S, tentativePi, Pi);
//
//    for(int j = aggregateIdx[i]; j < aggregateIdx[i + 1]; j++)
//    {
//
//      prolongator[j] = Pi[j];
//    }
//  }
//}

template <typename T>
struct scaled_multiply
{
    const T lambda;

    scaled_multiply(const T lambda) : lambda(lambda) {}

    __host__ __device__
    T operator()(const T& x, const T& y) const
    {
        return lambda * x * y;
    }
};

template <>
void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateProlongatorFull_d(IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx)
{
  int num_aggregates = this->nnout;
  Matrix_coo_d &S = Acoo_d;
  Matrix_coo_d T(A_d.num_rows, num_aggregates, A_d.num_rows);
  thrust::sequence(T.row_indices.begin(), T.row_indices.end());
  cusp::detail::offsets_to_indices(aggregateIdx, T.column_indices);
  thrust::fill(T.values.begin(), T.values.end(), 1);
  const AMGType lambda = prosmoothomega;

  // temp <- -lambda * S(i,j) * T(j,k)
  Matrix_coo_d temp(S.num_rows, T.num_cols, S.num_entries + T.num_entries);
  thrust::copy(S.row_indices.begin(), S.row_indices.end(), temp.row_indices.begin());
  thrust::gather(S.column_indices.begin(), S.column_indices.end(), T.column_indices.begin(), temp.column_indices.begin());
  thrust::transform(S.values.begin(), S.values.end(),
                    thrust::make_permutation_iterator(T.values.begin(), S.column_indices.begin()),
                    temp.values.begin(),
                    scaled_multiply<AMGType > (-lambda));
                    

  // temp <- D^-1
  {
    Vector_d D(S.num_rows);
    cusp::detail::extract_diagonal(S, D);
    thrust::transform(temp.values.begin(), temp.values.begin() + S.num_entries,
                      thrust::make_permutation_iterator(D.begin(), S.row_indices.begin()),
                      temp.values.begin(),
                      thrust::divides<AMGType > ());
  }

  // temp <- temp + T
  thrust::copy(T.row_indices.begin(), T.row_indices.end(), temp.row_indices.begin() + S.num_entries);
  thrust::copy(T.column_indices.begin(), T.column_indices.end(), temp.column_indices.begin() + S.num_entries);
  thrust::copy(T.values.begin(), T.values.end(), temp.values.begin() + S.num_entries);

  // sort by (I,J)
  cusp::detail::sort_by_row_and_column(temp.row_indices, temp.column_indices, temp.values);

  // compute unique number of nonzeros in the output
  // throws a warning at compile (warning: expression has no effect)
  IndexType NNZ = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.end(), temp.column_indices.end())) - 1,
                                        thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())) + 1,
                                        IndexType(0),
                                        thrust::plus<IndexType > (),
                                        thrust::not_equal_to< thrust::tuple<IndexType, IndexType> >()) + 1;

  //	printf("NNZ is: %d\n", NNZ);

  // allocate space for output
  P_d.resize(temp.num_rows, temp.num_cols, NNZ);

  // sum values with the same (i,j)
  thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.end(), temp.column_indices.end())),
                        temp.values.begin(),
                        thrust::make_zip_iterator(thrust::make_tuple(P_d.row_indices.begin(), P_d.column_indices.begin())),
                        P_d.values.begin(),
                        thrust::equal_to< thrust::tuple<IndexType, IndexType> >(),
                        thrust::plus<ValueType > ());

  cusp::transpose(P_d, R_d);
  prolongatorFull_d = P_d;

  //	printf("prolongatorFull_d:\n");
  //	cusp::print(prolongatorFull_d);
  restrictorFull_d = R_d;
  //	prolongatorFull = prolongatorFull_d;

}
//
//template <>
//void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateProlongatorFull(int* aggregateIdx, int* partitionIdx)
//{
//  Vector_h D(this->nn);
//  cusp::detail::extract_diagonal(A, D);
//
//  Matrix_h S = A;
//  generateSmoothingMatrix(S, D, prosmoothomega);
//
//  int num_aggregates = this->nnout;
//
//  Matrix_coo_h tentativeP(A.num_rows, num_aggregates, A.num_rows);
//  for(int i = 0; i < num_aggregates; i++)
//  {
//    for(int j = aggregateIdx[i]; j < aggregateIdx[i + 1]; j++)
//    {
//
//      tentativeP.row_indices[j] = j;
//      tentativeP.column_indices[j] = i;
//      tentativeP.values[j] = 1.0;
//    }
//  }
//  tentativeP.sort_by_row_and_column();
//
//  printf("tentative prolongator num_rows=%d, mum_cols=%d, num_entries=%d\n", tentativeP.num_rows, tentativeP.num_cols, tentativeP.num_entries);
//
//  cusp::multiply(S, tentativeP, prolongatorFull);
//  printf("prolongatorFull num_rows=%d, mum_cols=%d, num_entries=%d\n", prolongatorFull.num_rows, prolongatorFull.num_cols, prolongatorFull.num_entries);
//}

//template <>
//void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateNextLevelMatrix(Matrix_h& Athislevel, Vector_h & prolongator)
//{
//  Matrix_coo_h Atmp;
//  IdxVector_h aggrLabel(this->nn);
//  for(int i = 0; i<this->nnout; i++)
//  {
//    for(int j = aggregateIdx[i]; j < aggregateIdx[i + 1]; j++)
//    {
//      aggrLabel[j] = i;
//    }
//  }
//
//  vector<IndexType> idxi;
//  vector<IndexType> idxj;
//  vector<ValueType> values;
//
//  for(int i = 0; i<this->nnout; i++)
//  {
//    vector<ValueType> vals(this->nnout, 0.0);
//    for(int row = aggregateIdx[i]; row < aggregateIdx[i + 1]; row++)
//    {
//      for(int roff = Athislevel.row_offsets[row]; roff < Athislevel.row_offsets[row + 1]; roff++)
//      {
//        IndexType col = Athislevel.column_indices[roff];
//        vals[aggrLabel[col]] += Athislevel.values[roff] * prolongator[row] * prolongator[col];
//      }
//    }
//
//    for(int j = 0; j<this->nnout; j++)
//    {
//      if(vals[j] != 0.0)
//      {
//
//        idxi.push_back(i);
//        idxj.push_back(j);
//        values.push_back(vals[j]);
//      }
//
//    }
//
//
//  }
//
//  Atmp.row_indices = idxi;
//  Atmp.column_indices = idxj;
//  Atmp.num_rows = this->nnout;
//  Atmp.num_cols = this->nnout;
//  Atmp.num_entries = idxi.size();
//  Atmp.values = values;
//
//  Matrix_h& Anextlevel = this->next->getA();
//  Anextlevel = Atmp;
//
//
//
//
//}

//template <>
//void SmoothedMG_AMG_Level<Matrix_d, Vector_d>::generateNextLevelMatrix(Matrix_d& Athislevel, Vector_d& prolongator)
//{
//}

template <>
void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateNextLevelMatrixFull_d()
{
  Matrix_coo_d AP;
  Matrix_coo_d Atmp = A_d;
  cusp::multiply(Atmp, P_d, AP);
  Matrix_d& Anextlevel2 = this->next->getA_d();
  Matrix_coo_d tmpmtx;
  cusp::multiply(R_d, AP, tmpmtx);
  //	Anextlevel = tmpmtx;
  Anextlevel2 = tmpmtx;
  printf("Anextlevel num_rows=%d, num_cols=%d, num_entries=%d\n", Anextlevel2.num_rows, Anextlevel2.num_cols, Anextlevel2.num_entries);
}

//template <>
//void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateNextLevelMatrixFull(Matrix_h& Athislevel, Vector_h & prolongator)
//{
//
//  Matrix_coo_h R;
//  cusp::transpose(prolongatorFull, R);
//
//  Matrix_coo_h AP;
//  cusp::multiply(Athislevel, prolongatorFull, AP);
//  Matrix_h& Anextlevel = this->next->getA();
//  Matrix_coo_h tmpmtx;
//  //  cusp::multiply(R, AP, Anextlevel);
//  cusp::multiply(R, AP, tmpmtx);
//  Anextlevel = tmpmtx;
//  restrictorFull_d = R;
//  prolongatorFull_d = prolongatorFull;
//  int* tmpcolidx1 = thrust::raw_pointer_cast(&tmpmtx.row_indices[0]);
//  int* tmprowidx1 = thrust::raw_pointer_cast(&tmpmtx.column_indices[0]);
//
//  int* tmpcolidx = thrust::raw_pointer_cast(&Anextlevel.column_indices[0]);
//  int* tmprowidx = thrust::raw_pointer_cast(&Anextlevel.row_offsets[0]);
//
//  printf("Anextlevel num_rows=%d, num_cols=%d, num_entries=%d\n", Anextlevel.num_rows, Anextlevel.num_cols, Anextlevel.num_entries);
//}

template <>
void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::createNextLevel()
{
  cudaThreadSetCacheConfig(cudaFuncCachePreferShared);

  permutation_d = IdxVector_d(this->nn, -1);
  ipermutation_d = IdxVector_d(this->nn, -1);
  IdxVector_d partitionlabel_d(this->nn);
  IdxVector_h partitionlabelpermuted(this->nn);
  IdxVector_h partitionlabel(this->nn);
  int* aggregatetmp;
  int* partitiontmp;

  //compute permutation
  if(this->level_id == 0)
  {
    if(mesh_type == 0)
    {
      aggregator->computePermutation_d(this->m_meshPtr, permutation_d, ipermutation_d, aggregateIdx_d, partitionIdx_d, partitionlabel_d, m_xadjout_d, m_adjncyout_d, metis_size, part_max_size);// DHL
    }
    else
    {
      aggregator->computePermutation_d(this->m_tetmeshPtr, permutation_d, ipermutation_d, aggregateIdx_d, partitionIdx_d, partitionlabel_d, m_xadjout_d, m_adjncyout_d, metis_size, part_max_size); // DHL
    }
  }
  else
  {
    aggregator->computePermutation_d(m_xadj_d, m_adjncy_d, permutation_d, ipermutation_d, aggregateIdx_d, partitionIdx_d, partitionlabel_d, m_xadjout_d, m_adjncyout_d, metis_size, part_max_size); // DHL
  }

  std::cout << "size: " << partitionIdx_d.size() - 1 << std::endl;

  this->nnout = aggregateIdx_d.size() - 1;
  IdxVector_d ones(partitionlabel_d.size(), 1);
  IdxVector_d outputkeys(partitionIdx_d.size() - 1);
  IdxVector_d blocksizes(partitionIdx_d.size() - 1);
  thrust::reduce_by_key(partitionlabel_d.begin(), partitionlabel_d.end(), ones.begin(), outputkeys.begin(), blocksizes.begin());
  largestblocksize = thrust::reduce(blocksizes.begin(), blocksizes.end(), -1, thrust::maximum<int>());

  cout << "The largest block size is " << largestblocksize << endl;

  //generate  matrix
  int num_per_thread;
  switch(DS_type)
  {

    case 0:
      generateMatrixSymmetric_d(permutation_d, aggregateIdx_d, partitionIdx_d, partitionlabel_d);
      num_per_thread = ceil((double)largest_num_entries / largestblocksize);
      cout << "The largest num of entries per thread is " << num_per_thread << endl;
      break;

    case 1:
      //      generateMatrix(thrust::raw_pointer_cast(&permutation_h[0]), thrust::raw_pointer_cast(&aggregateIdx[0]), thrust::raw_pointer_cast(&partitionIdx[0]), thrust::raw_pointer_cast(&partitionlabel[0]));
      break;
    case 2:
			generateMatrixCsr(permutation_d, aggregateIdx_d, partitionIdx_d, partitionlabel_d);
      break;

    default:
      cout << "Wrong DStype 2!" << endl;
      exit(0);
  }

  //generate prolongator
  generateProlongatorFull_d(aggregateIdx_d, partitionIdx_d);

  //generate matrix for next level
  generateNextLevelMatrixFull_d();
}

template <>
void SmoothedMG_AMG_Level<Matrix_d, Vector_d>::createNextLevel()
{
}

template <class Matrix, class Vector>
void SmoothedMG_AMG_Level<Matrix, Vector>::computeProlongationOperator()
{

  this->Profile.tic("computeP");
  //  Matrix &A=this->A;
  //
  //  typedef typename Vector::memory_space MemorySpace;
  //  //allocate necessary memory
  //  array1d<float,MemorySpace> weights(A.num_rows,0.0);
  //  array1d<bool,MemorySpace> s_con(A.num_entries,false);


  //  //generate the interpolation matrix
  //  interpolator->generateInterpolationMatrix(A,cf_map,s_con,scratch,P);
  //
  //  weights.resize(0); weights.shrink_to_fit();
  //  s_con.resize(0); s_con.shrink_to_fit();
  //  cf_map.resize(0); cf_map.shrink_to_fit();
  //  scratch.resize(0); scratch.shrink_to_fit();
  //
  //  this->Profile.toc("computeP");
}

#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <cusp/blas.h>

/**********************************************
 * computes R=P^T
 **********************************************/
template <class Matrix, class Vector>
void SmoothedMG_AMG_Level<Matrix, Vector>::computeRestrictionOperator()
{
  //  this->Profile.tic("computeR");
  //  cusp::transpose(P,R);
  //  this->Profile.toc("computeR");
}

/**********************************************
 * computes the Galerkin product: A_c=R*A*P
 **********************************************/
template <class Matrix, class Vector>
void SmoothedMG_AMG_Level<Matrix, Vector>::computeAOperator()
{
  //  this->Profile.tic("computeA");
  //  Matrix RA;
  //  cusp::multiply(R,this->A,RA);
  //  cusp::multiply(RA,P,this->next->getA());
  //  this->Profile.toc("computeA");
}

/**********************************************
 * computes the restriction: rr=R*r 
 **********************************************/
template <class Matrix, class Vector>
void SmoothedMG_AMG_Level<Matrix, Vector>::restrictResidual(const Vector &r, Vector & rr)
{
  //  this->Profile.tic("restrictRes");
  //  cusp::multiply(R,r,rr);
  //  this->Profile.toc("restrictRes");
}

/**********************************************
 * prolongates the error: x+=P*e
 **********************************************/
template <class Matrix, class Vector>
void SmoothedMG_AMG_Level<Matrix, Vector>::prolongateAndApplyCorrection(const Vector &e, Vector &x, Vector & tmp)
{
  //  this->Profile.tic("proCorr");
  //  //prolongate
  //  cusp::multiply(P,e,tmp); 
  //  //apply
  //  cusp::blas::axpby(x,tmp,x,1,1);
  //  this->Profile.toc("proCorr");
}

/****************************************
 * Explict instantiations
 ***************************************/
template class SmoothedMG_AMG_Level<Matrix_h, Vector_h>;
template class SmoothedMG_AMG_Level<Matrix_d, Vector_d>;
