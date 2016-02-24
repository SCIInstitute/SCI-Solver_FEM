
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
SmoothedMG_AMG_Level<Matrix, Vector>::SmoothedMG_AMG_Level(AMG<Matrix, Vector> *amg) 
  : AMG_Level<Matrix, Vector>(amg)
{
   aggregator = Aggregator<Matrix, Vector>::allocate(amg->aggregatorType_); // DHL
}

template <class Matrix, class Vector>
SmoothedMG_AMG_Level<Matrix, Vector>::~SmoothedMG_AMG_Level()
{
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
void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateMatrixCsr(IdxVector_d &permutation,
  IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel)
{

   int numpart = partitionIdx.size() - 1;
   int numentries = A_d.num_entries;
   Acoo_d = A_d;
   cusp::array1d<int, cusp::device_memory> entrypartlabel(numentries, -1);

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
   int innum = thrust::reduce(redoutputvalue.begin(), redoutputvalue.begin() + numpart);
   int outnum = numentries - innum;
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
void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateMatrixSymmetric_d(IdxVector_d &permutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel, bool verbose)
{
   int numpart = partitionIdx.size() - 1;
   int numentries = A_d.num_entries;
   Acoo_d = A_d;
   cusp::array1d<int, cusp::device_memory> entrypartlabel(numentries, -1);

   size_t blocksize = 256;
   size_t blocknum = ceil((float)numentries / (float)blocksize);
   if(blocknum > 65535) printf("too many blocks!!\n");

   matrixpermute_kernel << <blocknum, blocksize >> >(numpart, A_d.num_entries,
         thrust::raw_pointer_cast(&Acoo_d.row_indices[0]),
         thrust::raw_pointer_cast(&Acoo_d.column_indices[0]),
         thrust::raw_pointer_cast(&Acoo_d.values[0]),
         thrust::raw_pointer_cast(&entrypartlabel[0]),
         thrust::raw_pointer_cast(&permutation[0]),
         thrust::raw_pointer_cast(&partitionlabel[0]));

   typedef IdxVector_d::iterator IntIterator;
   typedef Vector_d::iterator FloatIterator;
   typedef thrust::tuple<IntIterator, IntIterator, FloatIterator> IteratorTuple;

   typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
   ZipIterator iter(thrust::make_tuple(Acoo_d.row_indices.begin(), Acoo_d.column_indices.begin(), Acoo_d.values.begin()));

   thrust::sort_by_key(entrypartlabel.begin(), entrypartlabel.end(), iter);
   if (verbose)
      printf("partition number is %d\n", numpart);

   IdxVector_d redoutput(numentries);
   IdxVector_d redvalue(numentries, 1);
   IdxVector_d redoutputkey(2 * numpart + 1);
   IdxVector_d redoutputvalue(2 * numpart + 1);

   thrust::reduce_by_key(entrypartlabel.begin(), entrypartlabel.end(), redvalue.begin(), redoutputkey.begin(), redoutputvalue.begin());
   int innum = thrust::reduce(redoutputvalue.begin(), redoutputvalue.begin() + numpart);
   int outnum = numentries - innum - redoutputvalue[numpart * 2];
   IntIterator res = thrust::max_element(redoutputvalue.begin(), redoutputvalue.begin() + numpart);
   largest_num_entries = *res;

   if (verbose) {
      printf("largest_num_entries is %d\n", largest_num_entries);
      printf("inside number is %d, outside number is %d\n", innum, outnum);
   }
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
   Acoo_d.sort_by_row_and_column();
   A_d = Acoo_d;
}

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
   const AMGType lambda = this->amg->proOmega_;

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

   //  printf("NNZ is: %d\n", NNZ);

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

   restrictorFull_d = R_d;

}

template <>
void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::generateNextLevelMatrixFull_d(bool verbose)
{
   Matrix_coo_d AP;
   Matrix_coo_d Atmp = A_d;
   cusp::multiply(Atmp, P_d, AP);
   Matrix_d& Anextlevel2 = this->next->getA_d();
   Matrix_coo_d tmpmtx;
   cusp::multiply(R_d, AP, tmpmtx);
   Anextlevel2 = tmpmtx;
   if (verbose)  printf("Anextlevel num_rows=%lu, num_cols=%lud, num_entries=%lud\n", Anextlevel2.num_rows, Anextlevel2.num_cols, Anextlevel2.num_entries);
}

template <>
void SmoothedMG_AMG_Level<Matrix_h, Vector_h>::createNextLevel(bool verbose)
{
   cudaThreadSetCacheConfig(cudaFuncCachePreferShared);

   permutation_d = IdxVector_d(this->nn, -1);
   ipermutation_d = IdxVector_d(this->nn, -1);
   IdxVector_d partitionlabel_d(this->nn);
   IdxVector_h partitionlabelpermuted(this->nn);
   IdxVector_h partitionlabel(this->nn);

   if (verbose)  std::cout << "Initialized IdxVector host & device vars." << std::endl;

   //compute permutation
   if(this->level_id == 0)
   {
      if(this->amg->triMesh_ != NULL)
      {
         if (verbose)
           std::cout << "calling computePermutation_d with tri mesh." << std::endl;
         aggregator->computePermutation_d(this->m_meshPtr, permutation_d, ipermutation_d,
           aggregateIdx_d, partitionIdx_d, partitionlabel_d, m_xadjout_d, m_adjncyout_d,
           this->amg->metisSize_, this->amg->partitionMaxSize_, verbose);// DHL
         if (verbose)
           std::cout << "computePermutation_d called with tri mesh." << std::endl;
      }
      else
      {
         if (verbose)
           std::cout << "calling computePermutation_d with tet mesh." << std::endl;
         aggregator->computePermutation_d(this->m_tetmeshPtr, permutation_d,
           ipermutation_d, aggregateIdx_d, partitionIdx_d, partitionlabel_d,
           m_xadjout_d, m_adjncyout_d, this->amg->metisSize_,
           this->amg->partitionMaxSize_, verbose); // DHL
         if (verbose)
           std::cout << "computePermutation_d called with tet mesh." << std::endl;
      }
   }
   else
   {
         if (verbose)
           std::cout << "calling computePermutation_d with level_id != 0." << std::endl;
      aggregator->computePermutation_d(m_xadj_d, m_adjncy_d, permutation_d,
        ipermutation_d, aggregateIdx_d, partitionIdx_d, partitionlabel_d,
        m_xadjout_d, m_adjncyout_d, this->amg->metisSize_, 
        this->amg->partitionMaxSize_, verbose); // DHL
         if (verbose)
           std::cout << "computePermutation_d called with level_id != 0." << std::endl;
   }

   if (verbose)
      std::cout << "size: " << partitionIdx_d.size() - 1 << std::endl;

   this->nnout = aggregateIdx_d.size() - 1;
   IdxVector_d ones(partitionlabel_d.size(), 1);
   IdxVector_d outputkeys(partitionIdx_d.size() - 1);
   IdxVector_d blocksizes(partitionIdx_d.size() - 1);
   thrust::reduce_by_key(partitionlabel_d.begin(), partitionlabel_d.end(), ones.begin(), outputkeys.begin(), blocksizes.begin());
   largestblocksize = thrust::reduce(blocksizes.begin(), blocksizes.end(), -1, thrust::maximum<int>());

   if (verbose)
      std::cout << "The largest block size is " << largestblocksize << std::endl;

   //generate  matrix
   int num_per_thread;
   switch (amg->dsType_)
   {

   case 0:
      generateMatrixSymmetric_d(permutation_d, aggregateIdx_d, partitionIdx_d, partitionlabel_d, verbose);
      num_per_thread = ceil((double)largest_num_entries / largestblocksize);
      if (verbose)
         std::cout << "The largest num of entries per thread is " << num_per_thread << std::endl;
      break;

   case 1:
      break;
   case 2:
      generateMatrixCsr(permutation_d, aggregateIdx_d, partitionIdx_d, partitionlabel_d);
      break;

   default:
      std::cout << "Wrong DStype 2!" << std::endl;
      exit(0);
   }

   //generate prolongator
   generateProlongatorFull_d(aggregateIdx_d, partitionIdx_d);

   //generate matrix for next level
   generateNextLevelMatrixFull_d(verbose);
}

template <>
void SmoothedMG_AMG_Level<Matrix_d, Vector_d>::createNextLevel(bool verbose)
{
}

template <class Matrix, class Vector>
void SmoothedMG_AMG_Level<Matrix, Vector>::computeProlongationOperator()
{
   this->Profile.tic("computeP");
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
}

/**********************************************
 * computes the Galerkin product: A_c=R*A*P
 **********************************************/
template <class Matrix, class Vector>
void SmoothedMG_AMG_Level<Matrix, Vector>::computeAOperator()
{
}

/**********************************************
 * computes the restriction: rr=R*r
 **********************************************/
template <class Matrix, class Vector>
void SmoothedMG_AMG_Level<Matrix, Vector>::restrictResidual(const Vector &r, Vector & rr)
{
}

/**********************************************
 * prolongates the error: x+=P*e
 **********************************************/
template <class Matrix, class Vector>
void SmoothedMG_AMG_Level<Matrix, Vector>::prolongateAndApplyCorrection(const Vector &e, Vector &x, Vector & tmp)
{
}

/****************************************
 * Explict instantiations
 ***************************************/
template class SmoothedMG_AMG_Level<Matrix_h, Vector_h>;
template class SmoothedMG_AMG_Level<Matrix_d, Vector_d>;
