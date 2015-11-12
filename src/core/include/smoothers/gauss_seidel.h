#ifndef __GAUSSSEIDEL_H__
#define __GAUSSSEIDEL_H__

#include <string>
#include <smoothers/smoother.h>
#include <cusp/multiply.h>
#include <my_timer.h>
#include <FEMSolver.h>

/*****************************************************
 * Jacobi smoother
 ****************************************************/
template<class Matrix, class Vector>
class gauss_seidel : public Smoother < Matrix, Vector >
{
public:
  typedef typename Matrix::value_type ValueType;
  typedef typename Matrix::index_type IndexType;
  typedef typename Matrix::memory_space MemorySpace;

  gauss_seidel(FEMSolver * cfg, const Matrix_d& Ainit);
  void find_diag(const Matrix_ell_d& A);
  void smooth(const Matrix& A, const Vector& b, Vector& x);
  void smooth_with_0_initial(const Matrix& A, const Vector &b, Vector &x);

  void preRRRFull(const cusp::ell_matrix<IndexType, ValueType, MemorySpace>& AinEll,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutCoo,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::hyb_matrix<IndexType, ValueType, MemorySpace>& restrictor,
    const cusp::array1d<IndexType, MemorySpace>& permutation,
    cusp::array1d<ValueType, MemorySpace>& b,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& bc,
    int level_id,
    int largestblksz);

  void preRRRFullCsr(const cusp::csr_matrix<IndexType, ValueType, MemorySpace>& AinCsr,
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
    int largestnumperrow);

  void preRRRFullSymmetric(const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AinSysCoo,
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
    int largestblksz,
    int largestnumentries,
    bool verbose = false);
  void preRRRFullSymmetricSync(const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AinSysCoo,
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
    int largestblksz,
    int largestnumentries);

  void postPCR(const cusp::ell_matrix<IndexType, ValueType, MemorySpace>& AinEll,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutCoo,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::array1d<ValueType, MemorySpace>& P,
    const cusp::array1d<ValueType, MemorySpace>& b,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& xc);

  void postPCRFull(const cusp::ell_matrix<IndexType, ValueType, MemorySpace>& AinEll,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutCoo,
    const cusp::array1d<IndexType, MemorySpace>& AoutBlockIdx,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::hyb_matrix<IndexType, ValueType, MemorySpace>& prolongator,
    const cusp::array1d<IndexType, MemorySpace>& permutation,
    const cusp::array1d<ValueType, MemorySpace>& b,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& xc,
    int level_id,
    int largestblksz);

  void postPCRFullCsr(const cusp::csr_matrix<IndexType, ValueType, MemorySpace>& AinCsr,
    const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutCoo,
    const cusp::array1d<IndexType, MemorySpace>& AoutBlockIdx,
    const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
    const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
    const cusp::hyb_matrix<IndexType, ValueType, MemorySpace>& prolongator,
    const cusp::array1d<IndexType, MemorySpace>& permutation,
    const cusp::array1d<ValueType, MemorySpace>& b,
    cusp::array1d<ValueType, MemorySpace>& x,
    cusp::array1d<ValueType, MemorySpace>& xc,
    int level_id,
    int largestblksz,
    int largestnumentries,
    int largestnumperrow);

  void postPCRFullSymmetric(const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AinSysCoo,
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
    int largestblksz,
    int largestnumentries);

  void postPCRFullSymmetricSync(const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AinSysCoo,
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
    const cusp::array1d<IndexType, MemorySpace>& segSyncIdx,
    const cusp::array1d<IndexType, MemorySpace>& partSyncIdx,
    int level_id,
    int largestblksz,
    int largestnumentries);

public:
  double weight;
  int nPreInnerIter;
  int nPostInnerIter;
  int post_relaxes;

};
#endif
