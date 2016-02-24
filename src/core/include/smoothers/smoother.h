#ifndef __SMOOTHER_H__
#define __SMOOTHER_H__
template <class Matrix, class Vector> class Smoother;

enum SmootherType
{
  JACOBI, JACOBI_NO_CUSP, GAUSSSEIDEL, POLYNOMIAL, GSINNER
};

#include <error.h>
#include <types.h>

class FEMSolver;

/*************************************
 * Smoother base class
 *************************************/
template<class Matrix, class Vector>
class Smoother
{
  typedef typename Matrix::value_type ValueType;
  typedef typename Matrix::index_type IndexType;
  typedef typename Matrix::memory_space MemorySpace;
public:
  virtual void preRRRFull(const cusp::ell_matrix<IndexType, ValueType, MemorySpace>& AinEll,
                          const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutCoo,
                          const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
                          const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
                          const cusp::hyb_matrix<IndexType, ValueType, MemorySpace>& restrictor,
                          const cusp::array1d<IndexType, MemorySpace>& permutation,
                          cusp::array1d<ValueType, MemorySpace>& b,
                          cusp::array1d<ValueType, MemorySpace>& x,
                          cusp::array1d<ValueType, MemorySpace>& bc,
                          int level_id,
                          int largestblksz) = 0;
  virtual void preRRRFullCsr(const cusp::csr_matrix<IndexType, ValueType, MemorySpace>& AinCsr,
                             const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutCoo,
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
                             int largestnumperrow) = 0;

  virtual void preRRRFullSymmetric(const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AinSysCoo,
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
                                   bool verbose = false) = 0;
  virtual void preRRRFullSymmetricSync(const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AinSysCoo,
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
                                       int largestnumentries) = 0;

  virtual void postPCR(const cusp::ell_matrix<IndexType, ValueType, MemorySpace>& AinEll,
                       const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AoutCoo,
                       const cusp::array1d<IndexType, MemorySpace>& aggregateIdx,
                       const cusp::array1d<IndexType, MemorySpace>& partitionIdx,
                       const cusp::array1d<ValueType, MemorySpace>& P,
                       const cusp::array1d<ValueType, MemorySpace>& b,
                       cusp::array1d<ValueType, MemorySpace>& x,
                       cusp::array1d<ValueType, MemorySpace>& xc) = 0;

  virtual void postPCRFull(const cusp::ell_matrix<IndexType, ValueType, MemorySpace>& AinEll,
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
                           int largestblksz) = 0;
  virtual void postPCRFullCsr(const cusp::csr_matrix<IndexType, ValueType, MemorySpace>& AinCsr,
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
															int largestnumperrow) = 0;

  virtual void postPCRFullSymmetric(const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AinSysCoo,
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
                                    int largestnumentries) = 0;
  virtual void postPCRFullSymmetricSync(const cusp::coo_matrix<IndexType, ValueType, MemorySpace>& AinSysCoo,
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
                                        int largestnumentries) = 0;


  virtual void smooth(const Matrix &A, const Vector &b, Vector &x) = 0;
  virtual void smooth_with_0_initial_guess(const Matrix &A, const Vector &b, Vector &x); //default initializes the vector to 0 and calls smooth
  virtual ~Smoother();
  static Smoother<Matrix, Vector>* allocate(double smootherWeight,
    int preInnerIters, int postInnerIters, int postRelaxes, const Matrix_d& A);
	Vector diag;
};
#endif
