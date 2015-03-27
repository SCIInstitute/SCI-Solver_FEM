#ifndef __SMOOTHEDMG_AMG_LEVEL_H__
#define __SMOOTHEDMG_AMG_LEVEL_H__

template <class Matrix, class Vector> class SmoothedMG_AMG_Level;

#include <amg_level.h>
#include <smoothedMG/aggregators/aggregator.h>
#include <cusp/multiply.h>
#include <cusp/precond/aggregation/smooth.h>
#include <cusp/transpose.h>

using namespace std;

/***************************************************
 * Classical AMG Base Class
 *  Defines the AMG solve algorithm, decendents must
 *  define markCoarseFinePoints() and 
 *  generateInterpoloationMatrix()
 **************************************************/
template <class Matrix, class Vector>
class SmoothedMG_AMG_Level : public AMG_Level<Matrix, Vector>
{
  friend class AMG<Matrix, Vector>;
  typedef typename Matrix::value_type ValueType;
  typedef typename Matrix::index_type IndexType;
  typedef typename Matrix::memory_space MemorySpace;
public:
  SmoothedMG_AMG_Level(AMG<Matrix, Vector> *amg);
  ~SmoothedMG_AMG_Level();

  //  void setup();
  void createNextLevel();
  void restrictResidual(const Vector &r, Vector &rr);
  void prolongateAndApplyCorrection(const Vector &c, Vector &x, Vector &tmp);

protected:


  //  void generateMatrix(int* permutation, int* aggregateIdx, int* partitionIdx, int* partitionlabel);
  //	void generateMatrixCsr(int* permutation, int* aggregateIdx, int* partitionIdx, int* partitionlabel);
  void generateMatrixCsr(IdxVector_d &permutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel);
  //  void generateMatrixSymmetric(int* permutation, int* aggregateIdx, int* partitionIdx, int* partitionlabel);
  void generateMatrixSymmetric_d(IdxVector_d &permutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel);
  //	void generateMatrixSymmetricSync(int* permutation, int* aggregateIdx, int* partitionIdx, int* partitionlabel);
  //  void generateProlongator(int* aggregateIdx, int* partitionIdx);
  //  void generateProlongatorFull(int* aggregateIdx, int* partitionIdx);
  void generateProlongatorFull_d(IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx);
  void computeProlongationOperator();
  void computeRestrictionOperator();
  //  void generateSmoothingMatrix(Matrix& S, Vector D, ValueType omega);
  //  void generateNextLevelMatrix(Matrix& Athislevel, Vector& prolongator);
  //  void generateNextLevelMatrixFull(Matrix& Athislevel, Vector& prolongator);
  void generateNextLevelMatrixFull_d();
  void computeAOperator();


  Matrix P, R;
  Matrix_coo_d P_d, R_d;
  Matrix_coo_h Acoo;
  Matrix_coo_d Acoo_d;

  Matrix_coo_h AinCoo;



  Aggregator<Matrix, Vector>* aggregator;
    //vector<IndexType> aggregateIdx;
    //vector<IndexType> partitionIdx;
    IdxVector_h aggregateIdx;
    IdxVector_h partitionIdx;
    IdxVector_h permutation_h;
    IdxVector_h ipermutation_h;

  //  IdxVector_d d_aggregateIdx;
  //  IdxVector_d d_partitionIdx;
  //  IdxVector_d d_permutation;
  //  IdxVector_d d_ipermutation;


  double prosmoothomega;
  int DS_type;
  int metis_size;
  int mesh_type;
  int part_max_size;
  //IndexType* aggregateIdx;
  //IndexType* partitionIdx;
};
#endif
