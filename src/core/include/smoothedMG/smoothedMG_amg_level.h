#ifndef __SMOOTHEDMG_AMG_LEVEL_H__
#define __SMOOTHEDMG_AMG_LEVEL_H__

template <class Matrix, class Vector> class SmoothedMG_AMG_Level;

#include <amg_level.h>
#include <smoothedMG/aggregators/aggregator.h>
#include <cusp/multiply.h>
#include <cusp/precond/aggregation/smooth.h>
#include <cusp/transpose.h>

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
   void createNextLevel(bool verbose = false);
   void restrictResidual(const Vector &r, Vector &rr);
   void prolongateAndApplyCorrection(const Vector &c, Vector &x, Vector &tmp);

   protected:

   void generateMatrixCsr(IdxVector_d &permutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel);
   void generateMatrixSymmetric_d(IdxVector_d &permutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel, bool verbose = false);
   void generateProlongatorFull_d(IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx);
   void computeProlongationOperator();
   void computeRestrictionOperator();
   void generateNextLevelMatrixFull_d(bool verbose = false);
   void computeAOperator();

   Matrix P, R;
   Matrix_coo_d P_d, R_d;
   Matrix_coo_h Acoo;
   Matrix_coo_d Acoo_d;

   Matrix_coo_h AinCoo;

   Aggregator<Matrix, Vector>* aggregator;
   IdxVector_h aggregateIdx;
   IdxVector_h partitionIdx;
   IdxVector_h permutation_h;
   IdxVector_h ipermutation_h;
};
#endif
