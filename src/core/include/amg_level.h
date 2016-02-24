#ifndef __AMG_LEVEL_H__
#define __AMG_LEVEL_H__
template <class Matrix, class Vector> class AMG_Level;

#include <amg.h>
#include <smoothers/smoother.h>
#include <cycles/cycle.h>
#include <my_timer.h>
#include <vector>
#include <types.h>
#include "TriMesh.h"
#include "tetmesh.h"
#include <cusp/print.h>

/********************************************************
 * AMG Level class:
 *  This class is a base class for AMG levels.  This
 *  class is a linked list of levels where each
 *  level contains the solution state for that level.
 ********************************************************/
template <class Matrix, class Vector>
    class AMG_Level
{
   friend class AMG<Matrix, Vector>;
   public:

   AMG_Level(AMG<Matrix, Vector> *amg) : smoother(0), amg(amg), next(0), init(false)
   {};
   virtual ~AMG_Level();

   virtual void restrictResidual(const Vector &r, Vector &rr) = 0;
   virtual void prolongateAndApplyCorrection(const Vector &c, Vector &x, Vector &tmp) = 0;
   virtual void createNextLevel(bool verbose = false) = 0;

   void setup();
   void cycle(CycleType cycle, Vector_d &b, Vector_d &x, bool verbose = false);
   void cycle_level0(CycleType cycle, Vector_d_CG &b, Vector_d_CG &x, bool verbose = false);

   void setInitCycle()
   {
      init = true;
   }

   void unsetInitCycle()
   {
      init = false;
   }

   int getLevel()
   {
      return level_id;
   }

   bool isInitCycle()
   {
      return init;
   }

   inline Matrix_d& getA_d()
   {
      return A_d;
   }

   inline bool isFinest()
   {
      return level_id == 0;
   }

   inline bool isCoarsest()
   {
      return next == NULL;
   }

   static AMG_Level<Matrix, Vector>* allocate(AMG<Matrix, Vector>*amg);

   protected:
   typedef typename Matrix::index_type IndexType;
   typedef typename Matrix::value_type ValueType;
   typedef typename Matrix::memory_space MemorySpace;
   levelProfile Profile;
   std::vector<int> originalRow;
   std::vector<int> getOriginalRows();

   protected:
   TriMesh* m_meshPtr;
   TetMesh* m_tetmeshPtr;
   int nn;
   IdxVector_h m_xadj;
   IdxVector_h m_adjncy;

   IdxVector_d m_xadj_d;
   IdxVector_d m_adjncy_d;

   int nnout;
   //  int* m_xadjout;
   //  int* m_adjncyout;
   IdxVector_h m_xadjout;
   IdxVector_h m_adjncyout;
   IdxVector_d m_xadjout_d;
   IdxVector_d m_adjncyout_d;

   int largestblock;
   int largestblocksize;
   //  Matrix A;
   Vector prolongator; //incomplete prolongator
   Matrix_coo_h prolongatorFull;
   Matrix_ell_h AinEll;
   Matrix_h     AinCsr;
   //  Matrix_coo_h AinSysCoo;
   Matrix_coo_h Aout;
   //  Matrix_coo_h AoutSys;
   IdxVector_h partSyncIdx_h;
   IdxVector_h segSyncIdx_h;


   Vector_d prolongator_d; //incomplete prolongator
   Matrix_hyb_d prolongatorFull_d;
   Matrix_hyb_d restrictorFull_d;
   Matrix_d A_d;
   Matrix_ell_d AinEll_d;
   Matrix_d AinCSR_d;
   Matrix_coo_d Aout_d;
   IdxVector_d AinBlockIdx_d;
   IdxVector_d AoutBlockIdx_d;
   Matrix_coo_d AinSysCoo_d;
   Matrix_coo_d AoutSys_d;
   Vector_d bc_d, xc_d, r_d;
   IdxVector_d aggregateIdx_d;
   IdxVector_d partitionIdx_d;
   IdxVector_d permutation_d;
   IdxVector_d ipermutation_d;
   IdxVector_d partSyncIdx_d;
   IdxVector_d segSyncIdx_d;
   Smoother<Matrix_d, Vector_d>* smoother;


   AMG<Matrix, Vector>* amg;
   AMG_Level* next;
   int largest_num_entries;
   int largest_num_per_row;
   int largest_num_segment;
   int level_id;
   bool init; //marks if the x vector needs to be initialized
};
#endif
