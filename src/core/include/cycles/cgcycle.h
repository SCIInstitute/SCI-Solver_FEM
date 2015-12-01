#ifndef __CGCYCLE_H__
#define __CGCYCLE_H__

template <class Matrix, class Vector> class CG_Cycle;
template <class Matrix, class Vector> class CG_Flex_Cycle;

#include <cycles/cycle.h>
#include <amg_level.h>
#include <types.h>

template <class Matrix, class Vector>
  class CG_Flex_Cycle {
    public:
      typedef typename Matrix::value_type ValueType; 
      CG_Flex_Cycle(CycleType next_cycle, int num_iters, 
        AMG_Level<Matrix_h,Vector_h> *next, const Matrix_hyb_d_CG &Aell,
        const Vector_d_CG &b, Vector_d_CG &x, CGType tol, int maxiters, bool verbose = false);
 };
#endif 
