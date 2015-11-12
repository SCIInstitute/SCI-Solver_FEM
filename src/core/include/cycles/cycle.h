#ifndef __CYCLE_H__
#define __CYCLE_H__

enum CycleType {V_CYCLE,W_CYCLE,F_CYCLE,K_CYCLE};

#include <error.h>
#include <amg_level.h>

template <class Matrix, class Vector> void dispatch_cycle(int num_iters, 
  CycleType cycle, AMG_Level<Matrix,Vector> *level, const Vector& b, Vector &x);

#include <amg.h>

#include <cycles/vcycle.h>
#include <cycles/wcycle.h>
#include <cycles/fcycle.h>
#include <cycles/cgcycle.h>
/*******************************************************
 * Dispatches the cycle that is passed in
 *******************************************************/
template <class Matrix, class Vector>
void dispatch_cycle(int num_iters, CycleType cycle, AMG_Level<Matrix,Vector> 
  *level, const Vector& b, Vector &x) {
  switch(cycle) {
    case V_CYCLE:
      V_Cycle<Matrix,Vector>(level,b,x);
      break;
    case W_CYCLE:
      W_Cycle<Matrix,Vector>(level,b,x);
      break;
    case F_CYCLE:
      F_Cycle<Matrix,Vector>(level,b,x);
      break;
    case K_CYCLE:
      CG_Flex_Cycle<Matrix,Vector>(K_CYCLE,num_iters,level,b,x);
      break;
    default:
      FatalError("dispatch_cycle not defined for cycle type");
  }
}

#endif 
