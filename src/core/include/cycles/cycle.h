#pragma once
enum CycleType {V_CYCLE,W_CYCLE,F_CYCLE,K_CYCLE};

#include <getvalue.h>
#include <error.h>
#include <amg_level.h>

inline const char* getString(CycleType p) {
  switch(p)
  {
    case V_CYCLE:
      return "V Cycle";
    case W_CYCLE:
      return "W Cycle";
    case F_CYCLE:
      return "F Cycle";
    case K_CYCLE:
      return "K Cycle";
    default:
      return "UNKNOWN";
  }
}

template <>
inline CycleType getValue<CycleType>(const char* name) {
  if(strncmp(name,"V",100)==0)
    return V_CYCLE;
  else if(strncmp(name,"W",100)==0)
    return W_CYCLE;
  else if(strncmp(name,"F",100)==0)
    return F_CYCLE;
  else if(strncmp(name,"K",100)==0)
    return K_CYCLE;

  char error[100];
  sprintf(error,"Cycle '%s' is not defined",name);
  FatalError(error);
}

template <class Matrix, class Vector> void dispatch_cycle(int num_iters, CycleType cycle, AMG_Level<Matrix,Vector> *level, const Vector& b, Vector &x);

#include <amg.h>

#include <cycles/vcycle.h>
#include <cycles/wcycle.h>
#include <cycles/fcycle.h>
#include <cycles/cgcycle.h>
/*******************************************************
 * Dispatches the cycle that is passed in
 *******************************************************/
template <class Matrix, class Vector>
void dispatch_cycle(int num_iters, CycleType cycle, AMG_Level<Matrix,Vector> *level, const Vector& b, Vector &x) {
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

  
