#ifndef __FCYCLE_H__
#define __FCYCLE_H__
template <class Matrix, class Vector>
class F_Cycle {
  public:
  inline F_Cycle(AMG_Level<Matrix,Vector> *next, const Vector& b, Vector &x) {
    if(next->isFinest())
      next->cycle(F_CYCLE,b,x);
    else {
      next->cycle(W_CYCLE,b,x);
      next->cycle(V_CYCLE,b,x);
    }
  }
};
#endif
