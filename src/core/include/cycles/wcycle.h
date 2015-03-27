#ifndef __WCYCLE_H__
#define __WCYCLE_H__


template <class Matrix, class Vector>
class W_Cycle {
  public:
  inline W_Cycle(AMG_Level<Matrix,Vector> *next, const Vector& b, Vector &x) {
    if(next->isFinest())
      next->cycle(W_CYCLE,b,x);
    else {
      next->cycle(W_CYCLE,b,x);
      next->cycle(W_CYCLE,b,x);
    }
  }
};

#endif 
