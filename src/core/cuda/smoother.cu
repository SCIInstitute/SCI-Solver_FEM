#include <smoothers/smoother.h>
#include <cusp/blas.h>
#include <types.h>

/***************************************
 * Source Definitions
 ***************************************/

template<class Matrix, class Vector>
Smoother<Matrix,Vector>::~Smoother() {
};

template<class Matrix, class Vector>
void Smoother<Matrix,Vector>::smooth_with_0_initial_guess(const Matrix &A, const Vector &b, Vector &x) {
  //by default set x to zero and call smooth.  smoothers can optimize this path if they wish
  cusp::blas::fill(x,0);
  smooth(A,b,x);  
};

#include <smoothers/gauss_seidel.h>
/*********************************************
 * Allocates smoothers based on passed in type
 *********************************************/
template <class Matrix, class Vector>
Smoother<Matrix, Vector>* Smoother<Matrix, Vector>::allocate(double smootherWeight,
  int preInnerIters, int postInnerIters, int postRelaxes, const Matrix_d& A)
{
  return new gauss_seidel<Matrix, Vector>( smootherWeight,
    preInnerIters, postInnerIters, postRelaxes, A);
}

/****************************************
 * Explict instantiations
 ***************************************/
//template class Smoother<Matrix_h,Vector_h>;
template class Smoother<Matrix_d,Vector_d>;

