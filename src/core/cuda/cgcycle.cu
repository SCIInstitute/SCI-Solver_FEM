#include <allocator.h>
#include <cycles/cgcycle.h>
#include <cusp/multiply.h>
#include <cusp/blas.h>

//template <class Matrix, class Vector>
//CG_Cycle<Matrix,Vector>::CG_Cycle(CycleType next_cycle, int num_iters, AMG_Level<Matrix,Vector> *next, const Vector& b, Vector &x) {
//
//  typedef typename Matrix::value_type ValueType;
//
//  int N=b.size();
//
//
//  //create temperary vectors
//  Vector* yp=Allocator<Vector>::allocate(N);
//  Vector* zp=Allocator<Vector>::allocate(N);
//  Vector* rp=Allocator<Vector>::allocate(N);
//  Vector* pp=Allocator<Vector>::allocate(N);
//
//  Vector &y=*yp;
//  Vector &z=*zp;
//  Vector &r=*rp;
//  Vector &p=*pp;
//
//  //TODO account for X being 0's
//  //not doing this optimization at the moment
//  if(next->isInitCycle()) {
//    cusp::blas::fill(x,0);
//    next->unsetInitCycle();
//  }
//
//  // y = Ax
//  cusp::multiply(next->getA(), x, y);
//
//  // r = b - A*x
//  cusp::blas::axpby(b, y, r, ValueType(1), ValueType(-1));
//
//  // z = M*r
//  next->setInitCycle();
//  next->cycle(next_cycle,r,z);
//
//  // p = z
//  cusp::blas::copy(z, p);
//
//  // rz = <r^H, z> 
//  ValueType rz=cusp::blas::dotc(r,z);
//  int k=0;
//  while(true)  
//  {
//    // y = Ap
//    cusp::multiply(next->getA(), p, y);
//
//    // alpha = <r,z>/<y,p>
//    ValueType alpha =  rz / cusp::blas::dotc(y, p);
//
//    // x = x + alpha * p
//    cusp::blas::axpy(p, x, alpha);
//
//    if(++k==num_iters)
//      break;
//
//    // r = r - alpha * y           
//    cusp::blas::axpy(y, r, -alpha);
//
//    //TODO:  if norm(r)<tolerance break
//
//    // z = M*r
//    next->setInitCycle();
//    next->cycle(next_cycle,r,z);
//
//    ValueType rz_old = rz;
//
//    // rz = <r^H, z>
//    rz = cusp::blas::dotc(r, z);
//
//    // beta <- <r_{i+1},z_{i+1}>/<r,z> 
//    ValueType beta = rz / rz_old;
//
//    // p += z + beta*p
//    cusp::blas::axpby(z, p, p, ValueType(1), beta);
//  }
//  Allocator<Vector>::free(yp,N);
//  Allocator<Vector>::free(zp,N);
//  Allocator<Vector>::free(rp,N);
//  Allocator<Vector>::free(pp,N);
//}

template <class Matrix, class Vector>
CG_Flex_Cycle<Matrix, Vector>::CG_Flex_Cycle(CycleType next_cycle, int num_iters, AMG_Level<Matrix_h, Vector_h> *next, const Matrix_hyb_d_CG &Aell, const Vector_d_CG &b, Vector_d_CG &x, CGType tol, int maxiters)
{

  typedef typename Matrix::value_type ValueType;
  typedef typename Matrix::index_type IndexType;
  typedef typename Matrix::memory_space MemorySpace;


  int N = b.size();
	ValueType bnorm = cusp::blas::nrm2(b);
  Vector_d_CG y(N);
  
  Vector_d_CG z(N);
  Vector_d_CG r(N);
  Vector_d_CG d(N);
  Vector_d_CG p(N);

  cusp::multiply(Aell, x, y);
  cusp::blas::axpby(b, y, r, ValueType(1), ValueType(-1));
  next->cycle_level0(next_cycle, r, z);
  cusp::blas::copy(z, p);

  ValueType rzold = cusp::blas::dotc(r, z);
  ValueType rznew;

  int niter = 0;
	double iter_start, iter_stop;
	iter_start = CLOCK();
        double last_iter_clock = iter_start;
  while(niter < maxiters)
  {

    cusp::multiply(Aell, p, y);
    ValueType yp = cusp::blas::dotc(y, p);
    ValueType alpha = rzold / yp;
    cusp::blas::axpy(p, x, alpha);
    cusp::blas::axpy(y, r, -alpha);
		ValueType normr = cusp::blas::nrm2(r);
    cout << "normr=" << scientific << normr << "  niter=" << niter << endl;
    
    double temp_time = CLOCK();
    last_iter_clock = temp_time;
    
    if( (normr / bnorm) <= tol)
      break;
    
    niter++;
    next->cycle_level0(next_cycle, r, z);
    rznew = cusp::blas::dotc(z, r);
    ValueType beta = rznew / rzold;
    cusp::blas::axpby(z, p, p, ValueType(1), beta);
    rzold = rznew;
  }
	cudaThreadSynchronize();
	iter_stop = CLOCK();
	cout << "average time per iteration:        " << (iter_stop-iter_start) / niter << endl;
	cout << "total solve time:        " << (iter_stop-iter_start) << endl;

  y.clear();
  z.clear();
  r.clear();
  d.clear();
  p.clear();
}

/****************************************
 * Explict instantiations
 ***************************************/
//template class CG_Cycle<Matrix_h,Vector_h>;
//template class CG_Cycle<Matrix_d,Vector_d>;
template class CG_Flex_Cycle<Matrix_h_CG, Vector_h_CG>;
//template class CG_Flex_Cycle<Matrix_d_CG, Vector_d_CG>;
