#include <allocator.h>
#include <cycles/cgcycle.h>
#include <cusp/multiply.h>
#include <cusp/blas.h>

template <class Matrix, class Vector>
CG_Flex_Cycle<Matrix, Vector>::CG_Flex_Cycle(CycleType next_cycle, int num_iters, AMG_Level<Matrix_h, Vector_h> *next, const Matrix_hyb_d_CG &Aell, const Vector_d_CG &b, Vector_d_CG &x, CGType tol, int maxiters, bool verbose)
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
   while(niter < maxiters)
   {

      cusp::multiply(Aell, p, y);
      ValueType yp = cusp::blas::dotc(y, p);
      ValueType alpha = rzold / yp;
      cusp::blas::axpy(p, x, alpha);
      cusp::blas::axpy(y, r, -alpha);
      ValueType normr = cusp::blas::nrm2(r);
      if (verbose)
         cout << "normr=" << scientific << normr << "  niter=" << niter << endl;

      if( (normr / bnorm) <= tol)
         break;

      niter++;
      next->cycle_level0(next_cycle, r, z, verbose);
      rznew = cusp::blas::dotc(z, r);
      ValueType beta = rznew / rzold;
      cusp::blas::axpby(z, p, p, ValueType(1), beta);
      rzold = rznew;
   }
   cudaThreadSynchronize();
   iter_stop = CLOCK();
   if (verbose) {
      cout << "average time per iteration:        " << (iter_stop-iter_start) / niter << endl;
      cout << "total solve time:        " << (iter_stop-iter_start) << endl;
   }

   y.clear();
   z.clear();
   r.clear();
   d.clear();
   p.clear();
}

/****************************************
 * Explict instantiations
 ***************************************/
template class CG_Flex_Cycle<Matrix_h_CG, Vector_h_CG>;
