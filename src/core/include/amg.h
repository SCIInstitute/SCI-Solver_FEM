#ifndef __AMG_H__
#define __AMG_H__
template <class Matrix, class Vector> class AMG;

enum SolverType {AMG_SOLVER,PCG_SOLVER};

enum ConvergenceType { ABSOLUTE_CONVERGENCE, RELATIVE_CONVERGENCE };

#include <cusp/detail/lu.h>
#include <error.h>

#include <FEMSolver.h>
#include <cycles/cycle.h>
#include <smoothedMG/smoothedMG_amg_level.h>
#include "TriMesh.h"
#include "tetmesh.h"

/*********************************************************
 * AMG Class
 *  This class provides the user interface to the AMG 
 *  solver library. 
 ********************************************************/
template <class Matrix, class Vector>
class AMG 
{
  typedef typename Matrix::value_type ValueType;
  friend class AMG_Level<Matrix,Vector>;
  friend class SmoothedMG_AMG_Level<Matrix,Vector>;

  public:
  AMG(FEMSolver * cfg);
  ~AMG();

  void solve(const Vector_d_CG &b, Vector_d_CG &x);
  void solve_iteration(const Vector_d_CG &b, Vector_d_CG &x);

  void setup(const Matrix_d &Acsr_d);

  void printGridStatistics();

  // profiling & debug output
  void printProfile();
  void printCoarsePoints();
  void printConnections();
  //config pointer
  FEMSolver * cfg;
  private:
  bool converged(const Vector &r, ValueType &nrm);

	cusp::detail::lu_solver<ValueType, cusp::host_memory> LU;

  AMG_Level<Matrix,Vector>* fine;
  ValueType initial_nrm;
  int iterations;
  int num_levels;
	int coarsestlevel;

	Matrix_hyb_d_CG Ahyb_d_CG;

  double solve_start, solve_stop;
  double setup_start, setup_stop;
};
#endif
