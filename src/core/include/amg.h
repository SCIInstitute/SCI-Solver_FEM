#ifndef __AMG_H__
#define __AMG_H__
template <class Matrix, class Vector> class AMG;

enum SolverType {AMG_SOLVER,PCG_SOLVER};

#include <getvalue.h>

#include <cusp/detail/lu.h>
#include <error.h>

#include <amg_config.h>
#include <cycles/cycle.h>
//#include <norm.h>
#include <convergence.h>
#include <smoothedMG/smoothedMG_amg_level.h>
#include "TriMesh.h"
#include "tetmesh.h"

inline const char* getString(SolverType p) {
  switch(p)
  {
    case PCG_SOLVER:
      return "Preconditioned CG";
    case AMG_SOLVER:
      return "AMG";
    default:
      return "UNKNOWN";
  }
}

template <>
inline SolverType getValue<SolverType>(const char* name) {
  if(strncmp(name,"AMG",100)==0)
    return AMG_SOLVER;
  else if(strncmp(name,"PCG",100)==0)
    return PCG_SOLVER;

  char error[100];
  sprintf(error,"Solver type '%s' is not defined",name);
  FatalError(error);
}

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
  AMG(AMG_Config cfg);
  ~AMG();

  void solve(const Vector_d_CG &b, Vector_d_CG &x, bool verbose = false);
  void solve_iteration(const Vector_d_CG &b, Vector_d_CG &x, bool verbose = false);

  void setup(const Matrix_h &Acsr_import_h, const Matrix_d &Acsr_d, TriMesh* meshPtr, TetMesh* tetmeshPtr, bool verbose = false);

  void printGridStatistics();

  // profiling & debug output
  void printProfile();
  void printCoarsePoints();
  void printConnections();

  private:
  bool converged(const Vector &r, ValueType &nrm);

	cusp::detail::lu_solver<ValueType, cusp::host_memory> LU;

  AMG_Level<Matrix,Vector>* fine;
  AMG_Config cfg;
  ValueType initial_nrm;
  CGType tolerance;
  int iterations;
  int max_iters;
  int cycle_iters;
  int num_levels;
	int coarsestlevel;
  int presweeps,postsweeps;

  CycleType cycle;
//  NormType norm;
  ConvergenceType convergence;
	int DS_type;
  SolverType solver;
	Matrix_hyb_d_CG Ahyb_d_CG;

  double solve_start, solve_stop;
  double setup_start, setup_stop;
};
#endif
