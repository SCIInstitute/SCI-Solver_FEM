#ifndef __AMG_H__
#define __AMG_H__
template <class Matrix, class Vector> class AMG;

enum SolverType {AMG_SOLVER,PCG_SOLVER};

enum ConvergenceType { ABSOLUTE_CONVERGENCE, RELATIVE_CONVERGENCE };

#include <cusp/detail/lu.h>
#include <error.h>
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
    AMG(bool verbose, int convergeType, int cycleType,
      int solverType, double tolerance, int cycleIters, int maxIters,
      int maxLevels, int topSize, double smootherWeight,
    int preInnerIters, int postInnerIters, int postRelaxes,
    int dsType, int randMisParameters, int partitionMaxSize, double proOmega,
    int aggregatorType, int blockSize, TriMesh* triMesh, TetMesh* tetMesh);
  ~AMG();

  void solve(const Vector_d_CG &b, Vector_d_CG &x);
  void solve_iteration(const Vector_d_CG &b, Vector_d_CG &x);

  void setup(const Matrix_d &Acsr_d);

  void printGridStatistics();

  // profiling & debug output
  void printProfile();
  void printCoarsePoints();
  void printConnections();
  //config parameters
  bool verbose_;
  ConvergenceType convergeType_;
  CycleType cycleType_;
  SolverType solverType_;
  double tolerance_;
  int cycleIters_;
  int maxIters_;
  int maxLevels_;
  int topSize_;
  double smootherWeight_;
  int preInnerIters_;             // the pre inner iterations for GSINNER
  int postInnerIters_;            // the post inner iterations for GSINNER
  int postRelaxes_;               // the number of post relax iterations
  int dsType_;
  int randMisParameters_;
  int partitionMaxSize_;
  double proOmega_;
  int aggregatorType_;
  int blockSize_;
  TriMesh* triMesh_;
  TetMesh* tetMesh_;
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
