#ifndef __FEMSOLVER_H__
#define __FEMSOLVER_H__

#include <stdio.h>
#include <iostream>
#include <signal.h>
#include <exception>
#include <fstream>
#include <types.h>
#include <TriMesh.h>
#include <tetmesh.h>
#include <cutil.h>
#include <FEM/FEM2D.h>
#include <FEM/FEM3D.h>
#include <my_timer.h>
#include <amg.h>
#include <typeinfo>
#include <time.h>
#include <cmath>
#include "smoothers/smoother.h"
#include "cycles/cycle.h"
#include "convergence.h"
#include "amg_level.h"

#ifdef WIN32
#include <cstdlib>
#define srand48 srand
#endif

/** The class that represents all of the available options for FEM */
class FEMSolver {
  class SparseEntry_t {
  public:
    int32_t row_;
    int32_t col_;
    float   val_;
    SparseEntry_t(int32_t r, int32_t c, int32_t v) : row_(r), col_(c), val_(
      static_cast<float>(v)) {}
    ~SparseEntry_t() {}
  };

public:
  FEMSolver(std::string fname = "../src/test/test_data/sphere334",
    bool verbose = false);
  virtual ~FEMSolver();
public:
  void solveFEM(Matrix_ell_h* A_d, Vector_h_CG* x_h, Vector_h_CG* b_h);
  void getMatrixFromMesh(Matrix_ell_h* A_h);
  int readMatlabSparseMatrix(const std::string &filename, Matrix_ell_h *A_h);
  int readMatlabNormalMatrix(const std::string &filename, vector<double> *A_h);
  int writeMatlabArray(const std::string &filename, const Vector_h_CG &array);
  void checkMatrixForValidContents(Matrix_ell_h* A_h);
  void writeVTK(std::vector <float> values, std::string fname);
private:
  bool InitCUDA();
  static bool compare_sparse_entry(SparseEntry_t a, SparseEntry_t b);
public:
  //data
  bool verbose_;                  // output verbosity
  std::string filename_;          // mesh file name
  int maxLevels_;                 // the maximum number of levels
  int minRows_;                   // the minimum number of rows in a level
  int maxIters_;                  // the maximum solve iterations
  int preInnerIters_;             // the pre inner iterations for GSINNER
  int postInnerIters_;            // the post inner iterations for GSINNER
  int postRelaxes_;               // the number of post relax iterations
  int preSweeps_;                 // the number of presmooth iterations
  int postSweeps_;                // the number of postsmooth iterations
  int cycleIters_;                // the number of CG iterations per outer iteration
  int dsType_;                    // data structure type
  int topSize_;                   // max size of coarsest level
  int metisSize_;                 // max size of coarsest level
  int partitionMaxSize_;          // max size of of the partition
  int aggregatorType_;            // aggregator metis (0) mis (1)
  ConvergenceType convergeType_;  // the convergence tolerance algorithm <absolute|relative>
  double tolerance_;              // the convergence tolerance
  CycleType cycleType_;           // the cycle algorithm <V|W|F|KCG|PCGV|PCGW|PCGF>
  SmootherType smootherType_;     // the smoothing algorithm <GAUSSSEIDEL>
  SolverType solverType_;         // the solving algorithm <AMG|PCG>
  AlgorithmType algoType_;        // the AMG algorithm <classical>
  double smootherWeight_;         // the weight parameter used in a smoother
  double proOmega_;               // the weight parameter used in prolongator smoother
  int device_;                    // the GPU device number to specify
  //The static pointer to the mesh
  TetMesh * tetMesh_;
  TriMesh * triMesh_;
};

#endif
