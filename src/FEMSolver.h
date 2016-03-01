#ifndef __FEMSOLVER_H__
#define __FEMSOLVER_H__

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include "TriMesh.h"
#include "tetmesh.h"
#include "types.h"

/** The class that represents all of the available options for FEM */
class FEMSolver {
  class SparseEntry_t {
  public:
    int32_t row_;
    int32_t col_;
    float   val_;
    SparseEntry_t(int32_t r, int32_t c, float v) : row_(r), col_(c), val_(
      static_cast<float>(v)) {}
    ~SparseEntry_t() {}
  };

  public:
  FEMSolver(std::string fname = "../src/test/test_data/simple",
      bool verbose = false);
  virtual ~FEMSolver();
  public:
  void solveFEM(Matrix_ell_h* A_h, Vector_h_CG* x_h, Vector_h_CG* b_h);
  void getMatrixFromMesh(Matrix_ell_h* A_h);
  static int readMatlabSparseMatrix(const std::string &filename, Matrix_ell_h *A_h);
  static int readMatlabNormalMatrix(const std::string &filename, std::vector<double> *A_h);
  int writeMatlabArray(const std::string &filename, const Vector_h_CG &array);
  void checkMatrixForValidContents(Matrix_ell_h* A_h);
  void writeVTK(std::vector <float> values, std::string fname);
  void printElementWithHeader(std::vector<double>& test, unsigned int index);
  void printMatlabReadContents(std::vector<double>& test);
  int importRhsVectorFromFile(std::string filename,
      Vector_h_CG& targetVector, bool verbose);
  int importStiffnessMatrixFromFile(std::string filename,
      Matrix_ell_h* targetMatrix, bool verbose);
  void debugPrintMatlabels(TetMesh* mesh);
  private:
  bool InitCUDA();
  static bool compare_sparse_entry(SparseEntry_t a, SparseEntry_t b);
  public:
  //data
  bool verbose_;                  // output verbosity
  std::string filename_;          // mesh file name
  int maxLevels_;                 // the maximum number of levels
  int maxIters_;                  // the maximum solve iterations
  int preInnerIters_;             // the pre inner iterations for GSINNER
  int postInnerIters_;            // the post inner iterations for GSINNER
  int postRelaxes_;               // the number of post relax iterations
  int cycleIters_;                // the number of CG iterations per outer iteration
  int dsType_;                    // data structure type
  int topSize_;                   // max size of coarsest level
  int randMisParameters_;         // max size of coarsest level
  int partitionMaxSize_;          // max size of of the partition
  int aggregatorType_;            // aggregator oldMis (0), metis bottom up (1), 
                                  //   metis top down (2), aggMisGPU (3), aggMisCPU (4), newMisLight (5)
  int convergeType_;              // the convergence tolerance algorithm <absolute (0)|relative (1)>
  double tolerance_;              // the convergence tolerance
  int cycleType_;                 // the cycle algorithm <V (0) | W (1) | F (2) | K (3)>
  int solverType_;                // the solving algorithm <AMG (0) | PCG (1)>
  double smootherWeight_;         // the weight parameter used in a smoother
  double proOmega_;               // the weight parameter used in prolongator smoother
  int device_;                    // the GPU device number to specify
  int blockSize_;
  //The pointers to the meshes
  TetMesh * tetMesh_;
  TriMesh * triMesh_;
};

#endif
