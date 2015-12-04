#include <cstdlib>
#include <cstdio>
#include "FEMSolver.h"
#include <cuda_resources.h>

/**
 * SCI-Solver_FEM :: Example 1
 * This example is the basic steps for running the solver:
 *  1. We define our main FEMSolver object.
 *  2. We set all of the parameters we want. (Otherwise defaults used.)
 *  3. We read in our input data mesh.
 *  4. We declare all the variables we need for the solver (matrices).
 *  5. We invoke the "setup_solver" call, which does all of the work.
 */

int main(int argc, char** argv)
{
  //Verbose option
  //Our main configuration object. We will set aspects where the
  // default values are not what we desire.
  FEMSolver cfg;
  bool zero_based = false;
  std::string Aname = "test.mat";
  cfg.filename_ = "../src/test/test_data/CubeMesh_size256step16";
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-v") == 0) {
      cfg.verbose_ = true;
    } else if (strcmp(argv[i], "-i") == 0) {
      if (i + 1 >= argc) break;
      cfg.filename_ = std::string(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-A") == 0) {
      if (i + 1 >= argc) break;
      Aname = std::string(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-z") == 0) {
      zero_based = true;
    }
  }
  //assuming our device is zero...
  cfg.device_ = 0;
  // Make sure part_max_size is representative of harware limits by default
  // when compiling the library, up to 64 registers were seen to be used on the
  // device. We can set our max allocation based on that number
  int max_registers_used = 64;
  cfg.partitionMaxSize_ = getMaxThreads(max_registers_used, cfg.device_);
  //set the convergence tolerance
  cfg.tolerance_= 1e-8;
  //set the weight parameter used in a smoother
  cfg.smootherWeight_ = 0.7;
  //set the weight parameter used in a prolongator smoother
  cfg.proOmega_ =  0.7;
  //set the maximum solve iterations
  cfg.maxIters_ = 10;
  //set the pre inner iterations for GSINNER
  cfg.preInnerIters_ = 2;
  //set the post inner iterations for GSINNER
  cfg.postInnerIters_ = 3;
  //set the Aggregator METIS (0) or MIS (1)
  cfg.aggregatorType_ = 1;
  //set the Max size of coarsest level
  cfg.metisSize_ = 90102;
  //set the solving algorithm
  cfg.solverType_ = /*(SolverType::)*/PCG_SOLVER;
  //set the cycle algorithm
  cfg.cycleType_ = /*(CycleType::)*/V_CYCLE;
  //set the convergence tolerance algorithm
  cfg.convergeType_ = /*(ConvergenceType::)*/ABSOLUTE_CONVERGENCE;
  //Now we read in the mesh of choice
  //TriMesh* meshPtr = TriMesh::read("mesh.ply"); //-----if we were reading a Triangle mesh
  //read in the Tetmesh
  cfg.tetMesh_ = TetMesh::read(
    (cfg.filename_ + ".node").c_str(),
    (cfg.filename_ + ".ele").c_str(), zero_based, cfg.verbose_);
  //The stiffness matrix A 
  Matrix_ell_h A_h;
  //get the basic stiffness matrix (constant) by creating the mesh matrix
  cfg.getMatrixFromMesh(&A_h);
  //intialize the b matrix to ones for now.
  Vector_h_CG b_h(A_h.num_rows, 1.0);
  //The answer vector.
  Vector_h_CG x_h(A_h.num_rows, 0.0); //intial X vector
  //************************ DEBUG : creating identity matrix for stiffness properties for now.
  Matrix_ell_h identity(A_h.num_rows, A_h.num_rows, A_h.num_rows, 1);
  for (int i = 0; i < A_h.num_rows; i++) {
	  identity.column_indices(i, 0) = i;
	  identity.values(i, 0) = 1;
  }
  //multiply the mesh matrix by the stiffness properties matrix.
  Matrix_ell_h out, test;
  Matrix_ell_h my_A(A_h);
  cusp::multiply(identity, my_A, out);
  A_h = Matrix_ell_h(out);

  if(FEMSolver::readMatlabSparseMatrix(Aname.c_str(),&test)) {
    std::cerr << "failed to read matlab file." << std::endl;
  }
  //************************ DEBUG*/
  //The final call to the solver
  cfg.checkMatrixForValidContents(&A_h);
  cfg.solveFEM(&A_h, &x_h, &b_h);
  //At this point, you can do what you need with the matrices.
  if (cfg.writeMatlabArray("output.mat", x_h)) {
	  std::cerr << "failed to write matlab file." << std::endl;
  }
  //write the VTK
  std::vector<float> vals;
  for (size_t i = 0; i < x_h.size(); i++){
    vals.push_back(x_h[i]);
  }
  int pos = cfg.filename_.find_last_of("/");
  if (pos == std::string::npos)
    pos = cfg.filename_.find_last_of("\\");
  cfg.filename_ = cfg.filename_.substr(pos + 1,
    cfg.filename_.size() - 1);
  cfg.writeVTK(vals, cfg.filename_);
  return 0;
}
