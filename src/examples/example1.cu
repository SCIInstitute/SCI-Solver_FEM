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
 *  5. We invoke the "setupFEM" call, which does all of the work.
 */

int main(int argc, char** argv)
{
  //Verbose option
  //Our main configuration object. We will set aspects where the
  // default values are not what we desire.
  FEMSolver cfg;
  std::string Aname = "", bName;
  cfg.filename_ = "../src/test/test_data/CubeMesh_size256step16";
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-v") == 0) {
      cfg.verbose_ = true;
    } else if (strcmp(argv[i], "-i") == 0) {
      if (i + 1 >= argc) break;
      cfg.filename_ = std::string(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-b") == 0) {
      if (i + 1 >= argc) break;
      bName = std::string(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-A") == 0) {
      if (i + 1 >= argc) break;
      Aname = std::string(argv[i + 1]);
      i++;
    }
  }
  //assuming our device is zero...
  cfg.device_ = 0;
  // Make sure part_max_size is representative of harware limits by default
  // when compiling the library, up to 64 registers were seen to be used on the
  // device. We can set our max allocation based on that number
  /*int max_registers_used = 64;
  cfg.partitionMaxSize_ = getMaxThreads(max_registers_used, cfg.device_);
  cfg.tolerance_= 1e-8;
  cfg.smootherWeight_ = 0.7;
  cfg.proOmega_ =  0.7;
  cfg.maxIters_ = 10;
  cfg.preInnerIters_ = 2;
  cfg.postInnerIters_ = 3;
  cfg.aggregatorType_ = 1;
  cfg.solverType_ = PCG_SOLVER;
  cfg.cycleType_ = V_CYCLE;
  cfg.convergeType_ = ABSOLUTE_CONVERGENCE;*/
  //Now we read in the mesh of choice
  //TriMesh* meshPtr = TriMesh::read("mesh.ply"); //-----if we were reading a Triangle mesh
  //read in the Tetmesh
  cfg.tetMesh_ = TetMesh::read(
    (cfg.filename_ + ".node").c_str(),
    (cfg.filename_ + ".ele").c_str(), cfg.verbose_);
  //The stiffness matrix A 
  Matrix_ell_h A_h;
  if (Aname.empty()) {
    //get the basic stiffness matrix (constant) by creating the mesh matrix
    cfg.getMatrixFromMesh(&A_h);
  } else {
    //Import stiffness matrix (A)
    if (cfg.importStiffnessMatrixFromFile(Aname, &A_h, cfg.verbose_) < 0)
      return 0;
  }
  //intialize the b matrix to ones for now.
  Vector_h_CG b_h(A_h.num_rows, 1.0);
  if (!bName.empty()) {
    //Import right-hand-side single-column array (b)
    if (cfg.importRhsVectorFromFile(bName, b_h, cfg.verbose_) < 0)
      return 0;
  }
  //The answer vector.
  Vector_h_CG x_h(A_h.num_rows, 0.0); //intial X vector
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
