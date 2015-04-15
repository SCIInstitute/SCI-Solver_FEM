#include <cstdlib>
#include <cstdio>
#include "setup_solver.h"
#include "cuda_resources.h"

/**
 * SCI-Solver_FEM :: Example 1
 * This example is the basic steps for running the solver:
 *  1. We define our main AMG_Config object.
 *  2. We set all of the parameters we want. (Otherwise defaults used.)
 *  3. We read in our input data mesh.
 *  4. We declare all the variables we need for the solver (matrices).
 *  5. We invoke the "setup_solver" call, which does all of the work.
 */

int main(int argc, char** argv)
{
  //Verbose option
  bool verbose = false;
  for (int i = 0; i < argc; i++)
    if (strcmp(argv[i],"-v") == 0) {
      verbose = true;
      break;
    }
  //Our main configuration object. We will set aspects where the
  // default values are not what we desire.
  AMG_Config cfg;
  //assuming our device is zero...
  int dev_num = 0;
  cfg.setParameter("cuda_device_num", dev_num);
  // Make sure part_max_size is representative of harware limits by default
  // when compiling the library, up to 64 registers were seen to be used on the
  // device. We can set our max allocation based on that number
  int max_registers_used = 64;
  cfg.setParameter("part_max_size", getMaxThreads(max_registers_used,dev_num));
  //set the desired algorithm
  cfg.setParameter("algorithm", /*(AlgorithmType::)*/CLASSICAL);
  //set the convergence tolerance
  cfg.setParameter("tolerance", 1e-8);
  //set the weight parameter used in a smoother
  cfg.setParameter("smoother_weight", 0.7);
  //set the weight parameter used in a prolongator smoother
  cfg.setParameter("pro_omega", 0.7);
  //set the maximum solve iterations
  cfg.setParameter("max_iters", 10);
  //set the pre inner iterations for GSINNER
  cfg.setParameter("PreINNER_iters", 2);
  //set the post inner iterations for GSINNER
  cfg.setParameter("PostINNER_iters", 3);
  //set the Aggregator METIS (0) or MIS (1)
  cfg.setParameter("aggregator_type", 1);
  //set the Max size of coarsest level
  cfg.setParameter("metis_size", 90102);
  //set the solving algorithm
  cfg.setParameter("solver", /*(SolverType::)*/PCG_SOLVER);
  //set the cycle algorithm
  cfg.setParameter("cycle", /*(CycleType::)*/V_CYCLE);
  //set the convergence tolerance algorithm
  cfg.setParameter("convergence", /*(ConvergenceType::)*/ABSOLUTE_CONVERGENCE);
  //set the smoothing algorithm
  cfg.setParameter("smoother", /*(SmootherType::)*/GAUSSSEIDEL);
  //Now we read in the mesh of choice
  //TriMesh* meshPtr = TriMesh::read("mesh.ply"); //-----if we were reading a Triangle mesh
  //read in the Tetmesh
  std::string filename("../example_data/CubeMesh_size256step16");
  TetMesh* tetmeshPtr = TetMesh::read(
      (filename + ".node").c_str(),
      (filename + ".ele").c_str());
  //The stiffness matrix A 
  Matrix_d A;
  //get the basic stiffness matrix (constant) by creating the mesh matrix
  getMatrixFromMesh(cfg, tetmeshPtr, &A, verbose);
  //intialize the b matrix to ones for now. TODO @DEBUG
  Vector_h_CG b_d(A.num_rows, 1.0);
  //The answer vector.
  Vector_h_CG x_d(A.num_rows, 0.0); //intial X vector
  //************************ DEBUG : creating identity matrix for stiffness properties for now.
  cusp::ell_matrix<int, float, cusp::host_memory> identity(A.num_rows,A.num_rows,A.num_rows, 1);
  for (int i = 0; i < A.num_rows; i++) {
	  identity.column_indices(i, 0) = i;
	  identity.values(i, 0) = 1;
  }
  //multiply the mesh matrix by the stiffness properties matrix.
  cusp::ell_matrix<int, float, cusp::host_memory> out, test;
  cusp::ell_matrix<int, float, cusp::host_memory> my_A = A;
  cusp::multiply(identity, my_A, out);
  A = out;

  if(readMatlabFile("/home/sci/brig/Documents/Data/FEM_testing/mysparse.mat",&test)) {
    std::cerr << "failed to read matlab file." << std::endl;
  } else {
    cusp::print(test);
  }
  //************************ DEBUG*/
  //The final call to the solver
  setup_solver(cfg, tetmeshPtr, &A, &b_d, &x_d, verbose);
  //At this point, you can do what you need with the matrices.
  // DO Something....
  return 0;
}
