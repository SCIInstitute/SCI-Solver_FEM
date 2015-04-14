#include <cstdlib>
#include <cstdio>
#include "setup_solver.h"

/**
 * SCI-Solver_FEM :: Example 2
 * This example is nearly identical to Example 1, except:
 *  1. We are using a config file instead of setting parameters manually.
 *  2. We are using a different input mesh data set.
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
  //This example uses a difference data set and loads a config file
  cfg.parseFile("configs/PCGV");
  //Now we read in the mesh of choice
  //read in the Tetmesh
  std::string filename("example_data/CubeMesh_size256step16_correct");
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
  cusp::ell_matrix<int, float, cusp::host_memory> identity(A.num_rows, A.num_rows, A.num_rows, 1);
  for (int i = 0; i < A.num_rows; i++) {
	  identity.column_indices(i, 0) = i;
	  identity.values(i, 0) = 1;
  }
  //multiply the mesh matrix by the stiffness properties matrix.
  cusp::ell_matrix<int, float, cusp::host_memory> out;
  cusp::ell_matrix<int, float, cusp::host_memory> my_A = A;
  cusp::multiply(identity, my_A, out);
  A = out;
  //************************ DEBUG*/
  //The final call to the solver
  setup_solver(cfg, tetmeshPtr, &A, &b_d, &x_d, verbose);
  //At this point, you can do what you need with the matrices.
  // DO Something....
  return 0;
}
