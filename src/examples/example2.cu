#include <cstdlib>
#include <cstdio>
#include "FEMSolver.h"
#include <string>

/**
 * SCI-Solver_FEM :: Example 2
 * This example is nearly identical to Example 1, except:
 *  1. We are using a tri mesh data set.
 */

int main(int argc, char** argv)
{
  //Verbose option
  FEMSolver cfg;
  bool zero_based = false;
  for (int i = 0; i < argc; i++)
    if (strcmp(argv[i],"-v") == 0) {
      cfg.verbose_ = true;
      break;
    } else if (strcmp(argv[i],"-z") == 0) {
      zero_based = true;
    }
  //Our main configuration object. We will set aspects where the
  // default values are not what we desire.
  //Now we read in the mesh of choice
  //read in the Tetmesh
  cfg.filename_ = "example_data/sphere_266verts.ply";
  FEMSolver::triMesh_ = TriMesh::read(cfg.filename_.c_str());
  //The stiffness matrix A 
  Matrix_ell_h A;
  //get the basic stiffness matrix (constant) by creating the mesh matrix
  cfg.getMatrixFromMesh(&A);
  //intialize the b matrix to ones for now. TODO @DEBUG
  Vector_h_CG b_h(A.num_rows, 1.0);
  //The answer vector.
  Vector_h_CG x_h(A.num_rows, 0.0); //intial X vector
  //************************ DEBUG : creating identity matrix for stiffness properties for now.
  Matrix_ell_h identity(A.num_rows, A.num_rows, A.num_rows, 1);
  for (int i = 0; i < A.num_rows; i++) {
	  identity.column_indices(i, 0) = i;
	  identity.values(i, 0) = 1;
  }
  //multiply the mesh matrix by the stiffness properties matrix.
  Matrix_ell_h out;
  Matrix_ell_h my_A(A);
  cusp::multiply(identity, my_A, out);
  A = Matrix_ell_h(out);
  //************************ DEBUG*/
  //The final call to the solver
  cfg.checkMatrixForValidContents(&A);
  cfg.solveFEM(&A, &x_h, &b_h);
  //At this point, you can do what you need with the matrices.
  cfg.writeMatlabArray("output.mat", x_h);
  //write the VTK
  std::vector<float> vals;
  for (size_t i = 0; i < x_h.size(); i++){
    vals.push_back(x_h[i]);
  }
  auto pos = cfg.filename_.find_last_of("/");
  if (pos == std::string::npos)
    pos = cfg.filename_.find_last_of("\\");
  cfg.filename_ = cfg.filename_.substr(pos + 1,
    cfg.filename_.size() - 1);
  cfg.writeVTK(vals, cfg.filename_);
  return 0;
}
