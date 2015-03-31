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
  cfg.parseFile("../configs/PCGV");
  //Now we read in the mesh of choice
  //read in the Tetmesh
  std::string filename("../example_data/CubeMesh_size256step16_correct");
  TetMesh* tetmeshPtr = TetMesh::read(
      (filename + ".node").c_str(),
      (filename + ".ele").c_str());
  cfg.setParameter("mesh_type",1); //0 is Trimesh, 1 is Tetmesh
  //All of the required variables to run the solver.
  Matrix_d A;
  FEM2D fem2d;
  FEM3D fem3d;
  Vector_d_CG b_d;
  Vector_d_CG x_d;
  //The final call to the solver
  setup_solver(cfg, NULL, tetmeshPtr, &fem2d, &fem3d, &A, &b_d, &x_d, verbose);
  //At this point, you can do what you need with the matrices.
  // DO Something....
  return 0;
}
