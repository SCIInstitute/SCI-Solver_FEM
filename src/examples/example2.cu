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
  bool verbose = false;
  std::string filename = "../src/test/test_data/sphere_290verts.ply";
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-v") == 0) {
      verbose = true;
    } else if (strcmp(argv[i], "-i") == 0) {
      if (i + 1 >= argc) break;
      filename = std::string(argv[i + 1]);
      i++;
    }
  }
  //Our main configuration object. We will set aspects where the
  // default values are not what we desire.
  FEMSolver cfg(filename, false, verbose);
  //intialize the b matrix to ones for now. 
  Vector_h_CG b_h(cfg.getMatrixRows(), 1.0);
  //The answer vector.
  Vector_h_CG x_h(cfg.getMatrixRows(), 0.0); //intial X vector
  //The final call to the solver
  cfg.solveFEM(&x_h, &b_h);
  //At this point, you can do what you need with the matrices.
  cfg.writeMatlabArray("output.mat", x_h);
  //write the VTK
  std::vector<float> vals;
  for (size_t i = 0; i < x_h.size(); i++){
    vals.push_back(x_h[i]);
  }
  int pos = cfg.filename_.find_last_of("/");
  if (pos == std::string::npos)
    pos = cfg.filename_.find_last_of("\\");
  std::string outname = cfg.filename_.substr(pos + 1,
    cfg.filename_.size() - 1);
  pos = outname.find_last_of(".");
  outname = outname.substr(0, pos);
  cfg.writeVTK(vals, outname);
  return 0;
}
