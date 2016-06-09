#include "FEMSolver.h"
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
  //option
  std::string Aname, bName, ansName, fname = "../src/test/test_data/CubeMesh_size256step16";
  bool verbose = false;
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-v") == 0) {
      verbose = true;
    } else if (strcmp(argv[i], "-i") == 0) {
      if (i + 1 >= argc) break;
      fname = std::string(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-b") == 0) {
      if (i + 1 >= argc) break;
      bName = std::string(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-a") == 0) {
      if (i + 1 >= argc) break;
      ansName = std::string(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-A") == 0) {
      if (i + 1 >= argc) break;
      Aname = std::string(argv[i + 1]);
      i++;
    }
  }
  //Our main configuration object. We will set aspects where the
  // default values are not what we desire.
  FEMSolver cfg(fname, true, verbose);
  if (!Aname.empty()) {
    //Import stiffness matrix (A)
    if (cfg.readMatlabSparseMatrix(Aname) != 0)
      std::cerr << "Failed to read in A matrix: " << Aname << std::endl;
  }
  //intialize the b matrix to ones for now.
  Vector_h_CG b_h(cfg.getMatrixRows(), 1.0);
  if (!bName.empty()) {
    //Import right-hand-side single-column array (b)
    if (cfg.readMatlabArray(bName, &b_h) != 0)
      std::cerr << "Failed to read in b array: " << bName << std::endl;
  }
  //intialize the ans matrix to ones for now.
  Vector_h_CG ans_h(cfg.getMatrixRows(), 1.0);
  if (!ansName.empty()) {
    //Import right-hand-side single-column array (b)
    if (cfg.readMatlabArray(ansName, &ans_h) != 0)
      std::cerr << "Failed to read in ans array: " << bName << std::endl;
  }
  //write the VTK
  std::vector<double> ans;
  for (size_t i = 0; i < ans_h.size(); i++){
    ans.push_back(ans_h[i]);
  }
  cfg.writeVTK(ans, "answer");
  std::cout << "wrote out answer mesh" << std::endl;
  //The answer vector.
  Vector_h_CG x_h(cfg.getMatrixRows(), 0.0); //intial X vector
  //The final call to the solver
  cfg.solveFEM(&x_h, &b_h);
  //At this point, you can do what you need with the matrices.
  if (cfg.writeMatlabArray("output.mat", x_h)) {
	  std::cerr << "failed to write matlab file." << std::endl;
  }
  //write the VTK
  std::vector<double> vals;
  for (size_t i = 0; i < x_h.size(); i++){
    vals.push_back(x_h[i]);
  }
  int pos = cfg.filename_.find_last_of("/");
  if (pos == std::string::npos)
    pos = cfg.filename_.find_last_of("\\");
  std::string outname = cfg.filename_.substr(pos + 1,
    cfg.filename_.size() - 1);
  cfg.writeVTK(vals, outname);
  return 0;
}
