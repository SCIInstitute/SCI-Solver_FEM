#include <cstdlib>
#include <cstdio>
#include "setup_solver.h"
#include "cuda_resources.h"

/**
 * SCI-Solver_FEM :: Example 3
 * This example is the basic steps for running the solver:
 *  1. We define our main AMG_Config object.
 *  2. We set all of the parameters we want. (Otherwise defaults used.)
 *  3. We read in our input data mesh.
 *  4. We declare all the variables we need for the solver (matrices).
 *  5. We invoke the "setup_solver" call, which does all of the work.
 */

void printElementWithHeader(vector<double>& test, unsigned int index)
{
  std::cout << "element #" << index << " = " << test[index] << std::endl;
}

void printMatlabReadContents(vector<double>& test)
{
  std::cout << "test result vector is size = " << test.size() << std::endl;
  int incr = test.size() / 5;
  for (int j = 0; j < test.size(); j += incr) {
    printElementWithHeader(test, j);
  }
  if( test.size() > 0 )
    std::cout << "last element = " << test[test.size() - 1] << std::endl;
}

int importRhsVectorFromFile(string filename, Vector_h_CG& targetVector, bool verbose)
{
  vector<double> sourceRead;

  if( filename.empty() ) {
    string errMsg = "No matlab file provided for RHS (b) vector.";
    errMsg += " Specify the file using argument at commandline using -b switch.";
    std::cerr << errMsg << std::endl;
    return -1;
  }
  if( readMatlabNormalMatrix(filename, &sourceRead) < 0 ) {
    std::cerr << "Failed to read matlab file for RSH (b)." << std::endl;
    return -1;
  }
  targetVector = sourceRead;
  if (verbose) {
    int sizeRead = targetVector.size();
    std::cout << "Finished reading RHS (b) data file with ";
    std::cout << sizeRead << " values." << std::endl;
  }
  return 0;
}

int importStiffnessMatrixFromFile(string filename, Matrix_ell_h* targetMatrix, bool verbose)
{
  if( filename.empty() ) {
    string errMsg = "No matlab file provided for stiffness matrix (A).";
    errMsg += " Specify the file using argument at commandline using -A switch.";
    std::cerr << errMsg << std::endl;
    return -1;
  }
  if( readMatlabSparseMatrix(filename, targetMatrix) != 0 ) {
    std::cerr << "Failed to read matlab file for stiffness matrix (A)." << std::endl;
    return -1;
  }
  if (verbose) {
    string msg = "Finished reading stiffness matrix.";
    std::cout << msg << std::endl;
  }
  return 0;
}

void debugPrintMatlabels(TetMesh* mesh)
{
  std::cout << "Found " << mesh->matlabels.size() << " elements in matlabels." << std::endl;
  unsigned int numZeros = 0;
  for (vector<int>::iterator it = mesh->matlabels.begin(); it != mesh->matlabels.end(); ++it)
  {
    if( (*it) == 0 )
      numZeros++;
    else
      std::cout << (*it) << std::endl;
  }
  std::cout << numZeros << " zero values found." << std::endl;
}


int main(int argc, char** argv)
{
  //Verbose option
  bool verbose = false;
  bool zero_based = false;
  std::string filename, aFilename, bFilename;
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i],"-v") == 0) {
      verbose = true;
    } else if (strcmp(argv[i],"-i") == 0) {
      if (i+1 >= argc)
    	break;
      filename = std::string(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-b") == 0) {
      if (i+1 >= argc)
    	break;
      bFilename = std::string(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-A") == 0) {
      if (i+1 >= argc)
    	break;
      aFilename = std::string(argv[i+1]);
      i++;
    } else if (strcmp(argv[i],"-z") == 0) {
      zero_based = true;
    }
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
  if (filename.empty())
    filename = std::string("../example_data/CubeMesh_size256step16");
  if (verbose)
    std::cout << "Reading in file: " << filename << std::endl;
  TetMesh* tetmeshPtr = TetMesh::read(
      (filename + ".node").c_str(),
      (filename + ".ele").c_str(), zero_based, verbose);

  //The stiffness matrix A 
  Matrix_ell_h A_h;
  //get the basic stiffness matrix (constant) by creating the mesh matrix
  getMatrixFromMesh(cfg, tetmeshPtr, &A_h, true, verbose);

  //Import right-hand-side single-column array (b)
  Vector_h_CG b_h;
  if( importRhsVectorFromFile(bFilename, b_h, verbose) < 0 )
    return 0;

  //The answer vector.
  Vector_h_CG x_h(A_h.num_rows, 0.0); //intial X vector

#define USE_IDENTITY_MATRIX_ONLY

#ifndef USE_IDENTITY_MATRIX_ONLY
  //Import stiffness matrix (A)
  Matrix_ell_h A_h_import;
  if( importStiffnessMatrixFromFile(aFilename, &A_h_import, verbose) < 0 )
    return 0;
#else //USE_IDENTITY_MATRIX_ONLY
  Matrix_ell_h A_h_import(A_h.num_rows, A_h.num_rows, A_h.num_rows, 1.0);
  for (int i = 0; i < A_h.num_rows; i++) {
    A_h_import.column_indices(i, 0) = i;
    A_h_import.values(i, 0) = 1;
  }
#endif //#ifdef USE_IDENTITY_MATRIX_ONLY

  //multiply the mesh matrix by the stiffness properties matrix.
  Matrix_ell_h out;
  cusp::multiply(A_h_import, A_h, out);
  A_h = Matrix_ell_h(out);

  if( verbose )
    std::cout << "Calling setup_solver." << std::endl;
  //The final call to the solver
  checkMatrixForValidContents(&A_h, verbose);
  Matrix_ell_d A_d(A_h);

  setup_solver(cfg, tetmeshPtr, &A_d, &x_h, &b_h, verbose);
  //At this point, you can do what you need with the matrices.
  if (writeMatlabArray("output.mat", x_h)) {
    std::cerr << "failed to write matlab file." << std::endl;
  }
  return 0;
}
