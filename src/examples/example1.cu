#include <cstdlib>
#include <cstdio>
#include "setup_solver.h"
#include "cuda_resources.h"

int main(int argc, char** argv)
{
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
  cfg.setParameter("convergence", /*(ConvergenceType::)*/ABSOLUTE);
  //set the smoothing algorithm
  cfg.setParameter("smoother", /*(SmootherType::)*/GAUSSSEIDEL);
  //Now we read in the mesh of choice
  //TriMesh* meshPtr = TriMesh::read("mesh.ply"); //-----if we were reading a Triangle mesh
  //read in the Tetmesh
  std::string filename("../example_data/CubeMesh_size256step16");
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
