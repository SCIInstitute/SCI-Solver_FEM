#include <amg_config.h>
#include <amg_level.h>
//#include <norm.h>
#include <convergence.h>
#include <smoothers/smoother.h>
#include <cycles/cycle.h>
#include <smoothedMG/aggregators/aggregator.h>

inline void registerParameters()
{
  AMG_Config::registerParameter<int>("max_levels", "the maximum number of levels", 100);
  AMG_Config::registerParameter<int>("min_rows", "the minimum number of rows in a level", 1);
  AMG_Config::registerParameter<int>("max_iters", "the maximum solve iterations", 100);
  AMG_Config::registerParameter<int>("PreINNER_iters", "the pre inner iterations for GSINNER", 5);
  AMG_Config::registerParameter<int>("PostINNER_iters", "the post inner iterations for GSINNER", 5);
  AMG_Config::registerParameter<int>("post_relaxes", "the number of post relax iterations", 1);
  AMG_Config::registerParameter<int>("presweeps", "the number of presmooth iterations", 1);
  AMG_Config::registerParameter<int>("postsweeps", "the number of postsmooth iterations", 1);
  AMG_Config::registerParameter<int>("cycle_iters", "the number of CG iterations per outer iteration", 1);
  AMG_Config::registerParameter<int>("DS_type", "data structure type", 0);
  AMG_Config::registerParameter<int>("top_size", "max size of coarsest level", 256);
  AMG_Config::registerParameter<int>("metis_size", "max size of coarsest level", 256);
  AMG_Config::registerParameter<int>("mesh_type", "tri mesh (0) or tet mesh (1)", 0);
  AMG_Config::registerParameter<int>("part_max_size", "set maximum partition size", 600);
  AMG_Config::registerParameter<int>("aggregator_type", "aggregator metis (0) mis (1)", 0);
//  AMG_Config::registerParameter<double>("rescale_size", "mesh the rescale size", 0.0);
  AMG_Config::registerParameter<int>("cuda_device_num", "set the CUDA device number", 0);
//  AMG_Config::registerParameter<NormType > ("norm", "the norm used for convergence testing <L1|L2|LMAX>", L2);
  AMG_Config::registerParameter<ConvergenceType > ("convergence", "the convergence tolerance algorithm <absolute|relative>", ABSOLUTE);
  AMG_Config::registerParameter<double>("tolerance", "the convergence tolerance", 1e-6);
  AMG_Config::registerParameter<CycleType > ("cycle", "the cycle algorithm <V|W|F|KCG|PCGV|PCGW|PCGF>", V_CYCLE);
  AMG_Config::registerParameter<SmootherType > ("smoother", "the smoothing algorithm <JACOBI>", JACOBI);
  AMG_Config::registerParameter<SolverType > ("solver", "the solving algorithm <AMG|PCG>", AMG_SOLVER);
  AMG_Config::registerParameter<AlgorithmType > ("algorithm", "the AMG algorithm <classical>", CLASSICAL);
  AMG_Config::registerParameter<double>("smoother_weight", "the weight parameter used in a smoother", 1.0);
  AMG_Config::registerParameter<double>("pro_omega", "the weight parameter used in prolongator smoother", 0.67);
}
