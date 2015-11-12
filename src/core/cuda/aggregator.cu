
#include <smoothedMG/aggregators/mis.h>
/*********************************************
 * Allocates selector based on passed in type
 *********************************************/
template <class Matrix, class Vector>
Aggregator<Matrix, Vector>* Aggregator<Matrix, Vector>::allocate(FEMSolver *cfg)
{
  if (cfg->aggregatorType_ == 0)
    return new MIS_Aggregator < Matrix, Vector > ;
  else
    return new RandMIS_Aggregator < Matrix, Vector > ;
}

/****************************************
 * Explict instantiations
 ***************************************/
template class Aggregator < Matrix_h, Vector_h > ;
template class Aggregator < Matrix_d, Vector_d > ;
