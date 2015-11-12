#include <amg_level.h>
#include <types.h>
#include <cutil.h>

template <class Matrix, class Vector>
AMG_Level<Matrix, Vector>::~AMG_Level()
{
   if(smoother != 0) delete smoother;
   if(next != 0) delete next;
}

#include<smoothedMG/smoothedMG_amg_level.h>
template <class Matrix, class Vector>
AMG_Level<Matrix, Vector>* AMG_Level<Matrix, Vector>::allocate(AMG<Matrix, Vector>*amg)
{
  AlgorithmType alg = amg->cfg->algoType_;
   switch(alg)
   {
   case CLASSICAL:
   default:
      return new SmoothedMG_AMG_Level<Matrix, Vector > (amg);
   }
}

/******************************************************
 * Recusively solves the system on this level
 ******************************************************/
template <class Matrix, class Vector>
void AMG_Level<Matrix, Vector>::cycle(CycleType cycle, Vector_d& b_d, Vector_d& x_d, bool verbose)
{
   if(isCoarsest()) //solve directly
   {
      cusp::array1d<ValueType, cusp::host_memory> temp_b(b_d);
      cusp::array1d<ValueType, cusp::host_memory> temp_x(x_d.size());
      amg->LU(temp_b, temp_x);
      x_d = temp_x;
      return;
   }
   else
   {
      switch (DS_type)
      {
      case 0:
         smoother->preRRRFullSymmetric(AinSysCoo_d, AoutSys_d, AinBlockIdx_d, AoutBlockIdx_d, aggregateIdx_d, partitionIdx_d, restrictorFull_d, ipermutation_d, b_d, x_d, bc_d,
               level_id, largestblocksize, largest_num_entries, verbose);
         break;
      case 1:
         smoother->preRRRFull(AinEll_d, Aout_d, aggregateIdx_d, partitionIdx_d, restrictorFull_d, ipermutation_d, b_d, x_d, bc_d, level_id, largestblocksize);
         break;
      case 2:
         smoother->preRRRFullCsr(AinCSR_d, Aout_d, aggregateIdx_d, partitionIdx_d, restrictorFull_d, ipermutation_d, b_d, x_d, bc_d, level_id, largestblocksize, largest_num_entries, largest_num_per_row);
         break;
      default:
         cout << "Wrong DStype 1!" << endl;
         exit(0);

      }
      next->cycle(V_CYCLE, bc_d, xc_d, verbose);
      switch (DS_type)
      {
      case 0:
         smoother->postPCRFullSymmetric(AinSysCoo_d, AinBlockIdx_d, AoutSys_d, AoutBlockIdx_d, aggregateIdx_d, partitionIdx_d, prolongatorFull_d, ipermutation_d, b_d, x_d, xc_d,
               level_id, largestblocksize, largest_num_entries);
         break;
      case 1:
         smoother->postPCRFull(AinEll_d, Aout_d, AoutBlockIdx_d, aggregateIdx_d, partitionIdx_d, prolongatorFull_d, ipermutation_d, b_d, x_d, xc_d, level_id, largestblocksize);
         break;
      case 2:
         smoother->postPCRFullCsr(AinCSR_d, Aout_d, AoutBlockIdx_d, aggregateIdx_d, partitionIdx_d, prolongatorFull_d, ipermutation_d, b_d, x_d, xc_d, level_id, largestblocksize, largest_num_entries, largest_num_per_row);
         break;
      default:
         cout << "Wrong DStype 0!" << endl;
         exit(0);

      }

   }
}


template <class Matrix, class Vector>
void AMG_Level<Matrix, Vector>::cycle_level0(CycleType cycle, Vector_d_CG &b_d_CG, Vector_d_CG &x_d_CG, bool verbose)
{
   if(isCoarsest()) //solve directly
   {
      cusp::array1d<ValueType, cusp::host_memory> temp_b = b_d_CG;
      cusp::array1d<ValueType, cusp::host_memory> temp_x(x_d_CG.size());
      amg->LU(temp_b, temp_x);
      x_d_CG = temp_x;

      return;
   }
   else
   {
      Vector_d b_d = b_d_CG;
      Vector_d x_d(x_d_CG.size(), 0.0);
      switch (DS_type)
      {
      case 0:
         smoother->preRRRFullSymmetric(AinSysCoo_d, AoutSys_d, AinBlockIdx_d, AoutBlockIdx_d, aggregateIdx_d, partitionIdx_d, restrictorFull_d, ipermutation_d, b_d, x_d, bc_d,
               level_id, largestblocksize, largest_num_entries, verbose);
         break;
      case 1:
         smoother->preRRRFull(AinEll_d, Aout_d, aggregateIdx_d, partitionIdx_d, restrictorFull_d, ipermutation_d, b_d, x_d, bc_d, level_id, largestblocksize);
         break;
      case 2:
         smoother->preRRRFullCsr(AinCSR_d, Aout_d, aggregateIdx_d, partitionIdx_d, restrictorFull_d, ipermutation_d, b_d, x_d, bc_d, level_id, largestblocksize, largest_num_entries, largest_num_per_row);
         break;
      default:
         cout << "Wrong DStype 1!" << endl;
         exit(0);

      }
      next->cycle(V_CYCLE, bc_d, xc_d,verbose);
      switch (DS_type)
      {
      case 0:
         smoother->postPCRFullSymmetric(AinSysCoo_d, AinBlockIdx_d, AoutSys_d, AoutBlockIdx_d, aggregateIdx_d, partitionIdx_d, prolongatorFull_d, ipermutation_d, b_d, x_d, xc_d,
               level_id, largestblocksize, largest_num_entries);
         break;
      case 1:
         smoother->postPCRFull(AinEll_d, Aout_d, AoutBlockIdx_d, aggregateIdx_d, partitionIdx_d, prolongatorFull_d, ipermutation_d, b_d, x_d, xc_d, level_id, largestblocksize);
         break;
      case 2:
         smoother->postPCRFullCsr(AinCSR_d, Aout_d, AoutBlockIdx_d, aggregateIdx_d, partitionIdx_d, prolongatorFull_d, ipermutation_d, b_d, x_d, xc_d, level_id, largestblocksize, largest_num_entries, largest_num_per_row);
         break;
      default:
         cout << "Wrong DStype 0!" << endl;
         exit(0);

      }

      x_d_CG = x_d;
      b_d_CG = b_d;
   }
}

#include<smoothers/smoother.h>

template <class Matrix, class Vector>
void AMG_Level<Matrix, Vector>::setup()
{
   smoother = Smoother<Matrix_d, Vector_d>::allocate(amg->cfg, A_d);
}

template <class Matrix, class Vector>
std::vector<int> AMG_Level<Matrix, Vector>::getOriginalRows()
{
   return originalRow;
}

/****************************************
 * Explict instantiations
 ***************************************/
template class AMG_Level<Matrix_h, Vector_h>;
template class AMG_Level<Matrix_d, Vector_d>;
