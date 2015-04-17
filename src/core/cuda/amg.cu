#include <cusp/csr_matrix.h>

#include <amg.h>
#include <types.h>
#include <cutil.h>
#include<iostream>
#include<iomanip>

#include <amg_level.h>
#include <cycles/cgcycle.h>
#include <allocator.h>
#include <my_timer.h>

template<class Matrix, class Vector>
AMG<Matrix, Vector>::AMG(AMG_Config cfg) : fine(0), cfg(cfg)
{
   cycle_iters = cfg.getParameter<int>("cycle_iters");
   max_iters = cfg.getParameter<int>("max_iters");
   presweeps = cfg.getParameter<int>("presweeps");
   postsweeps = cfg.getParameter<int>("postsweeps");
   tolerance = cfg.getParameter<double> ("tolerance");
   cycle = cfg.getParameter<CycleType > ("cycle");
   //  norm = cfg.getParameter<NormType > ("norm");
   convergence = cfg.getParameter<ConvergenceType > ("convergence");
   solver = cfg.getParameter<SolverType > ("solver");
   DS_type = cfg.getParameter<int>("DS_type");
}

template<class Matrix, class Vector>
AMG<Matrix, Vector>::~AMG()
{
   //  delete fine;
}

/**********************************************************
 * Returns true of the solver has converged
 *********************************************************/
template<class Matrix, class Vector>
bool AMG<Matrix, Vector>::converged(const Vector &r, ValueType &nrm)
{
   //  nrm = get_norm(r, norm);

   nrm = cusp::blas::nrm2(r);
   if(convergence == ABSOLUTE_CONVERGENCE)
   {
      return nrm <= tolerance;
   }
   else //if (convergence==RELATIVE)
   {
      if(initial_nrm == -1)
      {
         initial_nrm = nrm;
         return false;
      }
      //if the norm has been reduced by the tolerance then return true
      if(nrm / initial_nrm <= tolerance)
         return true;
      else
         return false;
   }
}

/**********************************************************
 * Creates the AMG hierarchy
 *********************************************************/
template <class Matrix, class Vector>
void AMG<Matrix, Vector>::setup(const Matrix_d &Acsr_d, TriMesh* meshPtr, TetMesh* tetmeshPtr, bool verbose)
{

   int min_rows = cfg.getParameter<int>("min_rows");
   int max_levels = cfg.getParameter<int>("max_levels");
   int mesh_type = cfg.getParameter<int>("mesh_type");

   num_levels = 1;


   //allocate the fine level
   AMG_Level<Matrix, Vector>* level = AMG_Level<Matrix, Vector>::allocate(this);
   //set the fine level pointer
   fine = level;
   level->A_d = Acsr_d;
   //  Ahyb_d_CG = level->A_d;
   level->level_id = 0;
   level->nn = Acsr_d.num_rows;

   if(mesh_type == 0)
      level->m_meshPtr = meshPtr;
   else
      level->m_tetmeshPtr = tetmeshPtr;

   int topsize = cfg.getParameter<int>("top_size");

   while(true)
   {
      int N = level->A_d.num_rows;
      if(N < topsize || num_levels >= max_levels)
      {
         coarsestlevel = num_levels - 1;
         Matrix_h Atmp = level->A_d;
         cusp::array2d<ValueType, cusp::host_memory> coarse_dense(Atmp);
         LU = cusp::detail::lu_solver<ValueType, cusp::host_memory > (coarse_dense);
         break;
      }

      level->next = AMG_Level<Matrix, Vector>::allocate(this);
      level->createNextLevel(verbose);

      if(level->level_id == 0)
      {
         Ahyb_d_CG = level->A_d;
      }

      level->setup(); //allocate smoother !! must be after createNextLevel since A_d is used

      level->next->level_id = num_levels;
      level->next->nn = level->nnout;
      level->next->m_xadj_d = level->m_xadjout_d;
      level->next->m_adjncy_d = level->m_adjncyout_d;
      int nextN = level->next->A_d.num_rows;

      //resize vectors
      level->xc_d = Vector_d(nextN, -1);
      level->bc_d = Vector_d(nextN, -1);

      //advance to the next level
      level = level->next;

      //increment the level counter
      num_levels++;
   }

}

/***************************************************
 * Launches a single iteration of the outer solver
 ***************************************************/
template <class Matrix, class Vector>
void AMG<Matrix, Vector>::solve_iteration(const Vector_d_CG &b, Vector_d_CG &x, bool verbose)
{
   Vector_d b_d(b);
   Vector_d x_d(x);
   switch(solver)
   {
   case AMG_SOLVER:
      //perform a single cycle on the amg hierarchy
      fine->cycle(cycle, b_d, x_d, verbose);
      x = Vector_d_CG(x_d);
      break;
   case PCG_SOLVER:
      //create a single CG cycle (this will run CG immediatly)
      CG_Flex_Cycle<Matrix_h_CG, Vector_h_CG > (cycle, cycle_iters, fine, Ahyb_d_CG, b, x, tolerance, max_iters, verbose); //DHL
      break;
   }
}


/**********************************************************
 * Solves the AMG system
 *********************************************************/
template <class Matrix, class Vector>
void AMG<Matrix, Vector>::solve(const Vector_d_CG &b_d, Vector_d_CG &x_d, bool verbose)
{
   if (verbose)
      printf("AMG Solve:\n");
   iterations = 0;
   initial_nrm = -1;

   if (verbose) {
      std::cout << std::setw(15) << "iter" << std::setw(15) << "time(s)" << std::setw(15) << "residual" << std::setw(15) << "rate" << std::setw(15) << std::endl;
      std::cout << "         ----------------------------------------------------\n";
   }
   solve_start = CLOCK();
   bool done = false;
   do
   {
      //launch a single solve iteration
      solve_iteration(b_d, x_d, verbose);

      done = true; //converged(r_d, nrm);
   }
   while(++iterations < max_iters && !done);
   if (verbose)
      std::cout << "         ----------------------------------------------------\n";

   solve_stop = CLOCK();
   Allocator<Vector>::clear(); // DHL
}

template <class Matrix, class Vector>
void AMG<Matrix, Vector>::printGridStatistics()
{
   int total_rows = 0;
   int total_nnz = 0;
   AMG_Level<Matrix, Vector> *level = fine; // DHL
   std::cout << "AMG Grid:\n";
   std::cout << std::setw(15) << "LVL" << std::setw(10) << "ROWS" << std::setw(18) << "NNZ" << std::setw(10) << "SPRSTY" << std::endl;
   std::cout << "         ---------------------------------------------\n";

   level = fine;
   while(level != NULL)
   {
      total_rows += level->A_d.num_rows;
      total_nnz += level->A_d.num_entries;
      std::cout << std::setw(15) << level->level_id << std::setw(10) << level->A_d.num_rows << std::setw(18) << level->A_d.num_entries << std::setw(10) << std::setprecision(3) << level->A_d.num_entries / (double)(level->A_d.num_rows * level->A_d.num_cols) << std::setprecision(6) << std::endl;

      level = level->next;
   }
   // DHL
   std::cout << "         ---------------------------------------------\n";
   std::cout << "     Grid Complexity: " << total_rows / (double)fine->A_d.num_rows << std::endl;
   std::cout << "     Operator Complexity: " << total_nnz / (double)fine->A_d.num_entries << std::endl;

}

using std::scientific;
using std::fixed;

// print a line of length l, starting at character s

void printLine(const int l, const int s)
{
   cout << setw(s) << " ";
   for(int i = 0; i < l; i++)
   {
      cout << "-";
   }
   cout << endl;
}

template <class Matrix, class Vector>
void AMG<Matrix, Vector>::printProfile()
{
#ifdef PROFILE
   // print headers from first AMG level
   std::vector<const char *> headers = fine[0].Profile.getHeaders();
   std::vector<double> levelTimes;
   std::vector<std::vector<double> > globalTimes;
   cout << "\n" << setw(7) << "Level";

   typedef std::vector<const char *>::iterator headerIter;
   for(headerIter it = headers.begin(); it != headers.end(); ++it)
   {
      cout << setw(max((int)strlen(*it) + 1, 18)) << *it;
      // centerString(*it,16);
   }
   cout << setw(12) << "Total" << endl;
   // now print the sub titles
   cout << setw(7) << " ";
   for(headerIter it = headers.begin(); it != headers.end(); ++it)
   {
      cout << setw(6) << "t" << setw(6) << "l%" << setw(6) << "g%";
   }
   cout << setw(6) << "l" << setw(6) << "g%" << endl;

   // print a line across
   printLine(108, 2);

   AMG_Level<Matrix, Vector> *level = fine;
   while(level != NULL)
   {
      levelTimes = level->Profile.getTimes();
      globalTimes.push_back(levelTimes);
      // cout << setw(4) << level->level_id;
      //level->Profile.writeTimes();
      //cout << endl;
      level = level->next;
   }

   // now we have all of the times for all levels, work on them
   // get the global total time
   double tTotal = 0.0;
   double *levelTotals = new double[globalTimes.size()];
   for(int i = 0; i < globalTimes.size(); i++)
   {
      levelTotals[i] = 0.0;
   }

   // get both total (global) time and level totals
   for(int i = 0; i < globalTimes.size(); i++)
   {
      for(int j = 0; j < globalTimes[i].size(); j++)
      {
         tTotal += globalTimes[i][j];
         levelTotals[i] += globalTimes[i][j];
      }
   }

   // only ever print out 2 decimal places
   cout.precision(2);

   // loop over each level & print stats
   level = fine;
   while(level != NULL)
   {
      int level_id = level->level_id;
      cout << setw(7) << level_id;

      for(int i = 0; i < globalTimes[level_id].size(); i++)
      {
         double t = globalTimes[level_id][i];
         double levelPercent = t / levelTotals[level_id] * 100;
         double globalPercent = t / tTotal * 100;

         cout << scientific << setw(6) << fixed << t << setw(6) << levelPercent << setw(6) << globalPercent;
      }
      // totals here
      cout << setw(6) << fixed << levelTotals[level_id] << setw(6) << levelTotals[level_id] / tTotal * 100;
      cout << endl;

      // next level
      level = level->next;
   }

   // print final line across
   printLine(108, 2);
#endif
}

template <class Matrix, class Vector>
void AMG<Matrix, Vector>::printCoarsePoints()
{
#ifdef DEBUG
   typedef std::vector<int> iVec;
   typedef std::vector<int>::iterator iVecIter;

   ofstream coarsePoints("coarse_points.dat");

   iVec originalRows;

   AMG_Level<Matrix, Vector> *level = fine;
   while(level != NULL)
   {
      originalRows = level->getOriginalRows();

      level = level->next;
      if(level == NULL)
      {
         break;
      }

      coarsePoints << level->level_id << " " << level->getNumRows() << endl;

      for(iVecIter it = originalRows.begin(); it != originalRows.end(); ++it)
      {
         coarsePoints << *it << endl;
      }
   }
   coarsePoints.close();
#endif
}

template <class Matrix, class Vector>
void AMG<Matrix, Vector>::printConnections()
{
#ifdef DEBUG
   ofstream connFile("connections.dat");
   AMG_Level<Matrix, Vector> *level = fine;

   Matrix ATemp;
   while(level != NULL)
   {
      connFile << level->level_id << " " << level->getNumRows() << endl;

      ATemp = level->getA();

      for(int i = 0; i < ATemp.num_rows; i++)
      {
         // get the row offset & num rows
         int offset = ATemp.row_offsets[i];
         int numEntries = ATemp.row_offsets[i + 1] - offset;

         // # of connections is numEntries - 1 (ignoring diagonal)
         // this->numConnections.push_back(numEntries-1);
         connFile << numEntries - 1 << " ";

         // loop over non-zeros and add non-diagonal terms
         for(int j = offset; j < offset + numEntries; j++)
         {
            int columnIndex = ATemp.column_indices[j];
            if(i != columnIndex)
            {
               // this->connections.push_back(columnIndex);
               connFile << columnIndex << " ";
            }
         }
         connFile << endl;
      }
      level = level->next;
   }
#endif
}

/****************************************
 * Explict instantiations
 ***************************************/
template class AMG<Matrix_h, Vector_h>;
//template class AMG<Matrix_h_CG, Vector_h_CG>;
//template class AMG<Matrix_d,Vector_d>;

