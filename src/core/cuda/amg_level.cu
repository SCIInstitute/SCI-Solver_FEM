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
  AlgorithmType alg = amg->cfg.AMG_Config::getParameter<AlgorithmType > ("algorithm");
  switch(alg)
  {
    case CLASSICAL:
//    	 return 0;
      return new SmoothedMG_AMG_Level<Matrix, Vector > (amg);
  }
  return 0;
}

/******************************************************
 * Recusively solves the system on this level
 ******************************************************/
template <class Matrix, class Vector>
void AMG_Level<Matrix, Vector>::cycle(CycleType cycle, Vector_d& b_d, Vector_d& x_d)
{
  if(isCoarsest()) //solve directly 
  {
//		 double coarsestart, coarsestop;
//    double coarsetime = 0.0;
//    coarsestart = CLOCK();
    cusp::array1d<ValueType, cusp::host_memory> temp_b(b_d);
    cusp::array1d<ValueType, cusp::host_memory> temp_x(x_d.size());
    amg->LU(temp_b, temp_x);
    x_d = temp_x;
//		coarsestop = CLOCK();
//		coarsetime = coarsestop - coarsestart;
//		cout << "coarsest solve time for level "<< level_id <<" is: " << coarsetime << endl;


//        cout << "level = " << level_id << endl;
//        Vector_h tmpx = b_d;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//    
//        cout << endl;
//        tmpx = x_d;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//    
//        cout << endl;


    return;
  }
  else
  {
//		 Vector_h tmpx;
//		 cout << "level = " << level_id << endl;
//		 tmpx = b_d;
//		 for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//    
//        cout << endl;
//    
//        tmpx = x_d;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//        cout << endl;


    //    smoother->preRRR(AinEll_d, Aout_d, aggregateIdx_d, partitionIdx_d, prolongator_d, b_d, x_d, bc_d);
    //		cusp::print(AinEll_d);
    //		cout << "level = " << level_id << endl;
//    double prestart, prestop;
//    double pretime = 0.0;
//    prestart = CLOCK();
		switch (DS_type)
		{
			case 0:
//				smoother->preRRRFullSymmetricSync(AinSysCoo_d, AoutSys_d, AinBlockIdx_d, aggregateIdx_d, partitionIdx_d, restrictorFull_d, ipermutation_d, b_d, x_d, bc_d, 
//      																				segSyncIdx_d, partSyncIdx_d, level_id, largestblocksize, largest_num_entries);
				smoother->preRRRFullSymmetric(AinSysCoo_d, AoutSys_d, AinBlockIdx_d, AoutBlockIdx_d, aggregateIdx_d, partitionIdx_d, restrictorFull_d, ipermutation_d, b_d, x_d, bc_d,
      																				level_id, largestblocksize, largest_num_entries);
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
//    cudaThreadSynchronize();
//    prestop = CLOCK();
//    pretime = prestop - prestart;
//    if(level_id == 0)
//    {
//            cout << "pre time for level "<< level_id <<" is: " << pretime << endl;
//    }
        
//        tmpx = x_d;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//    
//        cout << endl;
//    
//        tmpx = bc_d;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//        cout << endl;

    next->cycle(V_CYCLE, bc_d, xc_d);
//    cudaThreadSynchronize();
//    double poststart, poststop;
//    double posttime = 0.0;
//    poststart = CLOCK();
		switch (DS_type)
		{
			case 0:
//				smoother->postPCRFullSymmetricSync(AinSysCoo_d, AinBlockIdx_d, AoutSys_d, AoutBlockIdx_d, aggregateIdx_d, partitionIdx_d, prolongatorFull_d, ipermutation_d, b_d, x_d, xc_d,
//                                      segSyncIdx_d, partSyncIdx_d, level_id, largestblocksize, largest_num_entries);
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

//	  cudaThreadSynchronize();	
//    poststop = CLOCK();
//    posttime = poststop - poststart;
//    if(level_id == 0)
//    {
//            cout << "post time for level "<< level_id <<" is: " << posttime << endl;
//    }
//        cout << "level = " << level_id << endl;
//        tmpx = xc_d;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//    
//        cout << endl;
//        //
//        //		
//        tmpx = x_d;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//    
//        cout << endl;
  }
}


template <class Matrix, class Vector>
void AMG_Level<Matrix, Vector>::cycle_level0(CycleType cycle, Vector_d_CG &b_d_CG, Vector_d_CG &x_d_CG)
{
  if(isCoarsest()) //solve directly 
  {
    cusp::array1d<ValueType, cusp::host_memory> temp_b = b_d_CG;
    cusp::array1d<ValueType, cusp::host_memory> temp_x(x_d_CG.size());
    amg->LU(temp_b, temp_x);
    x_d_CG = temp_x;


//        cout << "level = " << level_id << endl;
//        Vector_h tmpx = b_d_CG;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//    
//        cout << endl;
//        tmpx = x_d_CG;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//    
//        cout << endl;


    return;
  }
  else
  {
//		 Vector_h tmpx;
//		 cout << "level = " << level_id << endl;
//		 tmpx = b_d_CG;
//		 for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//    
//        cout << endl;
//    
//        tmpx = x_d_CG;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//        cout << endl;


    //    smoother->preRRR(AinEll_d, Aout_d, aggregateIdx_d, partitionIdx_d, prolongator_d, b_d, x_d, bc_d);
    //		cusp::print(AinEll_d);
    //		cout << "level = " << level_id << endl;
		Vector_d b_d = b_d_CG;
		Vector_d x_d(x_d_CG.size(), 0.0);
//    double prestart, prestop;
//    double pretime = 0.0;
//    prestart = CLOCK();
		switch (DS_type)
		{
			case 0:
//				smoother->preRRRFullSymmetricSync(AinSysCoo_d, AoutSys_d, AinBlockIdx_d, aggregateIdx_d, partitionIdx_d, restrictorFull_d, ipermutation_d, b_d, x_d, bc_d, 
//      																				segSyncIdx_d, partSyncIdx_d, level_id, largestblocksize, largest_num_entries);
				smoother->preRRRFullSymmetric(AinSysCoo_d, AoutSys_d, AinBlockIdx_d, AoutBlockIdx_d, aggregateIdx_d, partitionIdx_d, restrictorFull_d, ipermutation_d, b_d, x_d, bc_d,
      																				level_id, largestblocksize, largest_num_entries);
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
//    cudaThreadSynchronize();
//    prestop = CLOCK();
//    pretime = prestop - prestart;
//    if(level_id == 0)
//    {
//            cout << "pre time for level "<<level_id << " is: " << pretime << endl;
//    }
        
//        tmpx = x_d;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//    
//        cout << endl;
//    
//        tmpx = bc_d;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//        cout << endl;

    next->cycle(V_CYCLE, bc_d, xc_d);
//    cudaThreadSynchronize();
//    double poststart, poststop;
//    double posttime = 0.0;
//    poststart = CLOCK();
		switch (DS_type)
		{
			case 0:
//				smoother->postPCRFullSymmetricSync(AinSysCoo_d, AinBlockIdx_d, AoutSys_d, AoutBlockIdx_d, aggregateIdx_d, partitionIdx_d, prolongatorFull_d, ipermutation_d, b_d, x_d, xc_d,
//                                     segSyncIdx_d, partSyncIdx_d, level_id, largestblocksize, largest_num_entries);
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

//	cudaThreadSynchronize();	
//    poststop = CLOCK();
//    posttime = poststop - poststart;
//    if(level_id == 0)
//    {
//            cout << "post time for level " << level_id <<" is: " << posttime << endl;
//    }
//        cout << "level = " << level_id << endl;
//        tmpx = xc_d;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//    
//        cout << endl;
//        //
//        //		
//        tmpx = x_d;
//        for(int i = 0; i < 10; i++)
//          cout << tmpx[i] << " ";
//    
//        cout << endl;

		x_d_CG = x_d;
		b_d_CG = b_d;
  }
}

#include<smoothers/smoother.h>

template <class Matrix, class Vector>
void AMG_Level<Matrix, Vector>::setup()
{
  smoother = Smoother<Matrix_d, Vector_d>::allocate(amg->cfg, A_d);
//	if(level_id ==3)
//	{
//		Vector_h tmpv = smoother->diag;
//		for(int i =0; i<tmpv.size(); i++)
//			printf("%.15f\n", tmpv[i]);
//	}
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
