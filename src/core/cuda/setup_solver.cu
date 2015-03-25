#include <stdio.h>
#include <iostream>
#include <pthread.h>
#include <signal.h>
#include <exception>

#include <amg_config.h>
#include <types.h>
#include <TriMesh.h>
#include <tetmesh.h>
#include <cutil.h>
#include <FEM/FEM2D.h>
#include <FEM/FEM3D.h>
#include <timer.h>
#include <amg.h>
#include <string>

/*

#include <cusp/io/matrix_market.h>
#include <cusp/print.h>
#include <cusp/gallery/poisson.h>
#include <fstream>


 */

using namespace std;

void checkMatrixForValidContents(Matrix_d* A, const bool verbose)
{
  if (A->num_rows == 0) {
    if( verbose ) {
      printf("Error no matrix specified\n");
    }
    string error = "Error no matrix specified";
    throw invalid_argument(error);
  }
}

int setup_solver(AMG_Config& cfg, TriMesh* meshPtr, TetMesh* tetmeshPtr,
    FEM2D* fem2d, FEM3D* fem3d, Matrix_d* A,
    Vector_d_CG* x_d, Vector_d_CG* b_d, const bool verbose)
{
  srand48(0);
  char charBuffer[100];
  ostringstream strCout;

  int cudaDeviceNumber = cfg.getParameter<int>("cuda_device_num");
  cudaSetDevice(cudaDeviceNumber);
  if (cudaDeviceReset() != cudaSuccess) {
    sprintf(charBuffer, "CUDA device %d is no available on this system.", cudaDeviceNumber);
    exit(1);
  } else if( verbose ) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cudaDeviceNumber);
    size_t totalMemory = deviceProp.totalGlobalMem;
    int totalMB = totalMemory / 1000000;
    std::cout << "Device " << cudaDeviceNumber << " (" <<
      deviceProp.name << ") has:\n\tCompute capability: " <<
      deviceProp.major << "." << deviceProp.minor << "\n\t" <<
      deviceProp.regsPerBlock << "registers per block\n\t" <<
      totalMB << " Mb global memory" << std::endl;
  }

  //checkMatrixForValidContents(A, verbose);

  //double Assemblestart, Assemblestop;
  //double neednbstart, neednbstop;
  //double prepAssemstart, prepAssemstop;

  int meshType = cfg.getParameter<int>("mesh_type");
  if( meshType == 0 ) {
    //Triangular mesh
    meshPtr->rescale(4.0);
    //neednbstart = CLOCK();
    meshPtr->need_neighbors();
    meshPtr->need_meshquality();
    //neednbstop = CLOCK();
    Matrix_ell_d_CG Aell_d;
    Vector_d_CG RHS(meshPtr->vertices.size(), 0.0);

   // prepAssemstart = CLOCK();
    trimesh2ell<Matrix_ell_d_CG > (meshPtr, Aell_d);
    cudaThreadSynchronize();

    //prepAssemstop = CLOCK();
    //Assemblestart = CLOCK();
    fem2d->initializeWithTriMesh(meshPtr);

    fem2d->assemble(meshPtr, Aell_d, RHS);

    cudaThreadSynchronize();
    //Assemblestop = CLOCK();

    *A = Aell_d;
    Aell_d.resize(0, 0, 0, 0);
  } else if( meshType == 1 ) {
    //Tet mesh
    tetmeshPtr->need_neighbors();
    tetmeshPtr->need_meshquality();
    tetmeshPtr->rescale(1.0);

    Matrix_ell_d_CG Aell_d;
    Matrix_ell_h_CG Aell_h;
    Vector_d_CG RHS(tetmeshPtr->vertices.size(), 0.0);

    //prepAssemstart = CLOCK();
    tetmesh2ell<Matrix_ell_d_CG > (tetmeshPtr, Aell_d);
    cudaThreadSynchronize();
    //prepAssemstop = CLOCK();

    fem3d->initializeWithTetMesh(tetmeshPtr);
    //Assemblestart = CLOCK();
    fem3d->assemble(tetmeshPtr, Aell_d, RHS, true);
    cudaThreadSynchronize();
    //Assemblestop = CLOCK();
    //            cusp::print(Aell_d);
    *A = Aell_d;
    Aell_d.resize(0, 0, 0, 0);
  }

  checkMatrixForValidContents(A, verbose);

  Vector_h_CG b(A->num_rows, 1.0);
  Vector_h_CG x(A->num_rows, 0.0); //initial
  *x_d = x;
  *b_d = b;

  if( verbose ) {
    cfg.printAMGConfig();
  }
  AMG<Matrix_h, Vector_h> amg(cfg);
  amg.setup(*A, meshPtr, tetmeshPtr);
  if( verbose ) {
    amg.printGridStatistics();
  }
  amg.solve(*b_d, *x_d);
  if( !verbose ) {
    cout << strCout.str()<< endl;
  }
  return 0;
}
