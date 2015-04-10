#include <stdio.h>
#include <iostream>
#include <signal.h>
#include <exception>

#include <amg_config.h>
#include <types.h>
#include <TriMesh.h>
#include <tetmesh.h>
#include <cutil.h>
#include <FEM/FEM2D.h>
#include <FEM/FEM3D.h>
#include <my_timer.h>
#include <amg.h>
#include <string>
#include <typeinfo>
#include <setup_solver.h>

#ifdef WIN32
#include <cstdlib>
#define srand48 srand
#endif

using namespace std;

void displayCudaDevices (bool verbose)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        size_t totalMemory = deviceProp.totalGlobalMem;
        int totalMB = totalMemory / 1000000;
        printf("Device %d (%s) has compute capability %d.%d, %d regs per block, and %dMb global memory.\n",
                device, deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.regsPerBlock, totalMB);
    }
}

void verifyThatCudaDeviceIsValid(AMG_Config& cfg, bool verbose)
{
    int cudaDeviceNumber = cfg.getParameter<int>("cuda_device_num");
    char charBuffer[100];
    cudaSetDevice(cudaDeviceNumber);
    if (cudaDeviceReset() != cudaSuccess) {
        //exit(0);
        sprintf(charBuffer, "CUDA device %d is no available on this system.", cudaDeviceNumber);
     	string error = string(charBuffer);
       	throw invalid_argument(error);
    } else if( verbose ) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 1);
    }
}

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

template <>
int setup_solver<TriMesh>(AMG_Config& cfg, TriMesh* meshPtr, Matrix_d* A,
                 Vector_d_CG* x_d, Vector_d_CG* b_d, const bool verbose)
{
  srand48(0);
  ostringstream strCout;

  cfg.setParameter<int>("mesh_type", 0);

  if( verbose ) {
    displayCudaDevices(verbose);
  }

  verifyThatCudaDeviceIsValid(cfg, verbose);

  //double Assemblestart, Assemblestop;
  //double neednbstart, neednbstop;
  //double prepAssemstart, prepAssemstop;

  FEM2D* fem2d = new FEM2D;

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

  checkMatrixForValidContents(A, verbose);

  Vector_h_CG b(A->num_rows, 1.0);
  Vector_h_CG x(A->num_rows, 0.0); //initial
  *x_d = x;
  *b_d = b;

  if( verbose ) {
    cfg.printAMGConfig();
  }
  AMG<Matrix_h, Vector_h> amg(cfg);
  amg.setup(*A, meshPtr, NULL);
  if( verbose ) {
    amg.printGridStatistics();
  }
  amg.solve(*b_d, *x_d);
  if( !verbose ) {
    cout << strCout.str()<< endl;
  }
  delete fem2d;
  return 0;
}

template <>
int setup_solver<TetMesh>(AMG_Config& cfg, TetMesh* meshPtr, Matrix_d* A,
                 Vector_d_CG* x_d, Vector_d_CG* b_d, const bool verbose)
{
  srand48(0);
  ostringstream strCout;

  cfg.setParameter<int>("mesh_type", 1);

  if( verbose ) {
    displayCudaDevices(verbose);
  }

  verifyThatCudaDeviceIsValid(cfg, verbose);

  //double Assemblestart, Assemblestop;
  //double neednbstart, neednbstop;
  //double prepAssemstart, prepAssemstop;

  FEM3D* fem3d = new FEM3D;

  //neednbstart = CLOCK();
  meshPtr->need_neighbors();
  meshPtr->need_meshquality();
  meshPtr->rescale(1.0);
  //neednbstop = CLOCK();

  Matrix_ell_d_CG Aell_d;
  Vector_d_CG RHS(meshPtr->vertices.size(), 0.0);

  //prepAssemstart = CLOCK();
  tetmesh2ell<Matrix_ell_d_CG > (meshPtr, Aell_d);
  cudaThreadSynchronize();
  //prepAssemstop = CLOCK();

  //Assemblestart = CLOCK();
  fem3d->initializeWithTetMesh(meshPtr);
  fem3d->assemble(meshPtr, Aell_d, RHS, true);

  cudaThreadSynchronize();
  //Assemblestop = CLOCK();
  //            cusp::print(Aell_d);
  *A = Aell_d;
  Aell_d.resize(0, 0, 0, 0);

  checkMatrixForValidContents(A, verbose);

  Vector_h_CG b(A->num_rows, 1.0);
  Vector_h_CG x(A->num_rows, 0.0); //initial
  *x_d = x;
  *b_d = b;

  if( verbose ) {
    cfg.printAMGConfig();
  }
  AMG<Matrix_h, Vector_h> amg(cfg);

  amg.setup(*A, NULL, meshPtr);

  if( verbose ) {
    amg.printGridStatistics();
  }
  amg.solve(*b_d, *x_d);
  if( !verbose ) {
    cout << strCout.str()<< endl;
  }

  delete fem3d;
  return 0;
}
