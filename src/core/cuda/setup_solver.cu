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
        sprintf(charBuffer, "CUDA device %d is no available on this system.", cudaDeviceNumber);
		std::string error = std::string(charBuffer);
		throw std::invalid_argument(error);
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
	std::string error = "Error no matrix specified";
	throw std::invalid_argument(error);
  }
}

void getMatrixFromMesh(AMG_Config& cfg, TriMesh* meshPtr, Matrix_d* A, const bool verbose) {

	srand48(0);

	cfg.setParameter<int>("mesh_type", 0);

	if (verbose)
		displayCudaDevices(verbose);
	verifyThatCudaDeviceIsValid(cfg, verbose);

	FEM2D* fem2d = new FEM2D;

	meshPtr->rescale(4.0);
	meshPtr->need_neighbors();
	meshPtr->need_meshquality();

	Matrix_ell_d_CG Aell_d;
	Vector_d_CG RHS(meshPtr->vertices.size(), 0.0);

	trimesh2ell<Matrix_ell_d_CG >(meshPtr, Aell_d);
	cudaThreadSynchronize();

	fem2d->initializeWithTriMesh(meshPtr);
	fem2d->assemble(meshPtr, Aell_d, RHS);
	delete fem2d;

	cudaThreadSynchronize();
	*A = Aell_d;
}

int setup_solver(AMG_Config& cfg, TriMesh* meshPtr, Matrix_d* A,
                 Vector_h_CG* x_d, Vector_h_CG* b_d, const bool verbose) {
  checkMatrixForValidContents(A, verbose);

  if( verbose ) cfg.printAMGConfig();

  AMG<Matrix_h, Vector_h> amg(cfg);

  amg.setup(*A, meshPtr, NULL);

  if( verbose ) amg.printGridStatistics();

  Vector_d_CG my_x = *x_d;
  Vector_d_CG my_b = *b_d;
  amg.solve(my_b, my_x);
  *x_d = my_x;
  *b_d = my_b;

  return 0;
}

void getMatrixFromMesh(AMG_Config& cfg, TetMesh* meshPtr, Matrix_d* A, const bool verbose) {

	srand48(0);

	cfg.setParameter<int>("mesh_type", 1);

	if (verbose) 
		displayCudaDevices(verbose);
	verifyThatCudaDeviceIsValid(cfg, verbose);

	FEM3D* fem3d = new FEM3D;

	meshPtr->need_neighbors();
	meshPtr->need_meshquality();
	meshPtr->rescale(1.0);

	Matrix_ell_d_CG Aell_d;
	Vector_d_CG RHS(meshPtr->vertices.size(), 0.0);

	tetmesh2ell<Matrix_ell_d_CG >(meshPtr, Aell_d);
	cudaThreadSynchronize();

	fem3d->initializeWithTetMesh(meshPtr);
	fem3d->assemble(meshPtr, Aell_d, RHS, true);
	delete fem3d;

	cudaThreadSynchronize();
	*A = Aell_d;
}

int setup_solver(AMG_Config& cfg, TetMesh* meshPtr, Matrix_d* A,
                 Vector_h_CG* x_d, Vector_h_CG* b_d, const bool verbose) {
  checkMatrixForValidContents(A, verbose);

  if( verbose ) cfg.printAMGConfig();

  AMG<Matrix_h, Vector_h> amg(cfg);

  amg.setup(*A, NULL, meshPtr);

  if( verbose ) amg.printGridStatistics();
  
  Vector_d_CG my_x = *x_d;
  Vector_d_CG my_b = *b_d;
  amg.solve(my_b, my_x);
  *x_d = my_x;
  *b_d = my_b;

  return 0;
}
