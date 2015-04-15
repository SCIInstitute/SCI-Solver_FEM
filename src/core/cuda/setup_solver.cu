#include <stdio.h>
#include <iostream>
#include <signal.h>
#include <exception>
#include <fstream>
#include <amg_config.h>
#include <types.h>
#include <TriMesh.h>
#include <tetmesh.h>
#include <cutil.h>
#include <FEM/FEM2D.h>
#include <FEM/FEM3D.h>
#include <my_timer.h>
#include <amg.h>
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

bool compare_sparse_entry(SparseEntry_t a, SparseEntry_t b) {
  return ((a.row_ != b.row_) ? (a.row_ < b.row_) : (a.col_ < b.col_));
}

int readMatlabFile(std::string file,
    cusp::ell_matrix<int,float,cusp::host_memory> *mat) {
  //read in the description header
  std::ifstream in(file.c_str());
  char buffer[256];
  in.read(buffer,128);
  int32_t type;
  in.read((char*)&type,4);
  if (type == 15) {
    std::cerr << "Compression not supported. Save matlab data with '-v6' option." << std::endl;
    in.close();
    return 1;
  } else if (type != 14) {
    std::cerr << file << " is not a matlab matrix." << std::endl;
    in.close();
    return 1;
  }
  //read in the array flags
  uint32_t data_size;
  in.read((char*)&data_size,4);
  in.read((char*)&type,4);
  if (type != 6 && type != 5) {
    std::cerr << "Invalid type for sparse matrix. Must be 32bit." << std::endl;
    in.close();
    return 1;
  }
  int32_t byte_per_element;
  in.read((char*)&byte_per_element,4);
  uint32_t mclass;
  in.read((char*)&mclass,4);
  mclass &= 0x000000FF;
  if (mclass != 5) {
    std::cerr << "This is not a sparse matrix file." << std::endl;
    in.close();
    return 1;
  }
  uint32_t nzmax;
  in.read((char*)&nzmax,4);
  //read in the dimensions and name
  in.read((char*)&type,4);
  in.read((char*)&byte_per_element,4);
  if ((type != 6 && type != 5) || byte_per_element != 8) {
    std::cerr << "Matrix of wrong dimension type or # of dimensions." << std::endl;
    std::cerr << "Matrix must be 2 dimensions and of 32bit type." << std::endl;
    in.close();
    return 1;
  }
  int32_t x_dim,y_dim;
  in.read((char*)&x_dim,4);
  in.read((char*)&y_dim,4);
  in.read(buffer,8);
  //read in the row indices
  in.read((char*)&type,4);
  if (type != 6 && type != 5) {
    std::cerr << "Invalid type row index for sparse matrix. Must be 32bit." << std::endl;
    in.close();
    return 1;
  }
  in.read((char*)&byte_per_element,4);
  int32_t num_rows = byte_per_element / 4;
  std::vector<int32_t> row_vals;
  for(int i = 0; i < num_rows; i++) {
    int32_t val;
    in.read((char*)&val,4);
    row_vals.push_back(val);
  }
  in.read(buffer,byte_per_element % 8);
  //read in the column indices
  in.read((char*)&type,4);
  if (type != 6 && type != 5) {
    std::cerr << "Invalid column index type for sparse matrix. Must be 32bit." << std::endl;
    in.close();
    return 1;
  }
  in.read((char*)&byte_per_element,4);
  int32_t num_cols = byte_per_element / 4;
  std::vector<int32_t> col_vals;
  for(int i = 0; i < num_cols; i++) {
    int32_t val;
    in.read((char*)&val,4);
    col_vals.push_back(val);
  }
  in.read(buffer,byte_per_element % 8);
  //read in the data values
  in.read((char*)&type,4);
  if (type != 9) {
    std::cerr << "Invalid value for sparse matrix. " <<
      "Must be double float." << std::endl;
    in.close();
    return 1;
  }
  in.read((char*)&byte_per_element,4);
  int32_t val_count = byte_per_element / 8;
  std::vector<float> double_vals;
  for(int i = 0; i < val_count; i++) {
    double double_val;
    in.read((char*)&double_val,8);
    double_vals.push_back(double_val);
  }
  std::vector<SparseEntry_t> sparse_entries;
  int32_t num_entries = col_vals[y_dim];
  sparse_entries.reserve(num_entries);
  std::vector<int32_t> row_max;
  row_max.resize(x_dim);
  for(size_t i = 0; i < row_max.size(); i++)
    row_max[i] = 0;
  in.read(buffer,byte_per_element % 8);
  for(size_t i = 0; i < y_dim; i++) {
    int32_t idx = col_vals[i];
    int32_t idx_end = col_vals[i + 1] - 1;
    int32_t col = static_cast<int32_t>(i);
    for (int32_t j = idx; j <= idx_end; j++) {
      row_max[row_vals[j]]++;
      sparse_entries.push_back(
          SparseEntry_t(row_vals[j],col,
            static_cast<float>(double_vals[j])));
    }
  }
  //TODO now set up the ell matrix.
  //sort the sparse entries
  std::sort(sparse_entries.begin(),sparse_entries.end(),compare_sparse_entry);
  //determine the max nonzeros per row
  int32_t max_row = 0;
  for(size_t i = 0; i < row_max.size(); i++)
    max_row = std::max(max_row,row_max[i]);
  //set up the matrix
  mat->resize(x_dim,y_dim,num_entries,max_row);
  //iterate through to add values.
  // X is used to fill unused entries in the matrix
  const int bad_entry = cusp::ell_matrix<int,float,cusp::host_memory>::invalid_index;
  int32_t current_row = 0, row_count = 0;
  for(size_t i = 0; i < sparse_entries.size(); i++) {
    if (current_row != sparse_entries[i].row_) {
      while(row_count < max_row) {
        mat->column_indices(current_row,row_count) = bad_entry;
        mat->values(current_row,row_count) = 0;
        row_count++;
      }
      current_row = sparse_entries[i].row_;
      row_count = 0;
    }
    mat->column_indices(current_row,row_count) = sparse_entries[i].col_;
    mat->values(current_row,row_count) = sparse_entries[i].val_;
    row_count++;
  }
  in.close();
  return 0;
}
