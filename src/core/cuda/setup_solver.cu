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
    cudaGetDeviceProperties(&deviceProp, cudaDeviceNumber);
  }
}

void checkMatrixForValidContents(Matrix_ell_h* A_h, const bool verbose)
{
  if (A_h->num_rows == 0) {
    if( verbose ) {
      printf("Error no matrix specified\n");
    }
    std::string error = "Error no matrix specified";
    throw std::invalid_argument(error);
  }
}

void getMatrixFromMesh(AMG_Config& cfg, TriMesh* meshPtr, Matrix_ell_h* A_h, const bool verbose) {
  srand48(0);
  meshPtr->set_verbose(verbose);
  //type is triangle mesh
  cfg.setParameter<int>("mesh_type", 0);
  // print the device info
  if (verbose) displayCudaDevices(verbose);
  verifyThatCudaDeviceIsValid(cfg, verbose);
  // 2D fem solving object
  FEM2D* fem2d = new FEM2D;
  meshPtr->rescale(4.0);
  meshPtr->need_neighbors();
  meshPtr->need_meshquality();
  // create the initial A and b
  Matrix_ell_d_CG Aell_d;
  Vector_d_CG RHS(meshPtr->vertices.size(), 0.0);
  //generate the unit constant mesh stiffness matrix
  trimesh2ell<Matrix_ell_d_CG >(meshPtr, Aell_d);
  cudaThreadSynchronize();
  //assembly step
  fem2d->initializeWithTriMesh(meshPtr);
  fem2d->assemble(meshPtr, Aell_d, RHS);
  delete fem2d;
  //copy back to the host
  cudaThreadSynchronize();
  *A_h = Matrix_ell_h(Aell_d);
}

int setup_solver(AMG_Config& cfg, TriMesh* meshPtr, Matrix_ell_h* A_h,
    Vector_h_CG* x_h, Vector_h_CG* b_h, const bool verbose) {
  checkMatrixForValidContents(A_h, verbose);
  //print info
  if( verbose ) cfg.printAMGConfig();
  //register configuration parameters
  AMG<Matrix_h, Vector_h> amg(cfg);
  //copy to device
  Matrix_d A_d(*A_h);
  //setup device
  amg.setup(A_d, meshPtr, NULL,verbose);
  //print info
  if( verbose ) amg.printGridStatistics();
  //copy to device
  Vector_d_CG x_d(*x_h);
  Vector_d_CG b_d(*b_h);
  //run solver
  amg.solve(b_d, x_d, verbose);
  //copy back to host
  *x_h = Vector_h_CG(x_d);
  *b_h = Vector_h_CG(b_d);
  return 0;
}

void getMatrixFromMesh(AMG_Config& cfg, TetMesh* meshPtr, Matrix_ell_h* A_h, const bool verbose) {
  srand48(0);
  meshPtr->set_verbose(verbose);
  //type is tet mesh
  cfg.setParameter<int>("mesh_type", 1);
  // print the device info
  if (verbose) displayCudaDevices(verbose);
  verifyThatCudaDeviceIsValid(cfg, verbose);
  // 3D fem solving object
  FEM3D* fem3d = new FEM3D;
  meshPtr->need_neighbors();
  meshPtr->need_meshquality();
  meshPtr->rescale(1.0);
  // create the initial A and b
  Matrix_ell_d_CG Aell_d;
  Vector_d_CG RHS(meshPtr->vertices.size(), 0.0);
  //generate the unit constant mesh stiffness matrix
  tetmesh2ell<Matrix_ell_d_CG >(meshPtr, Aell_d);
  cudaThreadSynchronize();
  //assembly step
  fem3d->initializeWithTetMesh(meshPtr);
  fem3d->assemble(meshPtr, Aell_d, RHS, true);
  delete fem3d;
  //copy back to the host
  cudaThreadSynchronize();
  *A_h = Matrix_ell_h(Aell_d);
}

int setup_solver(AMG_Config& cfg, TetMesh* meshPtr, Matrix_ell_h* A_h,
    Vector_h_CG* x_h, Vector_h_CG* b_h, const bool verbose) {
  checkMatrixForValidContents(A_h, verbose);
  //print info
  if( verbose ) cfg.printAMGConfig();
  //register configuration properties
  AMG<Matrix_h, Vector_h> amg(cfg);
  //copy to device
  Matrix_d A_d(*A_h);
  //setup device
  amg.setup(A_d, NULL, meshPtr, verbose);
  //print info
  if (verbose) amg.printGridStatistics();
  //copy to device
  Vector_d_CG x_d(*x_h);
  Vector_d_CG b_d(*b_h);
  //run solver
  amg.solve(b_d, x_d, verbose);
  //copy back to host
  *x_h = Vector_h_CG(x_d);
  *b_h = Vector_h_CG(b_d);
  return 0;
}

bool compare_sparse_entry(SparseEntry_t a, SparseEntry_t b) {
  return ((a.row_ != b.row_) ? (a.row_ < b.row_) : (a.col_ < b.col_));
}

int readMatlabSparseMatrix(const std::string &filename, Matrix_ell_h *A_h) {
  //read in the description header
  std::ifstream in(filename.c_str());
  if (!in.is_open()) {
    std::cerr << "could not open file: " << filename << std::endl;
    return 1;
  }
  char buffer[256];
  in.read(buffer,128);
  int32_t type;
  in.read((char*)&type,4);
  if (type == 15) {
    std::cerr << "Compression not supported. Save matlab data with '-v6' option." << std::endl;
    in.close();
    return 1;
  } else if (type != 14) {
    std::cerr << filename << " is not a matlab matrix." << std::endl;
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

  //Array name
  uint32_t arrayName_type;
  in.read((char*)&arrayName_type, 4);
  if (arrayName_type != 1 && arrayName_type != 2) {
    std::cerr << "Invalid variable type for array name characters (Must be 8-bit)." << std::endl;
    in.close();
    return -1;
  }
  uint32_t arrayName_length;
  in.read((char*)&arrayName_length, 4);
  //Account for padding of array name to match 64-bit requirement
  int lenRemainder = arrayName_length % 8;
  if( lenRemainder != 0 )
	  arrayName_length = arrayName_length + 8 - lenRemainder;
  in.read(buffer,arrayName_length); //Read the array name (ignore)

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
  //now set up the ell matrix.
  //sort the sparse entries
  std::sort(sparse_entries.begin(),sparse_entries.end(),compare_sparse_entry);
  //determine the max nonzeros per row
  int32_t max_row = 0;
  for(size_t i = 0; i < row_max.size(); i++)
    max_row = std::max(max_row,row_max[i]);
  //set up the matrix
  Matrix_ell_h A(x_dim, y_dim, num_entries, max_row);
  //iterate through to add values.
  // X is used to fill unused entries in the matrix
  const int bad_entry = Matrix_ell_h::invalid_index;
  int32_t current_row = 0, row_count = 0;
  for(size_t i = 0; i < sparse_entries.size(); i++) {
	A.column_indices(current_row, row_count) = sparse_entries[i].col_;
	A.values(current_row, row_count) = sparse_entries[i].val_;
	row_count++;
	if (((i+1 < sparse_entries.size()) && (current_row != sparse_entries[i+1].row_)) 
		|| (i+1 == sparse_entries.size())) {
		while (row_count < max_row) {
			A.column_indices(current_row, row_count) = bad_entry;
			A.values(current_row, row_count) = 0.f;
			row_count++;
		}
		if (i+1 < sparse_entries.size())
			current_row = sparse_entries[i+1].row_;
		row_count = 0;
	}
  }
  in.close();
  *A_h = Matrix_ell_h(A);
  return 0;
}

int readMatlabNormalMatrix(const std::string &filename, vector<double> *A_h) {
  //read in the description header
  std::ifstream in(filename.c_str());
  if (!in.is_open()) {
    std::cerr << "could not open file: " << filename << std::endl;
    return -1;
  }
  char buffer[256];
  in.read(buffer,128);
  int32_t type;
  in.read((char*)&type,4);
  if (type == 15) {
    std::cerr << "Compression not supported. Save matlab data with '-v6' option." << std::endl;
    in.close();
    return -1;
  } else if (type != 14) {
    std::cerr << filename << " is not a matlab matrix." << std::endl;
    in.close();
    return -1;
  }
  //read in the array flags
  uint32_t data_size;
  in.read((char*)&data_size,4);
  in.read((char*)&type,4);
  if( type != 6 ) {
    std::cerr << "Invalid type for normal matrix. Must be double precision." << std::endl;
    in.close();
    return -1;
  }
  int32_t byte_per_element;
  in.read((char*)&byte_per_element,4);
  uint32_t mclass;
  in.read((char*)&mclass,4);
  mclass &= 0x000000FF;
  if (mclass == 5) {
    std::cerr << "This import routine is not for a sparse matrix file." << std::endl;
    in.close();
    return -1;
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
    return -1;
  }
  int32_t x_dim,y_dim;
  in.read((char*)&x_dim,4);
  in.read((char*)&y_dim,4);

  //Array name
  uint32_t arrayName_type;
  in.read((char*)&arrayName_type, 4);
  if (arrayName_type != 1 && arrayName_type != 2) {
    std::cerr << "Invalid variable type for array name characters (Must be 8-bit)." << std::endl;
    in.close();
    return -1;
  }
  uint32_t arrayName_length;
  in.read((char*)&arrayName_length, 4);
  //Account for padding of array name to match 64-bit requirement
  int lenRemainder = arrayName_length % 8;
  if( lenRemainder != 0 )
	  arrayName_length = arrayName_length + 8 - lenRemainder;
  in.read(buffer,arrayName_length); //Read the array name (ignore)

  //Data type in array field
  in.read((char*)&type,4);
  if( type != 9 ) {
    std::cerr << "Matrix data type must be miDOUBLE." << std::endl;
    in.close();
    return -1;
  }

  //Length of array field
  uint32_t arrayData_length;
  in.read((char*)&arrayData_length, 4);
  double readInDouble;
  unsigned int numValues = arrayData_length / byte_per_element;

  A_h->clear();
  for (int j = 0; j < numValues; ++j) {
	  in.read(buffer, byte_per_element);
	  memcpy(&readInDouble, buffer, sizeof(double));
	  A_h->push_back(readInDouble);
  }

  in.close();
  return numValues;
}

int writeMatlabArray(const std::string &filename, const Vector_h_CG &array) {

	//read in the description header
	std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
	//write description
	std::string desc = "MATLAB 5.0 MAT-file, Platform: GLNXA64, Created by SCI-Solver_FEM.";
	desc.resize(116, ' ');
	file.write((char*)desc.c_str(), desc.length());
	//write offset
	char zeros[32] = { 0 };
	file.write(zeros, 8);
	int16_t version = 0x0100;
	file.write((char*)&version, sizeof(int16_t));
	//write endian
	char endian[2] = { 'I', 'M' };
	file.write(endian, sizeof(int16_t));
	//write the matrix header and size.
	int32_t type = 14;
	file.write((char*)&type, sizeof(int32_t));
	int32_t totalSize = 0;
	long sizeAddress = (long)file.tellp();
	file.write((char*)&totalSize, sizeof(int32_t));
	long startAddress = (long)file.tellp();
	//write the array flags.
	int32_t flagsType = 6;
	int32_t flagsSize = 8;
	file.write((char*)&flagsType, sizeof(int32_t));
	file.write((char*)&flagsSize, sizeof(int32_t));
	//write the class
	uint32_t mclass = 6;
	file.write((char*)&mclass, sizeof(int32_t));
	file.write(zeros, 4);
	//write dimensions
	int32_t dimensionsType = 5;
	int32_t dimensionsSize = 8;
	int32_t dim_x = (int32_t)array.size();
	int32_t dim_y = 1;
	file.write((char*)&dimensionsType, sizeof(int32_t));
	file.write((char*)&dimensionsSize, sizeof(int32_t));
	file.write((char*)&dim_x, sizeof(int32_t));
	file.write((char*)&dim_y, sizeof(int32_t));
	//write array name
	int8_t  arrayName[8] = { 'x', '_', 'h', '\0' };
	int16_t arrayNameType = 1;
	int16_t arrayNameSize = 3;
	file.write((char*)&arrayNameType, sizeof(int16_t));
	file.write((char*)&arrayNameSize, sizeof(int16_t));
	file.write((char*)arrayName, 4 * sizeof(int8_t));
	//write the real data header
	int32_t arrayType = 9;
	int32_t arraySize = dim_x * 8;
	file.write((char*)&arrayType, sizeof(int32_t));
	file.write((char*)&arraySize, sizeof(int32_t));
	//finally write the data.
	for (size_t i = 0; i < array.size(); i++) {
		double val = static_cast <double> (array[i]);
		file.write((char*)&val, sizeof(double));
	}
	//now write back the size to the main header.
	long endAddress = (long)file.tellp();
	totalSize = endAddress - startAddress;
	file.seekp(sizeAddress);
	file.write((char*)&totalSize, sizeof(int32_t));
	file.close();
	return 0;

}
