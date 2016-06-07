#include "FEMSolver.h"
#include "FEM/FEM2D.h"
#include "FEM/FEM3D.h"
#include "cutil.h"
#include "amg.h"
#include <time.h>
#include "cusp/elementwise.h"

FEMSolver::FEMSolver(
  std::string fname, bool isTetMesh, bool verbose) :
  verbose_(verbose),                  // output verbosity
  filename_(fname),                   // mesh file name
  maxLevels_(100),                    // the maximum number of levels
  maxIters_(100),                     // the maximum solve iterations
  preInnerIters_(5),                  // the pre inner iterations for GSINNER
  postInnerIters_(5),                 // the post inner iterations for GSINNER
  postRelaxes_(1),                    // The number of post relax iterations
  cycleIters_(1),                     // The number of CG iterations per outer iteration
  dsType_(0),                         // Data Structure Type
  topSize_(256),                      // the Max size of coarsest level
  randMisParameters_(90102),          // the Max size of coarsest level
  partitionMaxSize_(512),             // the largest partition size (use getMaxThreads() to determine for your device)
  aggregatorType_(0),                 // aggregator oldMis (0), metis bottom up (1), 
                                      //   metis top down (2), aggMisGPU (3), aggMisCPU (4), newMisLight (5)
  convergeType_(0),                   // Convergence tolerance algo [ABSOLUTE_CONVERGENCE (0), RELATIVE_CONVERGENCE (1)]
  tolerance_(1e-6),                   // the convergence tolerance
  cycleType_(0),                      // set the cycle algorithm
  solverType_(0),                     // the solving algorithm [AMG_SOLVER (0),PCG_SOLVER (1)]
  smootherWeight_(1.0),               // the weight parameter used in a smoother
  proOmega_(0.67),                    // the weight parameter used in a prolongator smoother
  device_(0),                         // the device number to run on
  blockSize_(256),                    // maximum size of a block
  tetMesh_(NULL),                     // the tetmesh pointer
  triMesh_(NULL)                      // the trimesh pointer
{
  if (isTetMesh) {
    this->tetMesh_ = TetMesh::read((this->filename_ + ".node").c_str(),
      (this->filename_ + ".ele").c_str(), verbose);
  } else {
    this->triMesh_ = TriMesh::read(this->filename_.c_str());
  }
  this->getMatrixFromMesh();
}

FEMSolver::~FEMSolver() {
  if (this->tetMesh_ != NULL)
    delete this->tetMesh_;
  if (this->triMesh_ != NULL)
    delete this->triMesh_;
}
/**
* Creates the mesh, partitions the mesh, and runs the algorithm.
*
* @data The set of options for the Eikonal algorithm.
*       The defaults are used if nothing is provided.
*/
void FEMSolver::solveFEM(
  Vector_h_CG* x_h, Vector_h_CG* b_h) {
  this->checkMatrixForValidContents(&this->A_h_);
  clock_t starttime, endtime;
  starttime = clock();
  Matrix_ell_d_CG A_device(this->A_h_);
  //copy back to the host
  cudaThreadSynchronize();
  //register configuration parameters
  AMG<Matrix_h, Vector_h> amg(this->verbose_, this->convergeType_,
    this->cycleType_, this->solverType_, this->tolerance_,
    this->cycleIters_, this->maxIters_, this->maxLevels_,
    this->topSize_, this->smootherWeight_, this->preInnerIters_,
    this->postInnerIters_, this->postRelaxes_, this->dsType_,
    this->randMisParameters_, this->partitionMaxSize_, this->proOmega_,
    this->aggregatorType_, this->blockSize_,
    this->triMesh_, this->tetMesh_);
  //setup multi grid for solver
  amg.setup(A_device);
  //print info
  if (this->verbose_)
    amg.printGridStatistics();
  //copy to device
  Vector_d_CG x_d(*x_h);
  Vector_d_CG b_d(*b_h);
  //run solver
  amg.solve(b_d, x_d);
  //copy back to host
  *x_h = Vector_h_CG(x_d);
  *b_h = Vector_h_CG(b_d);
  endtime = clock();
  double duration = (double)(endtime - starttime) * 1000 / CLOCKS_PER_SEC;

  if (this->verbose_)
    printf("Computing time : %.10lf ms\n", duration);
}

size_t FEMSolver::getMatrixRows() {
  return this->A_h_.num_rows;
}

bool FEMSolver::InitCUDA() {
  int count = 0;
  bool found = false;
  cudaGetDeviceCount(&count);
  if (count == 0) {
    fprintf(stderr, "There is no device.\n");
    return false;
  }
  for (int i = 0; i < count; i++) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
      if (prop.major >= 1) {
        found = true;
        break;
      }
    }
  }
  if (!found) {
    fprintf(stderr, "There is no device supporting CUDA.\n");
    return false;
  }
  cudaDeviceProp props;
  cudaSafeCall(cudaSetDevice(this->device_));
  cudaSafeCall(cudaGetDeviceProperties(&props, 0));
  if (this->verbose_) {
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
      this->device_, props.name, props.major, props.minor);
    printf("CUDA initialized.\n");
  }
  return true;
}

void FEMSolver::checkMatrixForValidContents(Matrix_ell_h* A_h) {
  if (A_h->num_rows == 0) {
    if (this->verbose_) {
      printf("Error no matrix specified\n");
    }
    std::string error = "Error no matrix specified";
    throw std::invalid_argument(error);
  }
}

void FEMSolver::getMatrixFromMesh() {
  if (this->triMesh_ == NULL && this->tetMesh_ == NULL)
    exit(0);
  Matrix_ell_d_CG A_device;
  //assembly / generate matrix step
  if (this->triMesh_ != NULL) {
    this->triMesh_->set_verbose(this->verbose_);
    this->triMesh_->need_neighbors();
    this->triMesh_->need_meshquality();
    //generate the unit constant mesh stiffness matrix
    trimesh2ell<Matrix_ell_d_CG >(this->triMesh_, A_device);
    // 2D fem solving object
    FEM2D fem2d;
    Vector_d_CG RHS(this->triMesh_->vertices.size(), 0.0);
    fem2d.initializeWithTriMesh(this->triMesh_);
    fem2d.assemble(this->triMesh_, A_device, RHS);
  } else {
    this->tetMesh_->set_verbose(this->verbose_);
    this->tetMesh_->need_neighbors();
    this->tetMesh_->need_meshquality();
    //generate the unit constant mesh stiffness matrix
    tetmesh2ell<Matrix_ell_d_CG >(this->tetMesh_, A_device, this->verbose_);
    // 3D fem solving object
    FEM3D fem3d;
    Vector_d_CG RHS(this->tetMesh_->vertices.size(), 0.0);
    fem3d.initializeWithTetMesh(this->tetMesh_);
    fem3d.assemble(this->tetMesh_, A_device, RHS, true);
  }
  cudaThreadSynchronize();
  this->A_h_ = Matrix_ell_h(A_device);
}

bool FEMSolver::compare_sparse_entry(SparseEntry_t a, SparseEntry_t b) {
  return ((a.row_ != b.row_) ? (a.row_ < b.row_) : (a.col_ < b.col_));
}

int FEMSolver::readMatlabSparseMatrix(const std::string &filename) {
  //read in the description header
  std::ifstream in(filename.c_str(), std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "could not open file: " << filename << std::endl;
    return 1;
  }
  char buffer[256];
  in.read(buffer, 128);
  int32_t type;
  in.read((char*)&type, 4);
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
  in.read((char*)&data_size, 4);
  in.read((char*)&type, 4);
  if (type != 6 && type != 5) {
    std::cerr << "Invalid type for sparse matrix. Must be 32bit." << std::endl;
    in.close();
    return 1;
  }
  int32_t byte_per_element;
  in.read((char*)&byte_per_element, 4);
  uint32_t mclass;
  in.read((char*)&mclass, 4);
  mclass &= 0x000000FF;
  if (mclass != 5) {
    std::cerr << "This is not a sparse matrix file." << std::endl;
    in.close();
    return 1;
  }
  uint32_t nzmax;
  in.read((char*)&nzmax, 4);
  //read in the dimensions and name
  in.read((char*)&type, 4);
  in.read((char*)&byte_per_element, 4);
  if ((type != 6 && type != 5) || byte_per_element != 8) {
    std::cerr << "Matrix of wrong dimension type or # of dimensions." << std::endl;
    std::cerr << "Matrix must be 2 dimensions and of 32bit type." << std::endl;
    in.close();
    return 1;
  }
  int32_t x_dim, y_dim;
  in.read((char*)&x_dim, 4);
  in.read((char*)&y_dim, 4);

  //Array name
  uint32_t arrayName_type = 0;
  uint32_t arrayName_length = 0;
  uint8_t  byteAlignmentForPadding = 4;
  in.read((char*)&arrayName_type, 2);
  in.read((char*)&arrayName_length, 2);

  //If next 16-bits are zero, then MAT file is not using the small data
  // element format for storing array name
  if (arrayName_length == 0) {
    in.read((char*)&arrayName_length, 4);
    byteAlignmentForPadding = 8;
  }
  if (arrayName_type != 1 && arrayName_type != 2) {
    std::cerr << "WARNING: Invalid variable type (" << arrayName_type;
    std::cerr << ") for array name characters (Must be 8-bit)." << std::endl;
    in.close();
    return -1;
  }

  //Account for padding of array name to match the 32-bit or 64-bit requirement,
  // depending on the short or normal format for the array name format.
  int lenRemainder = arrayName_length % byteAlignmentForPadding;
  if (lenRemainder != 0)
    arrayName_length = arrayName_length + byteAlignmentForPadding - lenRemainder;
  in.read(buffer, arrayName_length); //Read the array name (ignore)

  //read in the row indices
  in.read((char*)&type, 4);
  if (type != 6 && type != 5) {
    std::cerr << "Invalid type row index for sparse matrix. Must be 32bit." << std::endl;
    in.close();
    return 1;
  }
  in.read((char*)&byte_per_element, 4);
  std::vector<int32_t> row_vals(byte_per_element / 4, 0);
  in.read(reinterpret_cast<char*>(row_vals.data()), byte_per_element);
  //read in remaining bytes
  in.read(buffer, byte_per_element % 8);
  //read in the column indices
  in.read((char*)&type, 4);
  if (type != 6 && type != 5) {
    std::cerr << "Invalid column index type for sparse matrix. Must be 32bit." << std::endl;
    in.close();
    return 1;
  }
  in.read((char*)&byte_per_element, 4);
  std::vector<int32_t> col_vals(byte_per_element / 4, 0);
  in.read(reinterpret_cast<char*>(col_vals.data()), byte_per_element);
  //read in remaining bytes
  in.read(buffer, byte_per_element % 8);
  //read in the data values
  in.read((char*)&type, 4);
  if (type != 9) {
    std::cerr << "Invalid value for sparse matrix. " <<
      "Must be double float." << std::endl;
    in.close();
    return 1;
  }
  in.read((char*)&byte_per_element, 4);
  std::vector<double> double_vals(byte_per_element / 8, 0);
  in.read(reinterpret_cast<char*>(double_vals.data()), byte_per_element);
  in.close();
  std::vector<SparseEntry_t> sparse_entries;
  int32_t num_entries = col_vals[y_dim];
  sparse_entries.reserve(num_entries);
  std::vector<int32_t> row_max;
  row_max.resize(x_dim);
  for (size_t i = 0; i < row_max.size(); i++)
    row_max[i] = 0;
  for (size_t i = 0; i < y_dim; i++) {
    int32_t idx = col_vals[i];
    int32_t idx_end = col_vals[i + 1] - 1;
    int32_t col = static_cast<int32_t>(i);
    for (int32_t j = idx; j <= idx_end; j++) {
      row_max[row_vals[j]]++;
      sparse_entries.push_back(
        SparseEntry_t(row_vals[j], col,
        static_cast<float>(double_vals[j])));
    }
  }
  //now set up the ell matrix.
  //sort the sparse entries
  std::sort(sparse_entries.begin(), sparse_entries.end(), compare_sparse_entry);
  //determine the max nonzeros per row
  int32_t max_row = 0;
  for (size_t i = 0; i < row_max.size(); i++)
    max_row = max_row > row_max[i] ? max_row : row_max[i];
  //set up the matrix
  Matrix_ell_h A(x_dim, y_dim, num_entries, max_row);
  //iterate through to add values.
  // X is used to fill unused entries in the matrix
  const int bad_entry = Matrix_ell_h::invalid_index;
  int32_t current_row = 0, row_count = 0;
  for (size_t i = 0; i < sparse_entries.size(); i++) {
    A.column_indices(current_row, row_count) = sparse_entries[i].col_;
    A.values(current_row, row_count) = sparse_entries[i].val_;
    row_count++;
    if (((i + 1 < sparse_entries.size()) && (current_row != sparse_entries[i + 1].row_))
      || (i + 1 == sparse_entries.size())) {
      while (row_count < max_row) {
        A.column_indices(current_row, row_count) = bad_entry;
        A.values(current_row, row_count) = 0.f;
        row_count++;
      }
      if (i + 1 < sparse_entries.size())
        current_row = sparse_entries[i + 1].row_;
      row_count = 0;
    }
  }
  Matrix_ell_h original(this->A_h_);
  cusp::add(A, original, this->A_h_);
  return 0;
}

int FEMSolver::readMatlabArray(const std::string &filename, Vector_h_CG* rhs) {
  //read in the description header
  std::ifstream in(filename.c_str());
  if (!in.is_open()) {
    std::cerr << "could not open file: " << filename << std::endl;
    return -1;
  }
  char buffer[256];
  in.read(buffer, 128);
  int32_t type;
  in.read((char*)&type, 4);
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
  in.read((char*)&data_size, 4);
  in.read((char*)&type, 4);
  if (type != 6) {
    std::cerr << "Invalid type for normal matrix. Must be double precision." << std::endl;
    in.close();
    return -1;
  }
  int32_t byte_per_element;
  in.read((char*)&byte_per_element, 4);
  uint32_t mclass;
  in.read((char*)&mclass, 4);
  mclass &= 0x000000FF;
  if (mclass == 5) {
    std::cerr << "This import routine is not for a sparse matrix file." << std::endl;
    in.close();
    return -1;
  }
  uint32_t nzmax;
  in.read((char*)&nzmax, 4);
  //read in the dimensions and name
  in.read((char*)&type, 4);
  in.read((char*)&byte_per_element, 4);
  if ((type != 6 && type != 5) || byte_per_element != 8) {
    std::cerr << "Matrix of wrong dimension type or # of dimensions." << std::endl;
    std::cerr << "Matrix must be 2 dimensions and of 32bit type." << std::endl;
    in.close();
    return -1;
  }
  int32_t x_dim, y_dim;
  in.read((char*)&x_dim, 4);
  in.read((char*)&y_dim, 4);

  //Array name
  uint32_t arrayName_type = 0;
  in.read((char*)&arrayName_type, 2);
  if (arrayName_type != 1 && arrayName_type != 2) {
    std::cerr << "WARNING: Invalid variable type (" << arrayName_type;
    std::cerr << ") for array name characters (Must be 8-bit)." << std::endl;
    in.close();
    return -1;
  }
  uint32_t arrayName_length = 0;
  in.read((char*)&arrayName_length, 2);
  //Account for padding of array name to match 32-bit requirement
  int lenRemainder = arrayName_length % 4;
  if (lenRemainder != 0)
    arrayName_length = arrayName_length + 4 - lenRemainder;
  in.read(buffer, arrayName_length); //Read the array name (ignore)

  //Data type in array field
  in.read((char*)&type, 4);
  if (type != 9) {
    std::cerr << "Matrix data type must be miDOUBLE (type is ";
    std::cerr << type << ")." << std::endl;
    in.close();
    return -1;
  }

  //Length of array field
  uint32_t arrayData_length;
  in.read((char*)&arrayData_length, 4);
  double readInDouble;
  unsigned int numValues = arrayData_length / byte_per_element;

  rhs->clear();
  for (int j = 0; j < numValues; ++j) {
    in.read(buffer, byte_per_element);
    memcpy(&readInDouble, buffer, sizeof(double));
    rhs->push_back(readInDouble);
  }

  in.close();
  return 0;
}

int FEMSolver::writeMatlabArray(const std::string &filename, const Vector_h_CG &array) {

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

void FEMSolver::writeVTK(std::vector <float> values, std::string fname)
{
  if (this->tetMesh_ != NULL) {
    int nv = this->tetMesh_->vertices.size();
    int nt = this->tetMesh_->tets.size();
    FILE* vtkfile;
    vtkfile = fopen((fname + ".vtk").c_str(), "w+");
    fprintf(vtkfile, "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID\n");
    fprintf(vtkfile, "POINTS %d float\n", nv);
    for (int i = 0; i < nv; i++) {
      fprintf(vtkfile, "%.12f %.12f %.12f\n",
        this->tetMesh_->vertices[i][0],
        this->tetMesh_->vertices[i][1],
        this->tetMesh_->vertices[i][2]);
    }
    fprintf(vtkfile, "CELLS %d %d\n", nt, nt * 5);
    for (int i = 0; i < nt; i++) {
      fprintf(vtkfile, "4 %d %d %d %d\n",
        this->tetMesh_->tets[i][0],
        this->tetMesh_->tets[i][1],
        this->tetMesh_->tets[i][2],
        this->tetMesh_->tets[i][3]);
    }
    fprintf(vtkfile, "CELL_TYPES %d\n", nt);
    for (int i = 0; i < nt; i++) {
      fprintf(vtkfile, "10\n");
    }
    fprintf(vtkfile, "POINT_DATA %d\nSCALARS traveltime float 1\nLOOKUP_TABLE default\n",
      nv, values.size());
    for (int i = 0; i < values.size(); i++) {
      fprintf(vtkfile, "%.12f\n ", values[i]);
    }
    fclose(vtkfile);
  } else if (this->triMesh_ != NULL) {
    size_t nv = this->triMesh_->vertices.size();
    size_t nt = this->triMesh_->faces.size();
    FILE* vtkfile;
    vtkfile = fopen((fname + ".vtk").c_str(), "w+");
    fprintf(vtkfile, "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID\n");
    fprintf(vtkfile, "POINTS %d float\n", nv);
    for (size_t i = 0; i < nv; i++) {
      fprintf(vtkfile, "%.12f %.12f %.12f\n",
        this->triMesh_->vertices[i][0],
        this->triMesh_->vertices[i][1],
        this->triMesh_->vertices[i][2]);
    }
    fprintf(vtkfile, "CELLS %d %d\n", nt, nt * 4);
    for (size_t i = 0; i < nt; i++) {
      fprintf(vtkfile, "3 %d %d %d\n",
        this->triMesh_->faces[i][0],
        this->triMesh_->faces[i][1],
        this->triMesh_->faces[i][2]);
    }
    fprintf(vtkfile, "CELL_TYPES %d\n", nt);
    for (size_t i = 0; i < nt; i++) {
      fprintf(vtkfile, "5\n");
    }
    fprintf(vtkfile, "POINT_DATA %d\nSCALARS traveltime float 1\nLOOKUP_TABLE default\n", nv);
    for (size_t i = 0; i < nv; i++) {
      fprintf(vtkfile, "%.12f\n", values[i]);
    }
    fclose(vtkfile);
  }
}
