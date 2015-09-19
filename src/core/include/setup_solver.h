#ifndef __SETUP_SOLVER_H__
#define __SETUP_SOLVER_H__
#include <amg_config.h>
#include <types.h>
#include <TriMesh.h>
#include <tetmesh.h>
#include <cutil.h>
#include <FEM/FEM2D.h>
#include <FEM/FEM3D.h>
#include <amg.h>
#include <cusp/linear_operator.h>
#include <cusp/print.h>
#include <cstring>

/*setup_solver functions for 2D and 3D meshes use the following parameters:
  cfg - This AMG_Config object contains the configuration parameters used to
  define how the solver will run.
 *meshPtr - points to the mesh object, which is either a TriMesh (2D) or
 TetMesh (3D) object defined in the trimesh library written by
 Szymon Rusinkiewicz.
 *A   - Stiffness matrix A, stored in Matrix_d
 *x_d - Solution vector x, stored in Vector_d_CG
 *b_d - RHS vector b, stored in Vector_d_CG
 verbose - bool for verbose output from the solver
 */


int setup_solver(AMG_Config& cfg, TetMesh* meshPtr, Matrix_ell_h* A_h_imported, Matrix_ell_d* A,
    Vector_h_CG* x_h, Vector_h_CG* b_h, const bool verbose);

int setup_solver(AMG_Config& cfg, TriMesh* meshPtr, Matrix_ell_d* A,
    Vector_h_CG* x_h, Vector_h_CG* b_h, const bool verbose);

void getMatrixFromMesh(AMG_Config& cfg, TetMesh* meshPtr, Matrix_ell_h* A, const bool generateStiffnessMatrix, const bool verbose);
void getMatrixFromMesh(AMG_Config& cfg, TriMesh* meshPtr, Matrix_ell_h* A, const bool getMatrixFromMesh, const bool verbose);

class SparseEntry_t {
  public:
    int32_t row_;
    int32_t col_;
    float   val_;
    SparseEntry_t(int32_t r, int32_t c, int32_t v) : row_(r), col_(c), val_(v) {}
    ~SparseEntry_t() {}
};

bool compare_sparse_entry(SparseEntry_t a, SparseEntry_t b);

int readMatlabSparseMatrix(const std::string &filename, Matrix_ell_h *A_h);

int readMatlabNormalMatrix(const std::string &filename, vector<double> *A_h);

int writeMatlabArray(const std::string &filename, const Vector_h_CG &array);

void checkMatrixForValidContents(Matrix_ell_h* A_h, const bool verbose);

#endif
