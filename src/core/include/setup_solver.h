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
int setup_solver(AMG_Config& cfg, TriMesh* meshPtr, TetMesh* tetmeshPtr,
                 FEM2D* fem2d, FEM3D* fem3d, Matrix_d* A,
                 Vector_d_CG* x_d, Vector_d_CG* b_d, bool verbose);
#endif
