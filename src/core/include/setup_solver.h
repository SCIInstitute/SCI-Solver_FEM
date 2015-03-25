#pragma once

int setup_solver(AMG_Config& cfg, TriMesh* meshPtr, TetMesh* tetmeshPtr,
                 FEM2D* fem2d, FEM3D* fem3d, Matrix_d* A,
                 Vector_d_CG* x_d, Vector_d_CG* b_d, bool verbose);
