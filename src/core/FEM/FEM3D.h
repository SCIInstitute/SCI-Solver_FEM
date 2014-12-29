#pragma once

#include <tetmesh.h>
#include <types.h>
#define DEGREE 4

class FEM3D
{
public:
  typedef typename Matrix_ell_d_CG::index_type IndexType;
  typedef typename Matrix_ell_d_CG::value_type ValueType;

  FEM3D()
  {
  };
  FEM3D(TetMesh* meshPtr);
  void initializeWithTetMesh(TetMesh* meshPtr);
  void assemble(TetMesh* meshPtr, Matrix_ell_d_CG &A, Vector_d_CG &b, bool isdevice);
  void JacobiGLZW(Vector_h_CG& Z, Vector_h_CG& weight, int degree, int alpha, int beta);
  void JacobiGRZW(Vector_h_CG& Z, Vector_h_CG& weight, int degree, int alpha, int beta);
  void JacobiPoly(int degree, Vector_h_CG x, int alpha, int beta, Vector_h_CG &y);
  void JacobiPolyDerivative(int degree, Vector_h_CG &x, int alpha, int beta, Vector_h_CG &y);
  void JacobiGZeros(int degree, int alpha, int beta, Vector_h_CG &z);
  void Transform2StdTetSpace(const Vector_h_CG &z_x, const Vector_h_CG &z_y, const Vector_h_CG &z_z, CGType(*VecXYZ)[DEGREE][DEGREE][3]);
  void EvalBasisTet(CGType(*coefmatBaseTet)[4], const CGType(*qdTet)[DEGREE][DEGREE][3], CGType(*phiTet)[DEGREE][DEGREE][4]);
  void IntegrationInTet(Vector_h_CG &phi, Vector_h_CG &weight_x, Vector_h_CG &weight_y, Vector_h_CG &weight_z, Vector_h_CG &integralMass);
	CGType Integration_Quadrilateral_3d(ValueType(*fx)[DEGREE][DEGREE], Vector_h_CG &w_x, Vector_h_CG &w_y, Vector_h_CG &w_z);


  IdxVector_d d_tri0;
  IdxVector_d d_tri1;
  IdxVector_d d_tri2;
  IdxVector_d d_tri3;
  Vector_d_CG d_vx;
  Vector_d_CG d_vy;
  Vector_d_CG d_vz;
  int nv;
  int ne;
};

