#ifndef __FEM2D_H__
#define __FEM2D_H__

#include <TriMesh.h>
#include <types.h>
#include <vector>
using namespace std;

class FEM2D
{
public:
  typedef typename Matrix_ell_d_CG::index_type IndexType;
  typedef typename Matrix_ell_d_CG::value_type ValueType;

  FEM2D()
  {
  };
  FEM2D(TriMesh* meshPtr);
  void initializeWithTriMesh(TriMesh* meshPtr);
  void assemble(TriMesh* meshPtr, Matrix_ell_d_CG &A, Vector_d_CG &b);
  void assemble(TriMesh* meshPtr, Matrix_d_CG &A, Vector_d_CG &b);
  void JacobiGLZW(Vector_h_CG& Z, Vector_h_CG& weight, int degree, int alpha, int beta);
  void JacobiGRZW(Vector_h_CG& Z, Vector_h_CG& weight, int degree, int alpha, int beta);
  void JacobiPoly(int degree, Vector_h_CG x, int alpha, int beta, Vector_h_CG &y);
  void JacobiPolyDerivative(int degree, Vector_h_CG &x, int alpha, int beta, Vector_h_CG &y);
  void JacobiGZeros(int degree, int alpha, int beta, Vector_h_CG &z);


  IdxVector_d d_tri0;
  IdxVector_d d_tri1;
  IdxVector_d d_tri2;
  Vector_d_CG d_vx;
  Vector_d_CG d_vy;
  int nv;
  int ne;
};
#endif
