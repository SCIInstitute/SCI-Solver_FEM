#include <FEM/FEM3D.h>
#include <util.h>
#include <cuda/perform_element_loop_3D.cuh>

FEM3D::FEM3D(TetMesh* meshPtr)
{
  initializeWithTetMesh(meshPtr);
}

void FEM3D::initializeWithTetMesh(TetMesh* meshPtr)
{
  nv = meshPtr->vertices.size();
  ne = meshPtr->tets.size();
  IdxVector_h tri0(ne);
  IdxVector_h tri1(ne);
  IdxVector_h tri2(ne);
  IdxVector_h tri3(ne);

  for(int i = 0; i < ne; i++)
  {
    tri0[i] = meshPtr->tets[i][0];
    tri1[i] = meshPtr->tets[i][1];
    tri2[i] = meshPtr->tets[i][2];
    tri3[i] = meshPtr->tets[i][3];

  }

  Vector_h_CG vx(nv);
  Vector_h_CG vy(nv);
  Vector_h_CG vz(nv);

  for(int i = 0; i < nv; i++)
  {
    vx[i] = meshPtr->vertices[i][0];
    vy[i] = meshPtr->vertices[i][1];
    vz[i] = meshPtr->vertices[i][2];
  }

  d_tri0 = tri0;
  d_tri1 = tri1;
  d_tri2 = tri2;
  d_tri3 = tri3;

  d_vx = vx;
  d_vy = vy;
  d_vz = vz;

  tri0.resize(0);
  tri1.resize(0);
  tri2.resize(0);
  tri3.resize(0);
  vx.resize(0);
  vy.resize(0);
  vz.resize(0);

}

double compute_gamma_3d(double x)
{
  int i, k, m;
  double ga, gr, r, z;

  static double g[] = {
                       1.0,
                       0.5772156649015329,
                       -0.6558780715202538,
                       -0.420026350340952e-1,
                       0.1665386113822915,
                       -0.421977345555443e-1,
                       -0.9621971527877e-2,
                       0.7218943246663e-2,
                       -0.11651675918591e-2,
                       -0.2152416741149e-3,
                       0.1280502823882e-3,
                       -0.201348547807e-4,
                       -0.12504934821e-5,
                       0.1133027232e-5,
                       -0.2056338417e-6,
                       0.6116095e-8,
                       0.50020075e-8,
                       -0.11812746e-8,
                       0.1043427e-9,
                       0.77823e-11,
                       -0.36968e-11,
                       0.51e-12,
                       -0.206e-13,
                       -0.54e-14,
                       0.14e-14
  };

  if(x > 171.0) return 1e308; // This value is an overflow flag.
  if(x == (int)x)
  {
    if(x > 0.0)
    {
      ga = 1.0; // use factorial
      for(i = 2; i < x; i++)
      {
        ga *= i;
      }
    }
    else
      ga = 1e308;
  }
  else
  {
    if(fabs(x) > 1.0)
    {
      z = fabs(x);
      m = (int)z;
      r = 1.0;
      for(k = 1; k <= m; k++)
      {
        r *= (z - k);
      }
      z -= m;
    }
    else
      z = x;
    gr = g[24];
    for(k = 23; k >= 0; k--)
    {
      gr = gr * z + g[k];
    }
    ga = 1.0 / (gr * z);
    if(fabs(x) > 1.0)
    {
      ga *= r;
      if(x < 0.0)
      {
        ga = -M_PI / (x * ga * sin(M_PI * x));
      }
    }
  }
  return ga;
}

void FEM3D::JacobiPoly(int degree, Vector_h_CG x, int alpha, int beta, Vector_h_CG &y)
{
  int s = x.size();
  if(degree == 0)
  {

    y.resize(s);
    for(int i = 0; i < s; i++)
    {
      y[i] = 1.0;
    }

  }
  else if(degree == 1)
  {
    y.resize(s);
    for(int i = 0; i < s; i++)
    {

      y[i] = 0.5 * (alpha - beta + (alpha + beta + 2.0) * x[i]);

    }

  }
  else
  {
    double degm1 = degree - 1.0;
    double tmp = 2.0 * degm1 + alpha + beta;
    double a1 = 2.0 * (degm1 + 1)*(degm1 + alpha + beta + 1) * tmp;
    double a2 = (tmp + 1)*(alpha * alpha - beta * beta);
    double a3 = tmp * (tmp + 1.0)*(tmp + 2.0);
    double a4 = 2.0 * (degm1 + alpha)*(degm1 + beta)*(tmp + 2.0);
    Vector_h_CG poly1, poly2;
    JacobiPoly(degree - 1, x, alpha, beta, poly1);
    JacobiPoly(degree - 2, x, alpha, beta, poly2);

    int plolysize = poly1.size();
    y.resize(plolysize);



    for(int i = 0; i < plolysize; i++)
    {
      y[i] = ((a2 + a3 * x[i]) * poly1[i] - a4 * poly2[i]) / a1;
    }


  }


}

void FEM3D::JacobiPolyDerivative(int degree, Vector_h_CG &x, int alpha, int beta, Vector_h_CG &y)
{
  int s = x.size();
  if(degree == 0)
  {

    y.resize(s);
    for(int i = 0; i < s; i++)
    {
      y[i] = 0.0;
    }

  }
  else
  {
    Vector_h_CG poly;
    JacobiPoly(degree - 1, x, alpha + 1, beta + 1, poly);
    y.resize(poly.size());
    for(int i = 0; i < poly.size(); i++)
    {
      y[i] = 0.5 * (alpha + beta + degree + 1) * poly[i];
    }
  }
  //y = 0.5*(alpha+beta+degree+1)*JacobiPoly(degree-1,x,alpha+1,beta+1);
}

void FEM3D::JacobiGZeros(int degree, int alpha, int beta, Vector_h_CG &z)
{
  z.resize(degree);
  if(degree == 0)
  {
    for(int i = 0; i < degree; i++)
    {
      z[i] = 0.0;
    }
    return;
  }
  int maxit = 60;
  double EPS = 1.0e-6;
  double dth = double(PI) / (2.0 * degree);

  double rlast = 0.0;
  double one = 1.0;
  double two = 2.0;


  Vector_h_CG r;
  Vector_h_CG poly, pder;
  r.resize(1);
  poly.resize(1);
  pder.resize(1);



  double sum = 0;
  double delr;
  for(int k = 0; k < degree; k++)
  {
    r[0] = -cos((two * k + one) * dth);
    if(k)
      r[0] = 0.5 * (r[0] + rlast);


    for(int j = 0; j < maxit; j++)
    {

      JacobiPoly(degree, r, alpha, beta, poly);
      JacobiPolyDerivative(degree, r, alpha, beta, pder);

      sum = 0.0;
      for(int i = 0; i < k; i++)
        sum = sum + one / (r[0] - z[i]);

      delr = -poly[0] / (pder[0] - sum * poly[0]);
      r[0] = r[0] + delr;
      if(fabs(delr) < EPS)
        break;
    }

    z[k] = r[0];
    rlast = r[0];

  }

}

void FEM3D::JacobiGLZW(Vector_h_CG& Z, Vector_h_CG& weight, int degree, int alpha, int beta)
{
  Z.resize(degree);
  weight.resize(degree);

  double fac = 0;


  if(degree == 1)
  {
    Z[0] = 0.0;
    weight[0] = 0.0;
  }
  else
  {
    int apb = alpha + beta;

    Z[0] = -1;
    Z[degree - 1] = 1;

    Vector_h_CG tmppoly;
    JacobiGZeros(degree - 2, alpha + 1, beta + 1, tmppoly);

    for(int i = 1; i < degree - 1; i++)
    {
      Z[i] = tmppoly[i - 1];
    }
    //Z(2:degree-1) = JacobiGZeros(degree-2,alpha+one,beta+one);    
    JacobiPoly(degree - 1, Z, alpha, beta, weight);

    ValueType tmp1 = pow(ValueType(2), ValueType(apb + 1));
    ValueType tmp2 = compute_gamma_3d(alpha + degree);

    fac = tmp1 * tmp2 * compute_gamma_3d(beta + degree);
    fac = fac / ((degree - 1) * compute_gamma_3d(degree) * compute_gamma_3d(alpha + beta + degree + 1));

    for(int j = 0; j < degree; j++)
    {
      weight[j] = ValueType(fac) / (weight[j] * weight[j]);
    }
    //weight = fac./(w.*w);
    weight[0] = weight[0]*(beta + 1);
    weight[degree - 1] = weight[degree - 1]*(alpha + 1);
  }


}

void FEM3D::JacobiGRZW(Vector_h_CG& Z, Vector_h_CG& weight, int degree, int alpha, int beta)
{
  Z.resize(degree);
  weight.resize(degree);

  ValueType fac = 0;


  if(degree == 1)
  {
    Z[0] = 0.0;
    weight[0] = 2.0;
  }
  else
  {
    //one = 1.0;
    int apb = alpha + beta;
    //two = 2.0;

    Z[0] = -1;

    Vector_h_CG tmpPoly;
    JacobiGZeros(degree - 1, alpha, beta + 1, tmpPoly);

    for(int i = 1; i < degree; i++)
    {
      Z[i] = tmpPoly[i - 1];
    }
    //Z(2:degree-1) = JacobiGZeros(degree-1,alpha+one,beta+one);    
    JacobiPoly(degree - 1, Z, alpha, beta, weight);
    ValueType tmp = compute_gamma_3d(alpha + degree);

    fac = pow(ValueType(2), ValueType(apb)) * tmp * compute_gamma_3d(beta + degree);
    fac = fac / (compute_gamma_3d(degree)*(beta + degree) * compute_gamma_3d(apb + degree + 1));

    for(int j = 0; j < degree; j++)
    {
      weight[j] = ValueType(fac)*(1 - Z[j]) / (weight[j] * weight[j]);
    }

    weight[0] = weight[0]*(beta + 1);

  }

}

void FEM3D::Transform2StdTetSpace(const Vector_h_CG &z_x, const Vector_h_CG &z_y, const Vector_h_CG &z_z, CGType(*VecXYZ)[DEGREE][DEGREE][3])
{

  int nx = z_x.size();
  int ny = z_y.size();
  int nz = z_z.size();
  CGType cx, cy, cz;

  for(int i = 0; i < nx; i++)
  {
    cx = z_x[i];
    for(int j = 0; j < ny; j++)
    {
      cy = z_y[j];
      for(int k = 0; k < nz; k++)
      {
        cz = z_z[k];
        VecXYZ[i][j][k][0] = (1 + cx)*0.5 * (1 - cy) * 0.5 * (1 - cz) * 0.5;
        VecXYZ[i][j][k][1] = (1 + cy) * 0.5 * (1 - cz) * 0.5;
        VecXYZ[i][j][k][2] = (1 + cz) * 0.5;
      }
    }
  }
}

void FEM3D::EvalBasisTet(CGType(*coefmat)[4], const CGType(*VecXYZ)[DEGREE][DEGREE][3], CGType(*phi)[DEGREE][DEGREE][4])
{

  CGType* coef;
  CGType cx, cy, cz;
  for(int s = 0; s < 4; s++)
  {
    coef = coefmat[s];
    for(int i = 0; i < DEGREE; i++)
    {
      for(int j = 0; j < DEGREE; j++)
      {
        for(int k = 0; k < DEGREE; k++)
        {

          cx = VecXYZ[i][j][k][0];
          cy = VecXYZ[i][j][k][1];
          cz = VecXYZ[i][j][k][2];
          phi[i][j][k][s] = coef[0] + coef[1] * cx + coef[2] * cy + coef[3] * cz;


        }
      }
    }
  }


}


CGType FEM3D::Integration_Quadrilateral_3d(ValueType(*fx)[DEGREE][DEGREE], Vector_h_CG &w_x, Vector_h_CG &w_y, Vector_h_CG &w_z)
{
  ValueType integral = 0;
  ValueType tmp_y, tmp_z;

  for (int i = 0; i < DEGREE; i++)
  {
    tmp_y = 0.0;
    for (int j = 0; j < DEGREE; j++)
    {
      tmp_z = 0.0;
      for (int k = 0; k < DEGREE; k++)
      {
        //        tmp_z += fx[i][j][k] * c_w_z_3d[k];
        tmp_z += fx[i][j][k] * w_z[k];
      }
      //      tmp_y += tmp_z * c_w_y_3d[j];
      tmp_y += tmp_z * w_y[j];
    }
    //    integral += tmp_y * c_w_x_3d[i];
    integral += tmp_y * w_x[i];
  }

  return integral;
}

void FEM3D::IntegrationInTet(Vector_h_CG &phi, Vector_h_CG &weight_x, Vector_h_CG &weight_y, Vector_h_CG &weight_z, Vector_h_CG &integralMass)
{
	CGType integrandMass[DEGREE][DEGREE][DEGREE];
	int cnt = 0;
  for(int k = 0; k < 4; k++)
  {
    for(int g = k; g < 4; g++)
    {
      for(int p = 0; p < DEGREE; p++)
      {
        for(int q = 0; q < DEGREE; q++)
        {
          for(int r = 0; r < DEGREE; r++)
          {
            integrandMass[p][q][r] = phi[k * DEGREE * DEGREE * DEGREE + p * DEGREE * DEGREE + q * DEGREE + r] * phi[g * DEGREE * DEGREE * DEGREE + p * DEGREE * DEGREE + q * DEGREE + r];
          }
        }
      }
			integralMass[cnt++] = Integration_Quadrilateral_3d(integrandMass, weight_x, weight_y, weight_z);
    }
  }

  
}


//void FEM3D::IntegrationForce(Vector_h_CG &phi, Vector_h_CG &weight_x, Vector_h_CG &weight_y, Vector_h_CG &weight_z, Vector_h_CG &integralForce)
//{
//	CGType integrandForce[DEGREE][DEGREE][DEGREE];
//	int cnt = 0;
//  for(int k = 0; k < 4; k++)
//  {
//    for(int g = k; g < 4; g++)
//    {
//      for(int p = 0; p < DEGREE; p++)
//      {
//        for(int q = 0; q < DEGREE; q++)
//        {
//          for(int r = 0; r < DEGREE; r++)
//          {
//            integrandForce[p][q][r] = phi[k * DEGREE * DEGREE * DEGREE + p * DEGREE * DEGREE + q * DEGREE + r] * phi[g * DEGREE * DEGREE * DEGREE + p * DEGREE * DEGREE + q * DEGREE + r];
//          }
//        }
//      }
//			integralForce[cnt++] = Integration_Quadrilateral_3d(integrandForce, weight_x, weight_y, weight_z);
//    }
//  }
//}

void FEM3D::assemble(TetMesh* meshPtr, Matrix_ell_d_CG &A, Vector_d_CG &b, bool isdevice)
{
  int degree_x = DEGREE;
  int degree_y = DEGREE;
  int degree_z = DEGREE;

  Vector_h_CG z_x, z_y, z_z;
  Vector_h_CG weight_x, weight_y, weight_z;

  JacobiGLZW(z_x, weight_x, degree_x, 0, 0);
  JacobiGRZW(z_y, weight_y, degree_y, 1, 0);
  JacobiGRZW(z_z, weight_z, degree_z, 2, 0);

  for(int i = 0; i < degree_y; i++)
  {
    weight_y[i] /= 2;
    weight_z[i] /= 4;
  }

  CGType qdTet[DEGREE][DEGREE][DEGREE][3];

  Transform2StdTetSpace(z_x, z_y, z_z, qdTet);

  CGType coefmatBaseTet[4][4] = {
    {1, -1, -1, -1},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1}
  };

  CGType phiTet[DEGREE][DEGREE][DEGREE][4];


  EvalBasisTet(coefmatBaseTet, qdTet, phiTet);

  //	Vector_h_CG Integration(10);
  //
  //	int cnt = 0;
  //	for(int p =0;p<4; p++)
  //	{
  //		for(int q = p; q<4; q++)
  //		{
  //			 IntegrationInTet(phiTet, weight_x, weight_y, weight_z, integrand[cnt++]);
  //		}
  //	}

  Vector_h_CG phi(DEGREE * DEGREE * DEGREE * 4);
  for(int l = 0; l < 4; l++)
    for(int i = 0; i < DEGREE; i++)
      for(int j = 0; j < DEGREE; j++)
        for(int k = 0; k < DEGREE; k++)
          phi[l * DEGREE * DEGREE * DEGREE + i * DEGREE * DEGREE + j * DEGREE + k] = phiTet[i][j][k][l];

  //  Vector_d_CG phi_d = phi;
  //  phi.clear();

      Vector_h_CG integrandMass(10);
  IntegrationInTet(phi, weight_x, weight_y, weight_z, integrandMass);

//	Vector_h_CG integrandForce(10);
//	IntegrationForce(phi, weight_x, weight_y, weight_z, integrandForce);
	



  ValueType * tmp_w_x = thrust::raw_pointer_cast(&weight_x[0]);
  ValueType* tmp_w_y = thrust::raw_pointer_cast(&weight_y[0]);
  ValueType* tmp_w_z = thrust::raw_pointer_cast(&weight_z[0]);

  IdxVector_d matlabels = meshPtr->matlabels;
  Vector_d_CG integrandMass_d = integrandMass;	

  perform_element_loop_3d(d_vx, d_vy, d_vz, d_tri0, d_tri1, d_tri2, d_tri3, A, b, phi, weight_x, weight_y, weight_z, matlabels, integrandMass_d, isdevice);
  phi.clear();

}

