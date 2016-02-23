#include <FEM/FEM2D.h>
#include <cuda/perform_element_loop_2D.cuh>
#include <util.h>

#define  PI 3.1415927

FEM2D::FEM2D(TriMesh* meshPtr)
{
  initializeWithTriMesh(meshPtr);
}

void FEM2D::initializeWithTriMesh(TriMesh* meshPtr)
{
  nv = meshPtr->vertices.size();
  ne = meshPtr->faces.size();
  IdxVector_h tri0(ne);
  IdxVector_h tri1(ne);
  IdxVector_h tri2(ne);

  for(int i = 0; i < ne; i++)
  {
    tri0[i] = meshPtr->faces[i][0];
    tri1[i] = meshPtr->faces[i][1];
    tri2[i] = meshPtr->faces[i][2];
  }

  Vector_h_CG vx(nv);
  Vector_h_CG vy(nv);

  for(int i = 0; i < nv; i++)
  {
    vx[i] = meshPtr->vertices[i][0];
    vy[i] = meshPtr->vertices[i][1];
  }

  d_tri0 = tri0;
  d_tri1 = tri1;
  d_tri2 = tri2;

  d_vx = vx;
  d_vy = vy;

  tri0.resize(0);
  tri1.resize(0);
  tri2.resize(0);
  vx.resize(0);
  vy.resize(0);
}

double compute_gamma(double x)
{
	int i,k,m;
	double ga,gr,r,z;

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
		0.14e-14};

		if (x > 171.0) return 1e308;    // This value is an overflow flag.
		if (x == (int)x) {
			if (x > 0.0) {
				ga = 1.0;               // use factorial
				for (i=2;i<x;i++) {
					ga *= i;
				}
			}
			else
				ga = 1e308;
		}
		else {
			if (fabs(x) > 1.0) {
				z = fabs(x);
				m = (int)z;
				r = 1.0;
				for (k=1;k<=m;k++) {
					r *= (z-k);
				}
				z -= m;
			}
			else
				z = x;
			gr = g[24];
			for (k=23;k>=0;k--) {
				gr = gr*z+g[k];
			}
			ga = 1.0/(gr*z);
			if (fabs(x) > 1.0) {
				ga *= r;
				if (x < 0.0) {
					ga = -M_PI/(x*ga*sin(M_PI*x));
				}
			}
		}
		return ga;
}

void FEM2D::JacobiPoly(int degree, Vector_h_CG x,int alpha,int beta, Vector_h_CG &y)
{
	int s = x.size();
	if (degree == 0)
	{
		
		y.resize(s);
		for (int i =0; i< s; i++)
		{
			y[i] = 1.0;
		}

	}
	else if(degree == 1)
	{
		y.resize(s);
		for (int i =0; i<s; i++)
		{

			y[i] = 0.5*(alpha-beta+(alpha+beta+2.0)*x[i]);

		}
		
	}
	else
	{
		double degm1 = degree-1.0; 
		double tmp = 2.0*degm1+alpha+beta;
		double a1= 2.0*(degm1+1)*(degm1+alpha+beta+1)*tmp;
		double a2= (tmp+1)*(alpha*alpha-beta*beta);
		double a3= tmp*(tmp+1.0)*(tmp+2.0);
		double a4= 2.0*(degm1+alpha)*(degm1+beta)*(tmp+2.0);
		Vector_h_CG poly1, poly2;
		JacobiPoly(degree-1,x,alpha,beta, poly1);
		JacobiPoly(degree-2,x,alpha,beta, poly2);

		int plolysize = poly1.size();
		y.resize(plolysize);



		for (int i=0; i<plolysize; i++)
		{
			y[i] = ((a2+a3*x[i])*poly1[i]- a4*poly2[i] )/a1;
		}

		
	}


}

void FEM2D::JacobiPolyDerivative(int degree, Vector_h_CG &x, int alpha,int beta, Vector_h_CG &y)
{
	int s = x.size();
	if (degree == 0)
	{

		y.resize(s);
		for (int i =0; i< s; i++)
		{
			y[i] = 0.0;
		}

	}
	else
	{
		Vector_h_CG poly;
		JacobiPoly(degree-1,x,alpha+1,beta+1, poly);
		y.resize(poly.size());
		for (int i =0; i<poly.size(); i++)
		{
			y[i] = 0.5*(alpha+beta+degree+1)*poly[i];
		}
	}
	//y = 0.5*(alpha+beta+degree+1)*JacobiPoly(degree-1,x,alpha+1,beta+1);
}

void FEM2D::JacobiGZeros(int degree,int alpha,int beta, Vector_h_CG &z)
{
	z.resize(degree);
	if (degree == 0)
	{
		for (int i =0; i<degree; i++)
		{
			z[i] =0.0;
		}
		return;
	} 
	int	maxit = 60;
	double EPS = 1.0e-6;
	double dth =double(PI)/(2.0*degree);

	double rlast=0.0;
	double one = 1.0;
	double two = 2.0;


	Vector_h_CG r;
	Vector_h_CG poly, pder;
	r.resize(1);
	poly.resize(1);
	pder.resize(1);



	double sum = 0;
	double delr;
	for (int k=0; k< degree; k++) 
	{
		r[0] = -cos((two*k + one) * dth);
		if (k)
			r[0] = 0.5*(r[0] + rlast);
		

		for(int j = 0; j < maxit; j++) 
		{

			JacobiPoly(degree,r,alpha,beta, poly);
			JacobiPolyDerivative(degree,r,alpha,beta, pder);

			sum = 0.0;
			for (int i=0; i< k; i++)
				sum = sum + one/(r[0] - z[i]);
			  
			delr = -poly[0] / (pder[0] - sum * poly[0]);
			r[0]  = r[0] + delr;
			if (fabs(delr) < EPS)
				break;
		}
	
		z[k]  = r[0];
		rlast = r[0];
		
	}

}



void FEM2D::JacobiGLZW(Vector_h_CG&  Z,Vector_h_CG& weight,  int degree, int alpha, int beta)
{
	Z.resize(degree);
	weight.resize(degree);

	double  fac=0 ;


	if (degree == 1)
	{
		Z[0] = 0.0;
		weight[0] = 0.0;
	}
	else
	{
		//one = 1.0;
		int apb = alpha + beta;
		//two = 2.0;

		Z[0] = -1;
		Z[degree-1] = 1;

		Vector_h_CG tmppoly; 
	  JacobiGZeros(degree-2,alpha+1,beta+1, tmppoly);

		for (int i = 1; i< degree-1; i++)
		{
			Z[i] = tmppoly[i-1];
		}
		//Z(2:degree-1) = JacobiGZeros(degree-2,alpha+one,beta+one);    
		JacobiPoly(degree-1,Z,alpha,beta, weight);

    Matrix_ell_d_CG::value_type tmp1 = pow(Matrix_ell_d_CG::value_type(2), Matrix_ell_d_CG::value_type(apb + 1));
    Matrix_ell_d_CG::value_type tmp2 = compute_gamma(alpha + degree);

		 fac =  tmp1 * tmp2 * compute_gamma(beta + degree);
		fac = fac / ((degree-1)*compute_gamma(degree)*compute_gamma(alpha + beta + degree + 1));

		for (int j =0; j< degree; j++)
		{
      weight[j] = Matrix_ell_d_CG::value_type(fac) / (weight[j] * weight[j]);
		}
		//weight = fac./(w.*w);
		weight[0] = weight[0]*(beta+1);
		weight[degree-1] = weight[degree-1]*(alpha+1);
	}


}


void FEM2D::JacobiGRZW(Vector_h_CG&  Z, Vector_h_CG& weight,  int degree, int alpha, int beta)
{
	Z.resize(degree);
	weight.resize(degree);

  Matrix_ell_d_CG::value_type fac = 0;


	if (degree == 1)
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
	  JacobiGZeros(degree-1,alpha,beta+1, tmpPoly);

		for (int i = 1; i< degree; i++)
		{
			Z[i] = tmpPoly[i-1];
		}
		//Z(2:degree-1) = JacobiGZeros(degree-1,alpha+one,beta+one);    
		JacobiPoly(degree-1,Z,alpha,beta, weight);
    Matrix_ell_d_CG::value_type tmp = compute_gamma(alpha + degree);

    fac = pow(Matrix_ell_d_CG::value_type(2), Matrix_ell_d_CG::value_type(apb)) * tmp *compute_gamma(beta + degree);
		fac = fac / (compute_gamma(degree)*(beta+degree)*compute_gamma(apb+degree + 1));

		for (int j =0; j< degree; j++)
		{
      weight[j] = Matrix_ell_d_CG::value_type(fac)*(1 - Z[j]) / (weight[j] * weight[j]);
		}
		
		weight[0] = weight[0]*(beta+1);
		
	}

}


void FEM2D::assemble(TriMesh* meshPtr, Matrix_ell_d_CG &A, Vector_d_CG &b)
{
	int degree_x = 6;
	int degree_y = 6;

	int alpha1 = 0,  beta1 = 0;
	int alpha2 = 1,  beta2 = 0;

	Vector_h_CG z_x, z_y;
	Vector_h_CG weight_x, weight_y;

	JacobiGLZW(z_x,  weight_x, degree_x, alpha1, beta1);
	JacobiGRZW(z_y,  weight_y, degree_y, alpha2, beta2);

  Matrix_ell_d_CG::value_type* tmp_w_x = thrust::raw_pointer_cast(&weight_x[0]);
  Matrix_ell_d_CG::value_type* tmp_w_y = thrust::raw_pointer_cast(&weight_y[0]);
  Matrix_ell_d_CG::value_type* tmp_z_x = thrust::raw_pointer_cast(&z_x[0]);
  Matrix_ell_d_CG::value_type* tmp_z_y = thrust::raw_pointer_cast(&z_y[0]);

 perform_element_loop_2d(d_vx, d_vy, d_tri0, d_tri1, d_tri2, A, b, z_x, z_y, weight_x, weight_y);
}

void FEM2D::assemble(TriMesh* meshPtr, Matrix_d_CG &A, Vector_d_CG &b)
{
	int degree_x = 6;
	int degree_y = 6;

	int alpha1 = 0,  beta1 = 0;
	int alpha2 = 1,  beta2 = 0;

	Vector_h_CG z_x, z_y;
	Vector_h_CG weight_x, weight_y;

	JacobiGLZW(z_x,  weight_x, degree_x, alpha1, beta1);
	JacobiGRZW(z_y,  weight_y, degree_y, alpha2, beta2);


 perform_element_loop_2d_coo(d_vx, d_vy, d_tri0, d_tri1, d_tri2, A, b, z_x, z_y, weight_x, weight_y);
 

}
