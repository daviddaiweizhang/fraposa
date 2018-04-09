#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#define  ARMA_DONT_USE_WRAPPER
#include "armadillo"
#include <list>
using namespace arma;
using namespace std;
extern "C" void openblas_set_num_threads(int num_threads);

void pm(const mat &matrix, int cut_p, int cut_q);
void pm(const mat &matrix);
int procrustes(mat &X, mat &Y, mat &Xnew, double &t, double &rho, mat &A, rowvec &b, int ps=0);
double pprocrustes(mat &X, mat &Y, mat &Xnew, double &t, double &rho, mat &A, rowvec &b, int iter=10000, double eps=0.000001, int ps=0);

// Use "call pm(X)" in gdb to es.cpp
void pm(const mat &matrix, int cut_p, int cut_q) {
  cout << endl << size(matrix) << endl;
  int p = matrix.n_rows - 1;
  int q = matrix.n_cols - 1;
  if(p > cut_p){
    p = cut_p-1;
  }
  if (q > cut_q-1){
    q = cut_q;
  }
  mat submat = matrix.submat(0,0,p,q);
  submat.print();
  cout << endl;
}
void pm(const mat &matrix){
  pm(matrix, 5, 5);
}

int main(){
  // mat X;
  // vec b;
  // X.load("test_X.dat");
  // b.load("test_b.dat");
  mat PC_ref;
  mat PC_ref_fat;
  mat PC_new_head;
  PC_ref.load("test_PC_ref.dat");
  PC_ref_fat.load("test_PC_ref_fat.dat");
  PC_new_head.load("test_PC_new_head.dat");
  mat PC_new_head_trsfed;
  double t;
  double rho;
  mat A;
  rowvec c;

  // Test procrustes
  cout << "Testing TRACE's procrustes functions..." << endl;
  procrustes(PC_new_head, PC_ref_fat, PC_new_head_trsfed, t, rho, A, c, 0);
  A.save("procrustes_A.dat", raw_ascii);
  c.save("procrustes_c.dat", raw_ascii);
  ofstream fout;
  fout.open("procrustes_rho.dat");
  fout << rho;
  fout.close();
  cout << "Done." << endl;

  // Test projection procrustes
  cout << "Testing TRACE's projection procrustes functions..." << endl;
  pprocrustes(PC_new_head, PC_ref, PC_new_head_trsfed, t, rho, A, c, 10000, 0.000001, 0);
  A.save("pprocrustes_A.dat", raw_ascii);
  c.save("pprocrustes_c.dat", raw_ascii);
  fout.open("pprocrustes_rho.dat");
  fout << rho;
  fout.close();
  cout << "Done." << endl;
  return 0;
}

//######################### Projection Procrustes Analysis ##########################
double pprocrustes(mat &X, mat &Y, mat &Xnew, double &t, double &rho, mat &A, rowvec &b, int iter, double eps, int ps){
	double epsilon = 0;
	int NUM = X.n_rows;
	int DimX = X.n_cols;
	int DimY = Y.n_cols;
	int i = 0;
	if(DimX<DimY){
		cout << "Error: dimension of Y cannot be higher than dimension of X." <<endl;
		return 0;
	}else if(DimX==DimY){
		procrustes(X, Y, Xnew, t, rho, A, b, ps);
		return 0;
	}else{
		mat Z = zeros<mat>(NUM, DimX-DimY);
		for(i=0; i<iter; i++){
			mat W = Y;
			W.insert_cols(DimY, Z);
			double tt;
			procrustes(X, W, Xnew, tt, rho, A, b, ps);
			mat Znew = Xnew;
			Znew.shed_cols(0,(DimY-1));
			mat Zm = mean(Znew);
			mat Zc = Znew-repmat(Zm, NUM, 1);
			mat Zd = Znew-Z;
			epsilon = trace(Zd.t()*Zd)/trace(Zc.t()*Zc);
			if(epsilon<=eps){
				break;
			}else{
				Z = Znew;
			}
		}	
		mat Xnew2;
		mat A2;
		rowvec b2;
		double rho2;
		mat X2 = Xnew;
		X2.shed_cols(DimY, (DimX-1));
		procrustes(X2, Y, Xnew2, t, rho2, A2, b2, ps);
		return epsilon; 
	}
}

int procrustes(mat &X, mat &Y, mat &Xnew, double &t, double &rho, mat &A, rowvec &b, int ps){
	int NUM = X.n_rows;
	//======================= Center to mean =======================
	mat Xm = mean(X);
	mat Ym = mean(Y);
	mat Xc = X-repmat(Xm, NUM, 1);
	mat Yc = Y-repmat(Ym, NUM, 1);
	//======================  SVD =====================
	mat C = Yc.t()*Xc;
	mat U;
	vec s;
	mat V;
	bool bflag = svd(U, s, V, C, "dc");	// use "divide & conquer" algorithm
	//bool bflag = svd(U, s, V, C);
	if(!bflag){
		cout << "Error: singular value decomposition in procrustes() fails." << endl;
		return 0;
	}
	//===================== Transformation ===================
	double trXX = trace(Xc.t()*Xc);
	double trYY = trace(Yc.t()*Yc);
	double trS = sum(s);
	A = V*U.t(); 
	if(ps==1){     // Orthogonal Procrustes analysis, match variance between X and Y
		rho = sqrt(trYY/trXX);
	}else{ 
		rho = trS/trXX;
	}
	b = Ym-rho*Xm*A;
	//============= New coordinates and similarity score ========
	Xnew = rho*X*A+repmat(b, NUM, 1);	
	mat Z = Y-Xnew;
	double d = trace(Z.t()*Z);
	double D = d/trYY;
	t = sqrt(1-D);
	return 1;
}


