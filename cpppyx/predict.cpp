// The code works for Matrix of any dimension.

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>

#include "predict.h"

using namespace Eigen;
using namespace std;

typedef double dtype;

typedef Matrix<dtype, 1, Dynamic, RowMajor>  VectorN;  // w
typedef DiagonalMatrix<dtype, Dynamic>         DiagN;  // W

// X and Y  (X have one more dimension than the useful data (filled with 1.0)).
typedef Matrix<dtype, 1, Dynamic, RowMajor>         VectorM;  // Xq
typedef Matrix<dtype, Dynamic, 1, 0>                ColumnM;  // Yq, pinv internals
typedef Matrix<dtype, Dynamic, Dynamic, RowMajor> MatrixNxM;  // Y, X, WX = W*X
typedef Matrix<dtype, Dynamic, Dynamic, RowMajor> MatrixMxN;  // B = ((WX.T*WX)ˆ-1)*WX.T
typedef Matrix<dtype, Dynamic, Dynamic, RowMajor> MatrixMxM;  // WX.T*WX

void pseudoInverse(int dimX, MatrixMxM &a, MatrixMxM &result, double epsilon)
{
  JacobiSVD< MatrixMxM > svd = a.jacobiSvd(ComputeFullU | ComputeFullV);

  double tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs().maxCoeff();

  ColumnM single(dimX, 1);
  single = ((svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0));

  result = svd.matrixV()
          * single.asDiagonal()
          * svd.matrixU().adjoint();
}

void compute(int& dimX, int& dimY, int& dimNN, Map<VectorM>& Xq, Map<MatrixNxM>& X, Map<MatrixNxM>& Y, Map<VectorN>& w_nn, Map<ColumnM>& y)
{
    DiagN W(dimNN);                 W = w_nn.asDiagonal();
    MatrixNxM WX(dimNN, dimX);      WX = W*X;
    MatrixMxN WXT(dimX, dimNN);    WXT = WX.transpose();

    MatrixMxM WXTWX(dimX, dimX); WXTWX = WXT*WX;
    MatrixMxM WXTWX_inv(dimX, dimX);
    pseudoInverse(dimX, WXTWX, WXTWX_inv, 1e-15);
    MatrixMxN B(dimX, dimNN);        B = WXTWX_inv*WXT;


                           y = Xq*B*W*Y;
}

void predictLWR(int dimX, int dimY, int dimNN, double Xq[], double X[], double Y[], double w[], double Yq[])
{
      Map<VectorM>   mXq(Xq, 1, dimX);
      Map<MatrixNxM> mX(X, dimNN, dimX);
      Map<MatrixNxM> mY(Y, dimNN, dimY);
      Map<VectorN>   mw(w, dimNN);
      Map<ColumnM>   mYq(Yq, dimY, 1);

      compute(dimX, dimY, dimNN, mXq, mX, mY, mw, mYq);
}
