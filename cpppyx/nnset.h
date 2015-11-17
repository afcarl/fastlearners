#ifndef _NNSET_H_
#define _NNSET_H_

#include <vector>

class _cNNSet {
    public:
        ~_cNNSet() {};

        int dim_x;
        int dim_y;
        int size;

        virtual void reset() = 0;

        virtual void add_xy(double x[], double y[]) = 0;
        virtual void add_xy(std::vector<double>& x, std::vector<double>& y) = 0;

        virtual double get_xi (int index, int i) = 0;
        virtual void get_x (int index, double x[]) = 0;
        virtual void get_x (int index, std::vector<double>& x) = 0;

        virtual double get_yi (int index, int i) = 0;
        virtual void get_y (int index, double y[]) = 0;
        virtual void get_y (int index, std::vector<double>& y) = 0;

        virtual void get_x_padded (int index, double x[]) = 0;
        virtual void get_x_padded (int index, std::vector<double>& x) = 0;

        virtual void nn_x(int knn, double xq[], double dists[], int index[]) = 0;
        virtual void nn_x(int knn, double xq[], std::vector<double>& dists, std::vector<int>& index) = 0;

        virtual void nn_y(int knn, double yq[], double dists[], int index[]) = 0;
        virtual void nn_y(int knn, double yq[], std::vector<double>& dists, std::vector<int>& index) = 0;

        //virtual  void nn_xy(int knn, double xq[], double yq[], double dists[], int index[], double w_x, double w_y) = 0;
        //virtual void nn_xy_v(int knn, double xq[], double yq[], std::vector<double>& dists, std::vector<int>& index, double w_x, double w_y) = 0;
};

#endif // _NNSET_H_
