#ifndef _NNSET_FLANN_H_
#define _NNSET_FLANN_H_

#include <flann/flann.h>
#include <vector>

#include "nnset.h"

using namespace std;

class _cNNSetFlann: public _cNNSet {
    public:
        _cNNSetFlann(int dim_x, int dim_y);
        ~_cNNSetFlann() {};

        void reset();

        void add_xy(double x[], double y[]);
        void add_xy(std::vector<double>& x, std::vector<double>& y);

        double get_xi(int index, int i);
        void get_x(int index, double x[]);
        void get_x(int index, std::vector<double>& x);

        double get_yi(int index, int i);
        void get_y(int index, double y[]);
        void get_y(int index, std::vector<double>& y);

        void get_x_padded(int index, double x[]);
        void get_x_padded(int index, std::vector<double>& x);

        void nn_x(int knn, double xq[], double dists[], int index[]);
        void nn_x(int knn, double xq[], std::vector<double>& dists, std::vector<int>& index);

        void nn_y(int knn, double yq[], double dists[], int index[]);
        void nn_y(int knn, double yq[], std::vector<double>& dists, std::vector<int>& index);


    private:
        // Data
        vector<double> _data_x;
        vector<double> _data_y;

        // KDtree
        flann::Index<L2<double> >* _index_x;
        flann::Index<L2<double> >* _index_y;

        // Temporary data
        flann::Matrix<int>    _index;
        flann::Matrix<double> _dists;

        void _nn_x(int k, double xq[]);
        void _nn_y(int k, double yq[]);

};

#endif // _NNSET_FLANN_H_
