#ifndef _NNSET_BRUTE_H_
#define _NNSET_BRUTE_H_

#include <vector>
#include <queue>

#include "nnset.h"

class _cNNSetBrute: public _cNNSet {

    public:
        _cNNSetBrute(int dim_x, int dim_y);
        ~_cNNSetBrute() {};

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
        std::vector< std::vector<double> > _data_x;
        std::vector< std::vector<double> > _data_y;

        // Temporary data
        std::vector<int>    _index;
        std::vector<double> _dists;

        std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int> >, std::greater<std::pair<double, int> > > _heap;

        double L2(std::vector<double>& a, std::vector<double>&b);

};

#endif // _NNSET_BRUTE_H_
