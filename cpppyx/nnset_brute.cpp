#include <cmath>
#include <cassert>
#include <iostream>

#include "nnset_brute.h"

using namespace std;

_cNNSetBrute::_cNNSetBrute(int dim_x, int dim_y) {
    this->dim_x = dim_x;
    this->dim_y = dim_y;

    size = 0;
}

void _cNNSetBrute::reset() {
    size = 0;

    _data_x.clear();
    _data_y.clear();
}

void _cNNSetBrute::add_xy(double x[], double y[]) {

    vector<double> x_v, y_v;
    x_v.assign(x, x + dim_x);
    y_v.assign(y, y + dim_y);

    add_xy(x_v, y_v);
}

void _cNNSetBrute::add_xy(vector<double>& x, vector<double>& y) {
    _data_x.push_back(x);
    _data_y.push_back(y);
    size += 1;
}


double _cNNSetBrute::get_xi(int index, int i) {
    return _data_x[index][i];
}

void _cNNSetBrute::get_x(int index, double x[]) {
    for (int i = 0; i < this->dim_x; i++) {
        x[i] = _data_x[index][i];
    }
}

void _cNNSetBrute::get_x(int index, vector<double>& x) {
    x.resize(dim_x);
    copy(_data_x[index].begin(), _data_x[index].end(), x.begin());
}


double _cNNSetBrute::get_yi(int index, int i) {
    return _data_y[index][i];
}

void _cNNSetBrute::get_y(int index, double y[]) {
    for (int i = 0; i < this->dim_y; i++) {
        y[i] = this->_data_y[index][i];
    }
}

void _cNNSetBrute::get_y(int index, vector<double>& y) {
    y.resize(dim_y);
    copy(_data_y[index].begin(), _data_y[index].end(), y.begin());
}


void _cNNSetBrute::get_x_padded(int index, double x[]) {
    x[0] = 1.0;
    for (int i = 1; i < this->dim_x+1; i++) { // Can that copy be avoided ?
        x[i] = this->_data_x[index][i-1];
    }
}

void _cNNSetBrute::get_x_padded(int index, vector<double>& x) {
    x.resize(dim_x+1);
    x[0] = 1.0;
    copy(_data_x[index].begin(), _data_x[index].end(), x.begin()+1);
}


void _cNNSetBrute::nn_x(int knn, double xq[], double dists[], int index[]) {

    nn_x(knn, xq, _dists, _index);

    for (int i = 0; i < knn; i++) {
        dists[i] = this->_dists[i];
        index[i] = this->_index[i];
    }
}

void _cNNSetBrute::nn_x(int knn, double xq[], vector<double>& dists, vector<int>& index) {

    vector<double> xq_v;
    xq_v.assign(xq, xq + dim_x);

    dists.clear();
    index.clear();
    _heap = priority_queue<pair<double, int>, vector<pair<double, int> >, greater<pair<double, int> > >();

    for (int i = 0; i < int(_data_x.size()); i++) {
        double d = L2(xq_v, _data_x[i]);

        _heap.push(make_pair(d, i));
    }

    assert(int(_heap.size()) > knn);

    for (int i = 0; i < knn; i++) {
        dists.push_back(_heap.top().first);
        index.push_back(_heap.top().second);
        _heap.pop();
    }
}


void _cNNSetBrute::nn_y(int knn, double yq[], double dists[], int index[]) {

    nn_y(knn, yq, this->_dists, this->_index);
    for (int i = 0; i < knn; i++) {
        dists[i] = this->_dists[i];
        index[i] = this->_index[i];
    }
}

void _cNNSetBrute::nn_y(int knn, double yq[], vector<double>& dists, vector<int>& index) {

    vector<double> yq_v;
    yq_v.assign(yq, yq + dim_y);

    dists.clear();
    index.clear();
    _heap = priority_queue<pair<double, int>, vector<pair<double, int> >, greater<pair<double, int> > >();

    for (int i = 0; i < int(_data_y.size()); i++) {
        double d = L2(yq_v, _data_y[i]);

        _heap.push(make_pair(d, i));
    }

    assert(int(_heap.size()) > knn);

    for (int i = 0; i < knn; i++) {
        dists.push_back(_heap.top().first);
        index.push_back(_heap.top().second);
        _heap.pop();
    }
}

// void _cNNSetBrute::nn_xy(int knn, double xq[], double yq[], double dists[], int index[], double w_x = 1.0, double w_y = 1.0) {
//
//     nn_xy_v(knn, xq, yq, this->_dists, this->_index, w_x, w_y);
//     for (int i = 0; i < knn; i++) {
//         dists[i] = this->_dists[i];
//         index[i] = this->_index[i];
//     }
// }
//
// void _cNNSetBrute::nn_xy_v(int knn, double xq[], double yq[], vector<double>& dists, vector<int>& index, double w_x = 1.0, double w_y = 1.0) {
//
//     assert(w_x + w_y > 0);
//
//     vector<double> xq_v, yq_v;
//     xq_v.assign(xq, xq + dim_x);
//     yq_v.assign(yq, yq + dim_y);
//
//     dists.clear();
//     index.clear();
//     _heap = priority_queue<pair<double, int>, vector<pair<double, int> >, greater<pair<double, int> > >();
//
//     for (int i = 0; i < int(_data_x.size()); i++) {
//         double d = (w_x*L2(xq_v, _data_x[i]) + w_y*L2(yq_v, _data_y[i]))/(w_x + w_y);
//         //printf("%.2f : %.2f/%.2f %.2f/%.2f\n", d, xq_v[0], _data_x[i][0], yq_v[0], _data_y[i][0]);
//
//         _heap.push(make_pair(d, i));
//     }
//
//     assert(int(_heap.size()) > knn);
//
//     for (int i = 0; i < knn; i++) {
//         dists.push_back(_heap.top().first);
//         index.push_back(_heap.top().second);
//         _heap.pop();
//     }
// }

double _cNNSetBrute::L2(vector<double>& a, vector<double>&b) {
    double d = 0;
    for (unsigned int i = 0; i < a.size(); i++) {
        d += (a[i] - b[i])*(a[i] - b[i]);
    }
    return d;
}
