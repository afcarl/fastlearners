#include <cmath>
#include <cassert>
#include <iostream>

#include "nnset_flann.h"

const double EPS = 0.000000001;

_cNNSetFlann::_cNNSetFlann(int dim_x, int dim_y) {
    this->dim_x = dim_x;
    this->dim_y = dim_y;
    this->size = 0;

    this->_index_x = NULL;
    this->_index_y = NULL;

    this->_index = flann::Matrix<int>(new int[1], 1, 1);
    this->_dists = flann::Matrix<double>(new double[1], 1, 1);
}

void _cNNSetFlann::reset() {
    this->_data_x.clear();
    this->_data_y.clear();
    this->size = 0;

    if ( this->_index_x != NULL) {
        delete this->_index_x;
        this->_index_x = NULL;
    }
    if ( this->_index_y != NULL) {
        delete this->_index_y;
        this->_index_y = NULL;
    }
}

void _cNNSetFlann::add_xy(std::vector<double>& x, std::vector<double>& y) {
    return this->add_xy(&x[0], &y[0]);
}

void _cNNSetFlann::add_xy(double x[], double y[]) {

    for (int i = 0; i < this->dim_x; i++) {
        this->_data_x.push_back(x[i]);
    }
    for (int i = 0; i < this->dim_y; i++) {
        this->_data_y.push_back(y[i]);
    }

    if (this->size == 0) { // creating the indexes

        flann::Matrix<double> M_data_x = flann::Matrix<double>(x, 1, this->dim_x);
        //this->_index_x = new flann::Index<L2<double> >(M_data_x, flann::AutotunedIndexParams(1.0, 0.01, 0.0, 0.1));
        //this->_index_x = new flann::Index<L2<double> >(M_data_x, flann::KmeansIndexParams(32, 11, FLANN_CENTERS_KMEANSPP, 0.2));
        this->_index_x = new flann::Index<L2<double> >(M_data_x, flann::LinearIndexParams(), L2<double>());
        this->_index_x->buildIndex();

        flann::Matrix<double> M_data_y = flann::Matrix<double>(y, 1, this->dim_y);
        //this->_index_y = new flann::Index<L2<double> >(M_data_y, flann::AutotunedIndexParams(1.0, 0.01, 0.0, 0.1));
        this->_index_y = new flann::Index<L2<double> >(M_data_y, flann::KMeansIndexParams(32, 11, FLANN_CENTERS_KMEANSPP, 0.2));
        // this->_index_y = new flann::Index<L2<double> >(M_data_y, flann::LinearIndexParams(), L2<double>());
        this->_index_y->buildIndex();

    } else { // or adding incrementaly

        flann::Matrix<double> M_data_x = flann::Matrix<double>(x, 1, this->dim_x);
        this->_index_x->addPoints(M_data_x, 1.5);

        // flann::Matrix<double> M_data_y = flann::Matrix<double>(&this->_data_y[0], this->size+1, this->dim_y);
        flann::Matrix<double> M_data_y = flann::Matrix<double>(y, 1, this->dim_y);
        this->_index_y->addPoints(M_data_y, 1.5);
        // this->_index_y = new flann::Index<L2<double> >(M_data_y, flann::LinearIndexParams(), L2<double>());
        this->_index_y->buildIndex();
    }

    this->size += 1;
}

double _cNNSetFlann::get_xi(int index, int i) {
    return _data_x[index*dim_x + i];
}

void _cNNSetFlann::get_x(int index, vector<double>& x) {
    return get_x(index, &x[0]);
}

void _cNNSetFlann::get_x(int index, double x[]) {
    int offset = index*this->dim_x;
    for (int i = 0; i < this->dim_x; i++) {
        x[i] = this->_data_x[offset+i];
    }
}

void _cNNSetFlann::get_x_padded(int index, vector<double>& x) {
    return get_x_padded(index, &x[0]);
}


void _cNNSetFlann::get_x_padded (int index, double x[]) {
    int offset = index*this->dim_x;
    x[0] = 1.0;
    for (int i = 0; i < this->dim_x; i++) {
        x[i+1] = this->_data_x[offset + i];
    }
}

double _cNNSetFlann::get_yi(int index, int i) {
    return _data_y[index*dim_y + i];
}

void _cNNSetFlann::get_y(int index, vector<double>& y) {
    return get_y(index, &y[0]);
}

void _cNNSetFlann::get_y(int index, double y[]) {
    int offset = index*this->dim_y;
    for (int i = 0; i < this->dim_y; i++) {
        y[i] = this->_data_y[offset+i];
    }
}

void _cNNSetFlann::nn_x(int knn, double xq[], std::vector<double>& dists, std::vector<int>& index) {
    for (int i=0; i < knn; i++) {
        dists.push_back(-1.0);
        index.push_back(-1);
    }
    return nn_x(knn, xq, &dists[0], &index[0]);
}

void _cNNSetFlann::nn_x(int knn, double xq[], double dists[], int index[]) {
    assert(knn < this->size);

    this->_nn_x(knn, xq);

    assert((int(this->_dists[0].size()) == knn) && (int(this->_index[0].size()) == knn));

    for (int i = 0; i < knn; i++) {
        dists[i] = this->_dists[0][i];
        index[i] = this->_index[0][i];
    }
}

void _cNNSetFlann::nn_y(int knn, double yq[], std::vector<double>& dists, std::vector<int>& index) {
    return nn_y(knn, yq, &dists[0], &index[0]);
}

void _cNNSetFlann::nn_y(int knn, double yq[], double dists[], int index[]) {
    assert(knn < this->size);

    this->_nn_y(knn, yq);

    assert(int(this->_dists[0].size()) == knn && int(this->_index[0].size()) == knn);

    for (int i = 0; i < knn; i++) {
        dists[i] = this->_dists[0][i];
        index[i] = this->_index[0][i];
    }
}

// void _cNNSetFlann::nn_x(int knn, double xq[], vector<double>& dists, vector<int>& index) {
//     assert(knn < this->size);
//
//     this->_nn_x(knn, xq);
//
//     assert(int(this->_dists[0].size()) == knn && int(this->_index[0].size()) == knn);
//
//     dists = this->_dists[0];
//     index = this->_index[0];
//
//     assert(int(dists.size()) == knn && int(index.size()) == knn);
// }

// void _cNNSetFlann::nn_y(int knn, double yq[], vector<double>& dists, vector<int>& index) {
//     assert(knn < this->size);
//
//     this->_nn_y(knn, yq);
//
//     assert(int(this->_dists[0].size()) == knn && int(this->_index[0].size()) == knn);
//
//     dists = this->_dists[0];
//     index = this->_index[0];
//
//     assert(int(dists.size()) == knn && int(index.size()) == knn);
// }

void _cNNSetFlann::_nn_x(int knn, double xq[]) {
    if (this->_index.rows < knn){
        delete[] this->_index.ptr();
        delete[] this->_dists.ptr();
        this->_index = flann::Matrix<int>(new int[knn], 1, knn);
        this->_dists = flann::Matrix<double>(new double[knn], 1, knn);
    }

    flann::Matrix<double> Mxq = flann::Matrix<double>(xq, 1, this->dim_x);
    this->_index_x->knnSearch(Mxq, this->_index, this->_dists, knn, flann::SearchParams(32, 0.0, true));
}

void _cNNSetFlann::_nn_y(int knn, double yq[]) {
    if (this->_index.rows < knn){
        delete[] this->_index.ptr();
        delete[] this->_dists.ptr();
        this->_index = flann::Matrix<int>(new int[knn], 1, knn);
        this->_dists = flann::Matrix<double>(new double[knn], 1, knn);
    }

    flann::Matrix<double> Myq = flann::Matrix<double>(yq, 1, this->dim_y);
    this->_index_y->knnSearch(Myq, this->_index, this->_dists, 1, flann::SearchParams(32, 0.0, true));
}
