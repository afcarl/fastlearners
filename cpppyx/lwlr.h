#ifndef _LWLR_H_
#define _LWLR_H_

#include "nnset.h"

class _cLwlr {
    public:
        _cLwlr(int dim_x, int dim_y, int k, double sigma, _cNNSet* dset);
        ~_cLwlr();
        int dim_x, dim_y;
        int k;
        double sigma, sigma_sq;
        int es;
        _cNNSet* nnset;

        void predict_y(double xq[], double yq[], int k, double sigma_sq);

    private:
        // Temporary data
        std::vector<double>     _w;
        std::vector<double> _xqext;
        std::vector<double> _yqext;
        std::vector<double>     _x;
        std::vector<double>     _y;

        // Functions

        void _weights(int, double, std::vector<double>& dists);
        double _gauss(double, double);
};

#endif // _LWR_H_
