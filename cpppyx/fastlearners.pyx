# 3 steps:
#   1. cython fmodel.pyx  -> fmodel.cpp
#   2. link: python setup.py build_ext --inplace  -> fmodel.so, a dynamic library
#   3. python test.py

import numpy as np
cimport numpy as np

from cython.operator import dereference as deref

cdef extern from "nnset.h":
    cdef cppclass _cNNSet:
        int dim_x, dim_y
        int size

        void reset()
        void add_xy(double* x, double* y)

        void get_x(int index, double* X)
        void get_x_padded(int index, double* X)
        void get_y(int index, double* Y)

        void nn_x(int knn, double* xq, double* dists, int* index)
        void nn_y(int knn, double* yq, double* dists, int* index)
        #void nn_xy(int knn, double* xq, double* yq, double* dists, int* index, double w_x, double w_y)

cdef extern from "nnset_brute.h":
    cdef cppclass _cNNSetBrute(_cNNSet):
        _cNNSetBrute(int, int)
        int dim_x, dim_y
        int size

        void reset()
        void add_xy(double* x, double* y)

        void get_x(int index, double* X)
        void get_x_padded(int index, double* X)
        void get_y(int index, double* Y)

        void nn_x(int knn, double* xq, double* dists, int* index)
        void nn_y(int knn, double* yq, double* dists, int* index)
        #void nn_xy(int knn, double* xq, double* yq, double* dists, int* index, double w_x, double w_y)

# cdef extern from "nnset_flann.h":
#     cdef cppclass _cNNSetFlann(_cNNSet):
#         _cNNSetFlann(int, int)
#         int dim_x, dim_y
#         int size
#
#         void reset()
#         void add_xy(double* x, double* y)
#
#         void get_x(int index, double* X)
#         void get_x_padded(int index, double* X)
#         void get_y(int index, double* Y)
#
#         void nn_x(int knn, double* xq, double* dists, int* index)
#         void nn_y(int knn, double* yq, double* dists, int* index)
#         # void nn_xy(int knn, double* xq, double* yq, double* dists, int* index, double w_x, double w_y)



cdef class cNNSet:

    cdef _cNNSet *thisptr      # hold a C++ instance which we're wrapping

    @classmethod
    def from_xy(cls, x_array, y_array):
        dim_x = len(x_array[0])
        dim_y = len(y_array[0])
        dset = cls(dim_x, dim_y)
        for x, y in zip(x_array, y_array):
            dset.add_xy(x,y)
        return dset

    def __cinit__(self, int dim_x, int dim_y, brute=True):
        self.thisptr = new _cNNSetBrute(dim_x, dim_y)
        # flann bug with addPoints()
        # if brute:
        #     self.thisptr = new _cNNSetBrute(dim_x, dim_y)
        # else:
        #     # self.thisptr = new _cNNSetFlann(dim_x, dim_y)

    def __dealloc__(self):
        pass
        #del self.thisptr

    def __len__(self):
        return self.thisptr.size

    def reset(self):
        self.thisptr.reset()

    def get_x(self, int index):
        cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self.dim_x, dtype = np.double)
        self.thisptr.get_x(index, <double*> x.data)
        return x

    def get_x_padded(self, int index):
        cdef np.ndarray[np.double_t,ndim=1] x = np.zeros(self.dim_x+1, dtype = np.double)
        self.thisptr.get_x_padded(index, <double*> x.data)
        return x

    def get_y(self, int index):
        cdef np.ndarray[np.double_t,ndim=1] y = np.zeros(self.dim_y, dtype = np.double)
        self.thisptr.get_y(index, <double*> y.data)
        return y

    def get_xy(self, int index):
        return self.get_x(index), self.get_y(index)

    def add_xy(self, x, y):
        assert len(x) == self.dim_x and len(y) == self.dim_y
        cdef np.ndarray[np.double_t,ndim=1] x_nparray = np.asfarray(x, dtype = np.double)
        cdef np.ndarray[np.double_t,ndim=1] y_nparray = np.asfarray(y, dtype = np.double)
        self.thisptr.add_xy( <double*> x_nparray.data, <double*> y_nparray.data )

    def iter_x(self):
        index = 0
        while index < self.size:
            yield self.get_x(index)
            index += 1

    def iter_y(self):
        index = 0
        while index < self.size:
            yield self.get_y(index)
            index += 1

    def iter_xy(self):
        index = 0
        while index < self.size:
            yield self.get_x(index), self.get_y(index)
            index += 1

    def nn_x(self, xq, int k = 1):

        assert len(xq) == self.dim_x
        cdef np.ndarray[np.double_t,ndim=1] xq_nparray = np.asfarray(xq, dtype = np.double)

        cdef int nn_real = min(k, self.size)
        cdef np.ndarray[np.double_t,ndim=1] dists = np.zeros(nn_real, dtype = np.double)
        cdef np.ndarray[np.int32_t,ndim=1]    idx = np.zeros(nn_real, dtype = np.int32)
        self.thisptr.nn_x( nn_real, <double*> xq_nparray.data, <double*> dists.data, <int*> idx.data )
        return dists, idx

    def nn_y(self, yq, int k = 1):

        assert len(yq) == self.dim_y
        cdef np.ndarray[np.double_t,ndim=1] yq_nparray = np.asfarray(yq, dtype = np.double)

        cdef int nn_real = min(k, self.size)
        cdef np.ndarray[np.double_t,ndim=1] dists = np.zeros(nn_real, dtype = np.double)
        cdef np.ndarray[np.int32_t,ndim=1]    idx = np.zeros(nn_real, dtype = np.int32)
        self.thisptr.nn_y( nn_real, <double*> yq_nparray.data, <double*> dists.data, <int*> idx.data )
        return dists, idx

    # def nn_xy(self, xq, yq, int k = 1, double w_x = 1.0, double w_y = 1.0):
    #
    #     assert len(yq) == self.dim_y
    #     cdef np.ndarray[np.double_t,ndim=1] xq_nparray = np.asfarray(xq, dtype = np.double)
    #     cdef np.ndarray[np.double_t,ndim=1] yq_nparray = np.asfarray(yq, dtype = np.double)
    #
    #     cdef int nn_real = min(k, self.size)
    #     cdef np.ndarray[np.double_t,ndim=1] dists = np.zeros(nn_real, dtype = np.double)
    #     cdef np.ndarray[np.int32_t,ndim=1]    idx = np.zeros(nn_real, dtype = np.int32)
    #     self.thisptr.nn_xy( nn_real, <double*> xq_nparray.data, <double*> yq_nparray.data, <double*> dists.data, <int*> idx.data, w_x, w_y)
    #     return dists, idx

    property dim_x:
        def __get__(self): return self.thisptr.dim_x

    property dim_y:
        def __get__(self): return self.thisptr.dim_y

    property size:
        def __get__(self): return self.thisptr.size


cdef extern from "lwlr.h":
    cdef cppclass _cLwlr:
        _cLwlr(int, int, int, double, _cNNSet*)
        int dim_x, dim_y
        int k
        bint es
        double sigma, sigma_sq
        _cNNSet* nnset

        void predict_y(double* xq, double* yq, int k, double sigma_sq)


cdef class cLWLR:

    cdef _cLwlr  *thisptr # hold a C++ instance which we're wrapping
    #cdef _cNNSet *_cnnset
    cpdef cNNSet cnnset

    def __init__(self, *args, **kwargs):
        pass

    def __cinit__(self, int dim_x, int dim_y, double sigma=1.0, int k=-1, cNNSet dset=None):
        if k == -1:
            k = 2*dim_x+1
        if dset is not None:
            self.cnnset = dset
        else:
            self.cnnset = cNNSet(dim_x, dim_y)
        self.thisptr  = new _cLwlr(dim_x, dim_y, k, sigma, self.cnnset.thisptr)

    def __dealloc__(self):
        pass
        #del self.thisptr

    property nnset:
        def __get__(self): return self.c_nnset
        def __set__(self, cnnset):
            self.cnnset = <cNNSet> cnnset
            self.thisptr.nnset = self.cnnset.thisptr

    def reset(self):
        self.cnnset.reset()

    def add_xy(self, x, y):
        return self.cnnset.add_xy(x, y)

    def get_x(self, int index):
        return self.cnnset.get_x(index)

    def get_y(self, int index):
        return self.cnnset.get_y(index)

    def get_xy(self, int index):
        return self.cnnset.get_xy(index)

    def predict(self, xq,
                  int k = -1, double sigma = -1):

        sigma_sq = self.thisptr.sigma_sq if sigma == -1 else sigma*sigma
        k = self.thisptr.k if k == -1 else k
        cdef int k_real = min(k, self.cnnset.size)

        cdef np.ndarray[np.double_t,ndim=1] xq_nparray = np.asfarray(xq, dtype = np.double)
        cdef np.ndarray[np.double_t,ndim=2] yq = np.zeros((1, self.dim_y), dtype = np.double)
        self.thisptr.predict_y(<double*> xq_nparray.data, <double*> yq.data, k_real, sigma_sq)
        return yq.ravel()

    def size(self):
        return self.cnnset.size

    def config(self):
        return {'sigma': self.sigma, 'k' : self.k}

    property name:
        def __get__(self):
            if self.thisptr.es:
                return 'ES-LWLR'
            else:
                return 'LWLR'

    property dim_x:
        def __get__(self): return self.thisptr.dim_x

    property dim_y:
        def __get__(self): return self.thisptr.dim_y

    property sigma:
        def __get__(self): return self.thisptr.sigma
        def __set__(self, sigma):
            self.thisptr.sigma = sigma
            self.thisptr.sigma_sq = sigma*sigma

    property sigma_sq:
        def __get__(self): return self.thisptr.sigma_sq

    property es:
        # If true, then the LWRL operate under ES-LWLR behavior
        def __get__(self): return self.thisptr.es
        def __set__(self, es):
            self.thisptr.es = es

    property k:
        def __get__(self): return self.thisptr.k
        def __set__(self, k):
            self.thisptr.k = k
