import random
import numpy as np

import toolbox
import models.forward
from . import inverse
from . import sciopt
from .. import forward

relax = 0.0 # no relaxation

class NNLWRInverseModel(inverse.InverseModel):
    """Nearest Neighbor Inverse Model"""

    name = 'NNLWR'
    desc = 'Nearest Neighbor Perturbated by LWR'

    def __init__(self, dim_x, dim_y, constraints=(), pert=0.0, **kwargs):
        """
        :param pert: how much perturbation from the nearest neighbor do we allow
        """
        inverse.InverseModel.__init__(self, dim_x, dim_y, **kwargs)
        self.pert = max(0.0, pert)
        self.fmodel = models.forward.NNForwardModel(dim_x, dim_y, **kwargs)
        self.lwrmodel = forward.ESLWLRForwardModel(dim_x, dim_y, **kwargs)
        self.bfgsmodel  = sciopt.BFGSInverseModel.from_forward(self.lwrmodel, constraints = constraints, **kwargs)

    def infer_x(self, y):
        """Infer probable x from input y

        @param y  the desired output for infered x.
        """
        assert len(y) == self.fmodel.dim_y, "Wrong dimension for y. Expected %i, got %i" % (self.fmodel.dim_y, len(y))
        dists, index = self.fmodel.dataset.nn_y(y, k = 1)
        nn_x = [self.fmodel.dataset.get_x(index[0])]

        lwr_x = self.bfgsmodel.infer_x()
        d_nnlwr = toolbox.dist(lwr_x-nn_x)
        if d_nnlwr <= self.pert:
            return lwr_x
        else:
            return nn_x + pert*(lwr_x-nn_x)/d_nn_lwr

