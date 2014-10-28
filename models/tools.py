import math

def dist(p1, p2):
    return math.sqrt(sum((p1i-p2i)**2 for p1i, p2i in zip(p1, p2)))

def gaussian_kernel(d, sigma_sq):
    """Compute the guassian kernel function of a given distance
    @param d         the euclidean distance
    @param sigma_sq  sigma of the guassian, squared.
    """
    return math.exp(-(d*d)/(2*sigma_sq))
