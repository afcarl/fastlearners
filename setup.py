import os
from setuptools import setup

setup(
    name = "models",
    version = "0.3",
    author = "Fabien Benureau and Clement Moulin-Frier",
    author_email = "fabien.benureau@inria.fr, clement.moulin-frier@gmail.com",
    description = ("A python implementation of LWR and other classic forward and inverse models"),
    license = "Not Yet Decided.",
    keywords = "lwr machine-learning",
    url = "flowers.inria.fr",
    packages=['models', 'models.samples',
                        'models.forward',
                        'models.inverse',
                        'models.testbed',
                        'models.plots'],
    install_requires=['numpy', 'scipy', 'pandas'],
    #long_description=read('README'),
    classifiers=[],
)
