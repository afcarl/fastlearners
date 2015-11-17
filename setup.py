import commands

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

import versioneer

VERSION='1.0'

def compile_args(packages):
	return commands.getoutput('pkg-config {} --cflags'.format(' '.join(packages))).split()

def link_args(packages):
	return commands.getoutput('pkg-config {} --libs'.format(' '.join(packages))).split()

ext_modules = [Extension(
	name         = 'fastlearners.fastlearners_pyx',
	sources      = ['cpppyx/fastlearners.pyx',
			        'cpppyx/nnset_brute.cpp',
			        #'cpppyx/nnset_flann.cpp',
					'cpppyx/lwlr.cpp',
			        'cpppyx/predict.cpp'],
	# add your path to where the eigen3 include dir resides.
	include_dirs = [numpy.get_include()],
	language     = 'c++',
	# libraries=
	extra_compile_args = compile_args(['flann', 'eigen3']) + ['-O3'],
	extra_link_args    = link_args(['flann', 'eigen3'])
	)]

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

setup(
	name         = 'fastlearners',
    version      = VERSION,
	cmdclass     = cmdclass,
    author       = 'Fabien Benureau',
    author_email = 'fabien.benureau@gmail.com',
    url          = 'github.com/humm/fastlearners.git',
    download_url = 'https://github.com/humm/fastlearners/tarball/{}'.format(VERSION),
    maintainer   = 'Fabien Benureau',
    description  = 'C++ implementation of some algorithms for the python learners package',
    license      = 'Open Science License (see fabien.benureau.com/openscience.html)',
    keywords     = 'learning algorithm',
    packages     = ['fastlearners'],
    requires     = ['numpy', 'cython'],
	ext_modules  = ext_modules,
)
