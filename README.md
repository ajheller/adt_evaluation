# ADT_Evaluation

## Introduction

This is my latest sandbox for a Python/NumPy/SciPy/SymPy/Jax
implementation of the Ambisonic Decoder Toolbox (ADT). It is a work
in progress, so use at your own risk. One of my goals with this is to
have everything include unit tests for pedagogical reasons and so you
can convince yourself that the code is doing the correct thing.  In
some cases, like real_spherical_harmonics.py the unit tests are longer
than the implementation itself.

There are two code-bases here: Decoder generation and producing
evaluation plots of decoders generated by the MATLAB ADT.

To make Plotly plots of decoder performance:

1. Save the results from a MATLAB ADT run to a json "SCMD" file. For an
   example of how to do this see

           adt/examples/run_brh_spring2017.m

2. run rErV.py to make the 3D speaker layout sphere plots
3. run plotly_image.py to make the 2D performance plots

## Installation

This code is tested with Python3.8, although I think it should run
in version 3.6 or newer as it uses f-strings.  The core code also
needs:

 - NumPy
 
 - SciPy
 
 - SymPy
 
 - Pandas
 
 - Matplotlib

These are all available with the Anaconda distribution of Python

The optimizer needs:

 - Google JAX, https://github.com/google/jax#installation
 
 - Dominate, pip install dominate

The fancy 3D graphics need:
* Plotly, pip install plotly

 - Plotly, pip install plotly
 
## Running in the cloud

The code has been tested in Google's Colaboratory ( <https://colab.research.google.com/notebooks/intro.ipynb> ).
The optimizer will run somewhat faster in a runtime with a GPU, 
though it is not necessary. More instructions pending. 

## License

The code in this package is licensed under the GNU Affero Public
License.  The Faust code generated by the toolbox is covered by the
BSD 3-Clause License, so that it may be combined with other code
without restriction. If these terms are an impediment to your use of
the toolbox, please contact me with details of your application.

## Notation

I've been trying to clean up the notation I used in the MATLAB
ADT. I've been starting unit vector and components of unit vectors
with 'u', so [ux, uy, uz] and using a suffix of 0 to indicate a
flattened (ravel'ed) variable.

## NumPy representation of vectors and collections of vectors

> *Consistency is the last refuge of the unimaginative.* -- Oscar Wilde


I've struggled with the question of how to represent vectors and
arrays that are a collection of vectors. By linear algebra convention,
vectors in an N-dimensional space are written as Nx1 column vectors,
for 3-D often written in inline text as [x y z]^T (the transpose of a
row vector). By extension, a collection of M 3-D coordinates is
written as an [3 x M] array. Transforming those coordinates into
spherical harmonics results in a [rank(Y) x M] array, where rank(Y) is
the number of spherical harmonics in the basis.

In MATLAB, there are no 1-D arrays, so a column vector is a 3x1 array
and a row vector is 1x3. NumPy on the other hand, has 1-D arrays, but
if we make a list of M of them, and then turn that in into an nd-array
with numpy.array, we get an Mx3 array. numpy.column_stack gives the
desired result, but the printing is awkward, as it shows row-major.

    In[107]: v1 = (1,2,3); v2=(10,20,30)

	In[108]: np.array((v1, v2))
	Out[108]:
	array([[ 1,  2,  3],
           [10, 20, 30]])

	In[109]: np.column_stack((v1, v2))
	Out[109]:
	array([[ 1, 10],
           [ 2, 20],
           [ 3, 30]])

The use of [N x 1] arrays for vectors is consistent with NumPy and
SciPy linear algebra writeups, such as:

 * <https://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html>

 * <http://www2.lawrence.edu/fast/GREGGJ/Python/numpy/numpyLA.html>

There is a NumPy matrix object that has the semantics of MATLAB's 2-D
arrays, however its use is discouraged, and it may be removed in the
future.

 * <https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html>


So far so good... but now when we look at some SciPy functions that
take collections of vectors as their input, like those found in
scipy.spatial, we see that they use row vectors, hence must be called
with the transpose of the arrays we construct above. See
scipy.Delaunay, for example:

     points : ndarray of floats, shape (npoints, ndim)
              Coordinates of points to triangulate

Sigh...

Aaron Heller <heller@ai.sri.com>
23 Sept 2019

3 Jan 2020 Addendum:

I recently found a good discussion of this on Stack Exchange:
<https://stats.stackexchange.com/questions/284995/are-1-dimensional-numpy-arrays-equivalent-to-vectors>

------
<a href="https://scan.coverity.com/projects/ajheller-adt_evaluation">
  <img alt="Coverity Scan Build Status"
       src="https://scan.coverity.com/projects/20006/badge.svg"/>
</a>
