GPUTUM : FEM Solver
==============

GPUTUM FEM Solver is a C++/CUDA library written to solve an FEM linear system. It is designed to solve the FEM system quickly by using GPU hardware.

The code was written by Zhisong Fu and T. James Lewis at the Scientific Computing and Imaging Institute, 
University of Utah, Salt Lake City, USA. The theory behind this code is published in the papers linked below. 
Table of Contents
========
<img src="https://raw.githubusercontent.com/SCIInstitute/SCI-Solver_FEM/master/src/Resources/fem.png"  align="right" hspace="20" width=350>
- [FEM Aknowledgements](#fem-aknowledgements)
- [Requirements](#requirements)
- [Building](#building)<br/>
		- [Linux / OSX](#linux-and-osx)<br/>
		- [Windows](#windows)<br/>
- [Running Examples](#running-examples)
- [Using the Library](#using-the-library)
- [Testing](#testing)<br/>

<br/><br/><br/><br/><br/><br/><br/><br/>

<h4>FEM Aknowledgements</h4>
**<a href ="http://www.sciencedirect.com/science/article/pii/S0377042713004470">
Architecting the Finite Element Method Pipeline for the GPU</a>**<br/>
<img src="https://raw.githubusercontent.com/SCIInstitute/SCI-Solver_FEM/master/src/Resources/fem2.png"  align="right" hspace="20" width=360>

**AUTHORS:**
<br/>Zhisong Fu(*a*) <br/>
T. James Lewis(*b*) <br/>
Robert M. Kirby(*a*) <br/>
Ross T. Whitaker(*a*) <br/>

This library solves for the partial differential equations and coefficients values 
on vertices located on a tetrahedral or triangle mesh on the GPU. Several mesh formats
are supported, and are read by the <a href="http://wias-berlin.de/software/tetgen/">TetGen Library</a> and the
 <a href="http://graphics.stanford.edu/software/trimesh/">TriMesh Library</a>. 
The <a href="http://glaros.dtc.umn.edu/gkhome/metis/metis/download">METIS library</a> is used to partition unstructured 
meshes. <a href="https://code.google.com/p/googletest/">
Google Test</a> is used for testing.
<br/><br/>
Requirements
==============

 * Git, CMake (3.0+ recommended), and the standard system build environment tools.
 * You will need a CUDA Compatible Graphics card. See <a href="https://developer.nvidia.com/cuda-gpus">here</a> You will also need to be sure your card has CUDA compute capability of at least 2.0.
 * SCI-Solver_FEM is compatible with the latest CUDA toolkit (7.0). Download <a href="https://developer.nvidia.com/cuda-downloads">here</a>.
 * This project has been tested on OpenSuse 12.3 (Dartmouth) on NVidia GeForce GTX 570 HD, Ubuntu 14.04 on NVidia GeForce GTX 560 Ti, Windows 7 on NVidia GeForce GTX 775M, and OSX 10.10 on NVidia GeForce GTX 775M. 
 * If you have a CUDA compatible card with the above operating systems, and are experiencing issues, please contact the repository owners.
 * Windows: You will need Microsoft Visual Studio 2013 build tools. This document describes the "NMake" process.
 * OSX: Please be sure to follow setup for CUDA <a href="http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/#axzz3W4nXNNin">here</a>. There are several compatability requirements for different MAC machines, including using a different version of CUDA (ie. 5.5).

Building
==============

<h3>Linux / OSX</h3>
In a terminal:
```c++
mkdir SCI-SOLVER_FEM/build
cd SCI-SOLVER_FEM/build
cmake ../src
make
```

<h3>Windows</h3>
Open a Visual Studio (32 or 64 bit) Native Tools Command Prompt. 
Follow these commands:
```c++
mkdir C:\Path\To\SCI-Solver_FEM\build
cd C:\Path\To\SCI-Solver_FEM\build
cmake -G "NMake Makefiles" ..\src
nmake
```

**Note:** For all platforms, you may need to specify your CUDA toolkit location (especially if you have multiple CUDA versions installed):
```c++
cmake -DCUDA_TOOLKIT_ROOT_DIR="~/NVIDIA/CUDA-7.0" ../src
```
(Assuming this is the location).

**Note:** If you have compile errors such as <code>undefined reference: atomicAdd</code>, it is likely you need to set your compute capability manually. CMake outputs whether compute capability was determined automatically, or if you need to set it manually. The default (and known working) minimum compute capability is 2.0.

```c++
cmake -DCUDA_COMPUTE_CAPABILITY=20 ../src
make
```


Running Examples
==============

You will need to enable examples in your build to compile and run them

```c++
cmake -DBUILD_EXAMPLES=ON ../src
make
```

You will find the example binaries built in the <code>build/examples</code> directory.

Run the examples in the build directory:

```c++
examples/Example1
examples/Example2
...
```
Each example has a <code>-h</code> flag that prints options for that example. <br/>

Follow the example source code in <code>src/examples</code> to learn how to use the library.
<br/>
To run examples similar to the paper, the following example calls would do so:<br/>
<b>2D FEM, Egg Carton </b><br/>
<code>examples/Example2 -v -i ../src/test/test_data/simple.ply -A "../src/test/test_data/simpleTri.mat" -b "../src/test/test_data/simpleTrib.mat"</code><br/>

**NOTE** All examples output a set of <code>result.vtk</code> (name based off input 
filename) VTK files in the current directory. These files are easily viewed via VTK readers like Paraview.
You can clip and add iso-values to more distinctly visualize the result. An <code>output.mat</code>
MATLAB file is also written to file (solution coefficients).

Using the Library
==============

A basic usage of the library links to the <code>libFEM_CORE</code> library during build and 
includes the headers needed, which are usually no more than:

```c++
#include "FEMSolver.h"
```

Then a program would setup the FEM parameters using the 
<code>"FEMSolver object"</code> object and call
<code>object.solveFEM()</code> to generate
the answer matrices.

Here is a minimal usage example (using a tet mesh).<br/>
```c++
#include <FEMSolver.h>
int main(int argc, char *argv[])
{
  //the filename in the constructor below means ~/myTetMesh.node & ~/myTetMesh.ele
  FEMSolver data("~/myTetMesh", false, true); // tet mesh, not a tri mesh, and verbose
  //read in your Matrices (A matrix object is a member of FEMSolver)
  data.readMatlabSparseMatrix("~/A_MATRIX.mat");
  Vector_h_CG b_h(cfg.getMatrixRows(), 1.0);
  data.readMatlabArray("~/b_array.mat", &b_h)
  //The answer vector.
  Vector_h_CG x_h(cfg.getMatrixRows(), 0.0);
  //Run the solver
  data.solveFEM(&x_h, &b_h);
  //now use the result
  data.writeMatlabArray("outputName.mat", x_h);
  //write the VTK
  std::vector<double> vals;
  for (size_t i = 0; i < x_h.size(); i++){
    vals.push_back(x_h[i]);
  }
  data.writeVTK(vals, "outputName");
  return 0;
}
```

You can access the A matrix and meshes directly:
```c++
TetMesh * FEMSolver::tetMesh_;
TriMesh * FEMSolver::triMesh_;
```

<h3>FEM Solver Parameters</h3>

```C++
  class FEMSolver {
	  bool verbose_;                  // output verbosity
	  std::string filename_;          // mesh file name
	  int maxLevels_;                 // the maximum number of levels
	  int maxIters_;                  // the maximum solve iterations
	  int preInnerIters_;             // the pre inner iterations for GSINNER
	  int postInnerIters_;            // the post inner iterations for GSINNER
	  int postRelaxes_;               // the number of post relax iterations
	  int cycleIters_;                // the number of CG iterations per outer iteration
	  int dsType_;                    // data structure type
	  int topSize_;                   // max size of coarsest level
	  int randMisParameters_;         // max size of coarsest level
	  int partitionMaxSize_;          // max size of of the partition
	  int aggregatorType_;            // aggregator oldMis (0), metis bottom up (1), 
									  //   metis top down (2), aggMisGPU (3), aggMisCPU (4), newMisLight (5)
	  int convergeType_;              // the convergence tolerance algorithm <absolute (0)|relative (1)>
	  double tolerance_;              // the convergence tolerance
	  int cycleType_;                 // the cycle algorithm <V (0) | W (1) | F (2) | K (3)>
	  int solverType_;                // the solving algorithm <AMG (0) | PCG (1)>
	  double smootherWeight_;         // the weight parameter used in a smoother
	  double proOmega_;               // the weight parameter used in prolongator smoother
	  int device_;                    // the GPU device number to specify
	  int blockSize_;
      ...
  };
```
<br/>
You will need to make sure your CMake/Makfile/Build setup knows where 
to point for the library and header files. See the examples and their CMakeLists.txt.<br/><br/>
Testing
==============
The repo comes with a set of regression tests to see if recent changes break
expected results. To build the tests, you will need to set
<code>BUILD_TESTING</code> to "ON" in either <code>ccmake</code> or when calling CMake:

```c++
cmake -DBUILD_TESTING=ON ../src
```
After building, run <code>make test</code> or <code>ctest</code> in the build directory to run tests.<br/>
<h4>Windows</h4>
The gtest library included in the repo needs to be built with 
forced shared libraries on Windows, so use the following:

```c++
cmake -DBUILD_TESTING=ON -Dgtest_forced_shared_crt=ON ../src
```
Be sure to include all other necessary CMake definitions as annotated above.
