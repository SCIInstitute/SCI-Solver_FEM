SCI-Solver_FEM
==============

SCI-Solver_FEM is a C++/CUDA library written to solve an FEM linear system. It is designed to solve the FEM system quickly by using GPU hardware.

The code was written by Zhisong Fu and T. James Lewis. The theory behind this code is published in the paper:
"Architecting the Finite Element Method Pipeline for the GPU"

**AUTHORS** Zhisong Fu(a,b), T. James Lewis(a,b), Robert M. Kirby(a,b), Ross T. Whitaker(a,b)

`  `a-School of Computing, University of Utah, Salt Lake City, UT, USA

`  `b-Scientific Computing and Imaging Institute, University of Utah, Salt Lake City, USA

**ABSTRACT**
The finite element method (FEM) is a widely employed numerical technique
for approximating the solution of partial differential equations (PDEs) in var-
ious science and engineering applications. Many of these applications benefit
from fast execution of the FEM pipeline. One way to accelerate the FEM
pipeline is by exploiting advances in modern computational hardware, such as
the many-core streaming processors like the graphical processing unit (GPU).
In this paper, we present the algorithms and data-structures necessary to move
the entire FEM pipeline to the GPU. First we propose an efficient GPU-based
algorithm to generate local element information and to assemble the global lin-
ear system associated with the FEM discretization of an elliptic PDE. To solve
the corresponding linear system efficiently on the GPU, we implement a conju-
gate gradient method preconditioned with a geometry-informed algebraic multi-
grid (AMG) method preconditioner. We propose a new fine-grained parallelism
strategy, a corresponding multigrid cycling stage and efficient data mapping
to the many-core architecture of GPU. Comparison of our on-GPU assembly
versus a traditional serial implementation on the CPU achieves up to an 87×
speedup. Focusing on the linear system solver alone, we achieve a speedup of
up to 51× versus use of a comparable state-of-the-art serial CPU linear system
solver. Furthermore, the method compares favorably with other GPU-based,
sparse, linear solvers.
