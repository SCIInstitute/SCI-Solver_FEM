#ifndef __AGGREGATOR_H__
#define __AGGREGATOR_H__
template <class Matrix, class Vector> class Aggregator;

#include <error.h>
#include <types.h>
#include <TriMesh.h>
#include <tetmesh.h>

template <class Matrix, class Vector>
class Aggregator
{
  typedef typename Matrix::value_type ValueType;

  public:
  virtual void computePermutation(TriMesh* meshPtr, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitonlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize) = 0;
  virtual void computePermutation(TetMesh* meshPtr, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitonlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize) = 0;
  virtual void computePermutation(int nn, int* xadj, int* adjncy, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitionlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize) = 0;
  virtual void computePermutation_d(IdxVector_d &adjIndexesIn, IdxVector_d &adjacencyIn, IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, int aggregation_type, int parameters, int part_max_size, bool verbose = false) = 0;
  virtual void computePermutation_d(TriMesh* meshPtr, IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, int aggregation_type, int parameters, int part_max_size, bool verbose = false) = 0;
  virtual void computePermutation_d(TetMesh* meshPtr, IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, int aggregation_type, int parameters, int part_max_size, bool verbose = false) = 0;

  virtual ~Aggregator()
  {
  }
  static Aggregator<Matrix, Vector>* allocate(int type);

};
#endif
