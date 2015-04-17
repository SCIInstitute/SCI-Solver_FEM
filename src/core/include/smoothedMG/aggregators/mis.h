#ifndef __MIS_H__
#define __MIS_H__

#include <TriMesh.h>
#include "types.h"

extern "C"
{
#include "metis.h"
}

#include <vector>
#include <set>
template <class Matrix, class Vector> class MIS_Aggregator;
template <class Matrix, class Vector> class RandMIS_Aggregator;

#include <smoothedMG/aggregators/aggregator.h>

template <class Matrix, class Vector>
class MIS_Aggregator : public Aggregator<Matrix, Vector>
{
    typedef typename Matrix::value_type ValueType;
    public:
        void computePermutation(TriMesh* meshPtr, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitionlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize);
        void computePermutation(TetMesh* meshPtr, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitonlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize);
        void computePermutation(int nn, int* xadj, int* adjncy, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitionlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize);
	void computePermutation_d(IdxVector_d &adjIndexesIn, IdxVector_d &adjacencyIn, IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, int parameters, int part_max_size, bool verbose = false);
        void computePermutation_d(TriMesh* meshPtr, IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, int parameters, int part_max_size, bool verbose = false);
        void computePermutation_d(TetMesh* meshPtr, IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, int parameters, int part_max_size, bool verbose = false);
    private:
        void aggregateGraphMIS(int n, int *adjIndexes, int *adjacency, int *partition, int *partCount);   
};

template <class Matrix, class Vector>
class RandMIS_Aggregator : public Aggregator<Matrix, Vector>
{
    typedef typename Matrix::value_type ValueType;
    public:
        void computePermutation(TriMesh* meshPtr, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitionlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize);
        void computePermutation(TetMesh* meshPtr, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitonlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize);
        void computePermutation(int nn, int* xadj, int* adjncy, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitionlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize);
        void computePermutation_d(IdxVector_d &adjIndexesIn, IdxVector_d &adjacencyIn, IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, int parameters, int part_max_size, bool verbose = false);
        void computePermutation_d(TriMesh* meshPtr, IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, int parameters, int part_max_size, bool verbose = false);
        void computePermutation_d(TetMesh* meshPtr, IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionlabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, int parameters, int part_max_size, bool verbose = false);
    private:
        void extendedMIS(int n, int depth, int *adjIndexes, int *adjacency, int *partition, int *partCount, bool verbose = false);    
};
#endif
