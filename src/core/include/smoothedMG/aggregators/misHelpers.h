#ifndef __MIS_HELPERS_H__
#define __MIS_HELPERS_H__

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/graph/maximal_independent_set.h>
#include <types.h>
#include <TriMesh.h>
#include <tetmesh.h>
#include <AggMIS_Types.h>
#include <string>

typedef cusp::coo_matrix<int, int, cusp::device_memory> devMtx;
typedef cusp::csr_matrix<int, int, cusp::host_memory> hostMtx;
typedef thrust::device_vector<int> IntVector_d;
typedef thrust::device_vector<float> FloatVector_d;

namespace misHelpers {

   struct subtractFrom {
      const int a;

      subtractFrom(int _a) : a(_a) {
      }

      __host__ __device__
            int operator()(const int &x) const {
         return x - a;
      }

   };

   struct ifLabelOne {

      __host__ __device__
            int operator()(const int &a, const int &b) const {
         if (b == 0)
            return -1;

         return a - 1;
      }
   };

   struct removePart {
      const int a;

      removePart(int _a) : a(_a) {
      }

      __host__ __device__
            int operator()(const int &x) const {
         if (x == a)
            return -1;

         if (x > a)
            return x - 1;

         return x;
      }
   };

   struct findAggregated {

      __host__ __device__
            int operator()(const int &a) const {
         if (a == -1)
            return 0;
         else
            return 1;
      }
   };

   struct lessThan {
      const int a;

      lessThan(int _a) : a(_a) {
      }

      __host__ __device__
            int operator()(const int &x) const {
         return x < a;
      }
   };

   struct labelLessThan {
      const int a;

      labelLessThan(int _a) : a(_a) {
      }

      __host__ __device__
            int operator()(const int &x) const {
         return x < a ? 1 : 0;
      }
   };

   void getSizes(IdxVector_d &adjIndexes, IdxVector_d &sizes);

   void getMIS(IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &misStencil, int depth);

   void randomizedMIS(IdxVector_d adjIndexes, IdxVector_d adjacency, IdxVector_d &mis, int k);

   void getAdjacency(TriMesh * meshPtr, IdxVector_d &adjIndexes, IdxVector_d &adjacency);

   void getAdjacency(TetMesh * meshPtr, IdxVector_d &adjIndexes, IdxVector_d &adjacency);

   void aggregateGraph(int minSize, int depth, IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &misStencil, bool verbose);

   void aggregateWeightedGraph(int maxSize, int fullSize, int depth, IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &partIn, IdxVector_d &nodeWeights, bool verbose = false);

   bool removeRuntyParts(int minSize, IdxVector_d &partition);

   bool removeRuntyPartitions(int minSize, IdxVector_d &partition, IdxVector_d &nodeWeights, bool verbose = false);

   void getPartSizes(IdxVector_d &partition, IdxVector_d &partSizes);

   void getPartSizes(IdxVector_d &partition, IdxVector_d &partSizes, IdxVector_d &partIndices);

   void getPartIndices(IdxVector_d &sortedPartition, IdxVector_d &partIndices);

   void getPartIndicesNegStart(IdxVector_d& sortedPartition, IdxVector_d& partIndices);

   void fillWithIndex(IdxVector_d &tofill);

   void getInversePermutation(IdxVector_d &original, IdxVector_d &inverse);

   void permuteInitialAdjacency(IdxVector_d &adjIndexesIn, IdxVector_d &adjacencyIn, IdxVector_d &permutedAdjIndexesIn, IdxVector_d &permutedAdjacencyIn, IdxVector_d &ipermutation, IdxVector_d &fineAggregate);

   void getInducedGraphNeighborCounts(IdxVector_d &aggregateIdx, IdxVector_d &adjIndexesOut, IdxVector_d &permutedAdjIndexesIn, IdxVector_d &permutedAdjacencyIn);

   void fillCondensedAdjacency(IdxVector_d &aggregateIdx, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, IdxVector_d &permutedAdjIndexesIn, IdxVector_d &permutedAdjacencyIn);

   void fillPartitionLabel(IdxVector_d &coarseAggregate, IdxVector_d &fineAggregateSort, IdxVector_d &partitionLabel);

   void getAggregateStartIndices(IdxVector_d &fineAggregateSort, IdxVector_d &aggregateRemapIndex);

   void remapAggregateIdx(IdxVector_d &fineAggregateSort, IdxVector_d &aggregateRemapId);

   void mapAdjacencyToBlock(IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &adjacencyBlockLabel, IdxVector_d &blockMappedAdjacency, IdxVector_d &fineAggregate);

   void getInducedGraph(IdxVector_d &adjIndexesIn, IdxVector_d &adjacencyIn, IdxVector_d &partitionLabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut);

   void restrictPartitionSize(int maxSize, int fullSize, IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &partition, IdxVector_d &nodeWeights, bool verbose = false);

   void getWeightedPartSizes(IdxVector_d &partition, IdxVector_d &nodeWeights, IntVector_d &partSizes);

   void checkPartConnectivity(int partCount, IdxVector_d partition, IdxVector_d adjIndexes, IdxVector_d adjacency, char *message);

   void remapInducedGraph(IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &partition);

   namespace CP {
      namespace AT = AggMIS::Types;
      // Added for MIS paper tests:
      void OldMIS(IdxVector_d &adjIndexesIn,
                  IdxVector_d &adjacencyIn,
                  IdxVector_d &permutation,
                  IdxVector_d &ipermutation,
                  IdxVector_d &aggregateIdx,
                  IdxVector_d &partitionIdx,
                  IdxVector_d &partitionLabel,
                  IdxVector_d &adjIndexesOut,
                  IdxVector_d &adjacencyOut,
                  int parameters,
                  int part_max_size,
                  bool verbose = false);

      void MetisBottomUp(IdxVector_d &adjIndexesIn,
                         IdxVector_d &adjacencyIn,
                         IdxVector_d &permutation,
                         IdxVector_d &ipermutation,
                         IdxVector_d &aggregateIdx,
                         IdxVector_d &partitionIdx,
                         IdxVector_d &partitionLabel,
                         IdxVector_d &adjIndexesOut,
                         IdxVector_d &adjacencyOut,
                         int parameters,
                         int part_max_size,
                         bool verbose = false);

      void MetisTopDown(IdxVector_d &adjIndexesIn,
                        IdxVector_d &adjacencyIn,
                        IdxVector_d &permutation,
                        IdxVector_d &ipermutation,
                        IdxVector_d &aggregateIdx,
                        IdxVector_d &partitionIdx,
                        IdxVector_d &partitionLabel,
                        IdxVector_d &adjIndexesOut,
                        IdxVector_d &adjacencyOut,
                        int parameters,
                        int part_max_size,
                        bool verbose = false);

      void NewMIS(IdxVector_d &adjIndexesIn,
                  IdxVector_d &adjacencyIn,
                  IdxVector_d &permutation,
                  IdxVector_d &ipermutation,
                  IdxVector_d &aggregateIdx,
                  IdxVector_d &partitionIdx,
                  IdxVector_d &partitionLabel,
                  IdxVector_d &adjIndexesOut,
                  IdxVector_d &adjacencyOut,
                  int parameters,
                  int part_max_size,
                  bool verbose = false);
      void NewMIS_CPU(IdxVector_d &adjIndexesIn,
                      IdxVector_d &adjacencyIn,
                      IdxVector_d &permutation,
                      IdxVector_d &ipermutation,
                      IdxVector_d &aggregateIdx,
                      IdxVector_d &partitionIdx,
                      IdxVector_d &partitionLabel,
                      IdxVector_d &adjIndexesOut,
                      IdxVector_d &adjacencyOut,
                      int parameters,
                      int part_max_size,
                      bool verbose = false);
      void LightMIS_CPU(IdxVector_d &adjIndexesIn,
                        IdxVector_d &adjacencyIn,
                        IdxVector_d &permutation,
                        IdxVector_d &ipermutation,
                        IdxVector_d &aggregateIdx,
                        IdxVector_d &partitionIdx,
                        IdxVector_d &partitionLabel,
                        IdxVector_d &adjIndexesOut,
                        IdxVector_d &adjacencyOut,
                        int parameters,
                        int part_max_size,
                        bool verbose = false);
   }

   namespace Help {
      namespace AT = AggMIS::Types;
      int GetMetisAggregation(AT::IntVector_h &indices,
                              AT::IntVector_h &adjacency,
                              AT::IntVector_h &result,
                              int partSize,
                              bool verbose = false);
      int GetMetisAggregation_Large(AT::IntVector_h &indices,
                                    AT::IntVector_h &adjacency,
                                    AT::IntVector_h &result,
                                    int partSize,
                                    bool verbose = false);
      /**
       * This method takes a graph and an aggregation vector and produces the 
       * corresponding set of subgraphs along with a mapping of the nodes in
       * each subgraph.
       * @param indices Indices vector of input graph
       * @param adjacency Adjacency vector of input graph
       * @param partition Vector labeling each graph node with its partition
       * @param newIndices Vector of pointers to subgraph indices
       * @param newAdjacencies Vector of pointers to subgraph adjacencies
       */
      void GetSubGraphs(AT::IntVector_h &indices,
                        AT::IntVector_h &adjacency,
                        AT::IntVector_h &partition,
                        AT::IntVector_h_ptr &newIndices,
                        AT::IntVector_h_ptr &newAdjacencies,
                        AT::IntVector_h_ptr &nodeMaps,
                        bool verbose = false);
      int EnsureConnectedAndNonEmpty(AT::IntVector_h &indices,
                                     AT::IntVector_h &adjacency,
                                     AT::IntVector_h &aggregation);
      int BinarySearch(int value,
                       AT::IntVector_h &array);
      /**
       * This method records aggregation size, edge cut, and valence stats 
       * of the specified aggregation.
       * @param indices The indices vector of graph
       * @param adjacency The adjacency list of graph
       * @param aggregation The aggregation vector
       * @param prefix The prefix to distinguish this recording run
       */
      void RecordAllStats(IdxVector_d& indices,
                          IdxVector_d& adjacency,
                          IdxVector_d& aggregation,
                          std::string prefix);
      void RecordAllStats(IdxVector_d& indices,
                          IdxVector_d& adjacency,
                          IdxVector_d& aggregation,
                          IdxVector_d& nodeWeights,
                          std::string prefix);
      void RecordAggregationStats(IdxVector_d &aggregation,
                                  std::string prefix);
      void RecordAggregationStats(IdxVector_d &aggregation,
                                  IdxVector_d &nodeWeights,
                                  std::string prefix);
      void RecordValenceStats(IdxVector_d& indices,
                              IdxVector_d& adjacency,
                              std::string prefix);
      void RecordEdgeCut(IdxVector_d& indices,
                         IdxVector_d& adjacency,
                         IdxVector_d& aggregation,
                         std::string prefix);
   }
}
#endif
