/*
 * File:   AggMIS_MergeSplitConditioner.h
 * Author: T. James Lewis
 *
 * Created on April 15, 2013, 2:13 PM
 */
#ifndef AGGMIS_MERGESPLITCONDITIONER_H
#define	AGGMIS_MERGESPLITCONDITIONER_H
#include <string>
#include "AggMIS_Types.h"
#include "AggMIS_Aggregation_GPU.h"
#include "AggMIS_GraphHelpers.h"
#include "AggMIS_IOHelpers.h"
#include "thrust/count.h"
#include "thrust/iterator/constant_iterator.h"
#include "thrust/scan.h"

namespace AggMIS {
  namespace MergeSplitGPU {

    // GPU Kernels
    namespace Kernels {
      namespace T {
        template <typename T>
        struct AddTo {
          const T a;
          __device__ AddTo(T _a) :a(_a){};
          __device__ T operator() (const T &b) const {
            return a + b;
          }
        };
        template <typename T, typename F>
        __device__ void Transform(int size, T* array, F f) {
          if (threadIdx.x < size)
            array[threadIdx.x] = f(array[threadIdx.x]);
          __syncthreads();
        }
        //                template <typename T, typename F>
        //                __device__ T Reduce(int size, T* input, T* scratch, )
      }
      // Device helper functions
      namespace D {
        /**
         * Loads the in aggregate neighbors of the current node into an array,
         * translating them into local indices.
         * @param neighbors Pointer to array to place neighbor Ids
         * @param nextNeighbor Will contain the index of the element past
         * the last valid neighbor in the array.
         * @param aggSize The count of nodes in the aggregate
         * @param nodeIds The sorted list of nodes in the aggregate
         * @param adjIndices The graph indices
         * @param adjacency The graph adjacency
         */
        __device__ void LoadLocalNeighbors(int *neighbors,
          int *nextNeighbor,
          int aggSize,
          int *nodeIds,
          int *adjIndices,
          int *adjacency);
        /**
         * Performs a binary search returning the index of the given
         * value, or -1 if not found.
         * @param value The value to search for.
         * @param imin The index beginning the search range.
         * @param imax The index after the end of the search range.
         * @param array Pointer to array to be searched
         * @return The index of the given value, or -1 if not found.
         */
        __device__ int BinarySearch(int value,
          int imin,
          int imax,
          int *array);
        /**
         * This method flood fills distances from the specified node
         * within the subgraph.
         * @param starter The node to count distances from
         * @param array The array to mark the distances of each node
         * @param nodeCount The number of nodes in subgraph
         * @param neighbors The array of neighbors for the current
         * thread's node
         * @param neighborCount The count of neighbors of the current
         * thread's node
         * @param farthestId After the flood fill completes this will
         * contain the ID of a node with maximal distance.
         * @param incomplete Shared boolean flag to use within method
         */
        __device__ void FloodFillDistanceFrom(int starter,
          int* array,
          int nodeCount,
          int *neighbors,
          int neighborCount,
          int *farthestId,
          bool *incomplete);
        __device__ void PrintSharedArray(int size,
          int *array,
          const char *note);
        __device__ void WarpReport(const char* note);
        __device__ void SillyTest();
      }
      __global__ void MakeMerges(int size,
        int *mergeWith,
        int *offsets,
        int *mis);
      __global__ void MakeMerges_MarkSplits(int size,
        int *mergeWith,
        int *offsets,
        int *mis,
        int *splitsToMake);
      __global__ void MakeSplits(int baseAggregateIdx,
        int *splitting,
        int *aggregation,
        int *aggMapAdjIndices,
        int *aggMapAdjacency,
        int *adjIndices,
        int *adjacency);
      __global__ void MakeSplitsWeighted(int baseAggregateIdx,
        int* splitting,
        int* aggregation,
        int* aggMapAdjIndices,
        int* aggMapAdjacency,
        int* adjIndices,
        int* adjacency,
        int* weights);
      __global__ void MakeSplits_Large(int baseAggregateIdx,
        int *splitting,
        int *aggregation,
        int *aggMapAdjIndices,
        int *aggMapAdjacency,
        int *adjIndices,
        int *adjacency);
      __global__ void MakeSplitsWeighted_Large(int baseAggregateIdx,
        int* splitting,
        int* aggregation,
        int* aggMapAdjIndices,
        int* aggMapAdjacency,
        int* adjIndices,
        int* adjacency,
        int* weights);
      __global__ void MarkSplits(int size,
        bool force,
        int minPartSize,
        int maxPartSize,
        int *partSizes,
        int *splitsToMake);
      __global__ void FindDesirableMerges(int size,
        int minSize,
        int maxSize,
        bool force,
        int* adjIndices,
        int* adjacency,
        int *partSizes,
        int* desiredMerges,
        int* merging);
      __global__ void FindDesirableMergeSplits(int size,
        int minSize,
        int maxSize,
        int desiredSize,
        int* adjIndices,
        int* adjacency,
        int* partSizes,
        int* desiredMerges,
        int* merging);
      __global__ void MarkMerges(int size,
        int* desiredMerges,
        int* merging,
        int* mergesToMake,
        int* incomplete);
    }

    // Functors for Thrust calls
    namespace Functors {
      struct isOutSized :public thrust::unary_function<int, int> {
        const int a, b;
        isOutSized(int _a, int _b) :a(_a), b(_b){}

        __host__ __device__
          bool operator()(const int &x) const
        {
          return x < a || x > b;
        }
      };
      struct lessThan :public thrust::unary_function<int, int> {
        const int a;

        lessThan(int _a) : a(_a){}

        __host__ __device__
          int operator()(const int &x) const
        {
          return x < a;
        }
      };
      struct greaterThan :public thrust::unary_function<int, int> {
        const int a;

        greaterThan(int _a) : a(_a){}

        __host__ __device__
          int operator()(const int &x) const
        {
          return x > a;
        }
      };
      struct NotNegOne :public thrust::unary_function<int, int> {
        __host__ __device__
          int operator()(int a) const
        {
          // If value is negative return 0 else return one
          return a < 0 ? 0 : 1;
        }
      };
      struct EqualTo :public thrust::unary_function<int, int> {
        const int a;

        EqualTo(int _a) : a(_a){}

        __host__ __device__
          int operator()(const int &x) const
        {
          return x == a;
        }
      };
      struct SquaredDifference :public thrust::unary_function<int, double> {
        const int a;
        SquaredDifference(double _a) :a(_a){}
        __host__ __device__
          double operator()(const int &x) const {
          return (a - x)*(a - x);
        }
      };
      struct SplitFilter {
        const int a;
        const int b;
        SplitFilter(int _a, int _b) :a(_a), b(_b){}
        __host__ __device__
          bool operator()(const thrust::tuple<int, int> &x) const {
          return thrust::get<0>(x) == 1 &&
            thrust::get<1>(x) >= a &&
            thrust::get<1>(x) <= b;
        }
      };
    }

    // Merge and Split conditioner main container
    class MergeSplitConditionerGPU {
    public:
      /**
      * The primary constructor it creates a conditioner given the
      * specified graph and aggregation. It makes an internal copy of
      * the aggregation vector, but it uses a pointer back to the given
      * Graph object
      * @param graph The graph that the aggregation being conditioned
      * is an aggregation of.
      * @param aggregation A vector which labels every node in the graph
      * with an aggregate ID.
      */
      MergeSplitConditionerGPU(Types::Graph_d &graph,
        Types::IntVector_d &aggregation);
      void SetSizeBounds(int min, int max);
      void SetVerbose(bool v);
      /**
       * Sets the node weights by swapping the contents of the provided
       * vector into the nodeWeights member. Then it re-computes the part
       * sizes with weighting.
       * @param nodeWeights A vector containing the weights of each node
       * the contents of this vector are swapped out by the method.
       */
      void SetNodeWeights(Types::IntVector_d &nodeWeights);
      /**
       * Getter for the aggregation vector.
       * @return A pointer to the current aggregation vector
       */
      Types::IntVector_d* GetAggregation();
      /**
       * Getter for the NodeWeights vector
       * @return A pointer to the current NodeWeights vector
       */
      Types::IntVector_d* GetNodeWeights();
      void CycleMerges(bool force);
      void CycleSplits(bool force);
      void CycleMergeSplits(float minImprove,
        int desiredSize);
      bool Condition(int desiredSize,
        bool respectUpper,
        float tolerance,
        float minImprove,
        int maxCycles);
      void PrintProgress(std::ostream* output,
        std::string note,
        bool graphStat,
        bool progressStat,
        bool sizeStat,
        bool memStat);
      void PrintSizeStats(std::ostream* output,
        bool makeHeader);
      void PrintMemoryStats(std::ostream* output,
        bool makeHeader);
      void PrintProgressStats(std::ostream* output,
        bool makeHeader);
      void PrintGraphStats(std::ostream* output,
        bool makeHeader);
      void InteractiveConsole(std::string message);
    private:
      bool MarkMerges(bool force);
      bool MarkSplits(bool force);
      bool MarkMergeSplits(int desiredSize);
      void MakeSplits();
      void MakeMerges(bool markSplits);
      bool MakeMergeSplits(int desiredSize);

      // Data members
      Types::Graph_d *graph;
      Types::Graph_d *inducedGraph;

      int minSize,
        maxSize,
        outsizedParts,
        merges,
        mergeSplits,
        splits;

      Types::IntVector_d aggregation,
        nodeWeights,
        partSizes,
        weightedSizes,
        splitsToMake,
        mergesToMake,
        mergeOffsets;

      bool verbose;
    };
  }
}


#endif	/* AGGMIS_MERGESPLITCONDITIONER_H */

