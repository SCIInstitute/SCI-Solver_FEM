/*
 * File:   AggMIS_Aggregation_GPU.h
 * Author: T. James Lewis
 *
 * Created on April 19, 2013, 11:30 AM
 */

#ifndef AGGMIS_AGGREGATION_GPU_H
#define	AGGMIS_AGGREGATION_GPU_H
#include "AggMIS_Types.h"
#include "AggMIS_GraphHelpers.h"
#include "thrust/scan.h"
#include "thrust/count.h"
namespace AggMIS {
  namespace Aggregation {
    namespace Kernels {
      /**
       * In this kernel each node checks if it has been allocated
       * to an aggregate, if it has not, it checks to see if any
       * of its neighbors have been. If they have, it finds the
       * aggregate to which more neighbors belong to and assigns
       * itself to it.
       * @param size The number of nodes in the graph
       * @param adjIndexes The adjacency indices of the graph
       * @param adjacency The adjacency list of the graph
       * @param partIn The aggregation labeling from last cycle
       * @param partOut The aggregation labeling being found
       * @param aggregated Flags whether node has been aggregated
       */
      __global__ void allocateNodesKernel(int size,
        int *adjIndexes,
        int *adjacency,
        int *partIn,
        int *partOut,
        int *aggregated);
      /**
       * This kernel does the same flood filling that the allocate
       * kernel but only propogates between nodes of the same
       * aggregate. Used to verify that an aggregation consists of
       * aggregates which are connected components.
       * @param size Number of nodes in graph
       * @param adjIndices Graph adjacency indices
       * @param adjacency Graph adjacency
       * @param aggregation Current aggregation
       * @param valuesIn The values from last cycle
       * @param valuesOut The values to write this cycle
       * @param incomplete Flag which indicates whether done
       */
      __global__ void checkAggregationFillAggregates(int size,
        int *adjIndices,
        int *adjacency,
        int* aggregation,
        int* valuesIn,
        int* valuesOut,
        int* incomplete);
    }
    namespace Functors {
      /**
       * This functor is used to sequentially number elements
       * in a vector. Argument a is the element in the labeling
       * vector. Argument b is the element in the pre-fixed sum
       * vector of the labels. If the element is not labeled it
       * returns -1. Otherwise it returns the new label id
       */
      struct NumberParts {
        __host__ __device__
          int operator()(const int &a, const int &b) const
        {
          if (b == 0)
            return -1;

          return a - 1;
        }
      };
    }
    /**
     * This method allocates each node of the graph to the nearest
     * root node using simple path distance and breaking ties by
     * adjacency.
     * @param graph The input graph to be aggregated
     * @param roots The set of initial root points as a vector the size
     * of the number of graph nodes, with entries being either 0=non root
     * or 1=root
     * @return A vector with an entry for each graph node indicating which
     * zero indexed aggregate it was allocated to.
     */
    Types::IntVector_d* AggregateToNearest(Types::Graph_d &graph,
      Types::IntVector_d &roots);
    /**
     * Checks if the given labeling constitutes a valid aggregation
     * of the graph. Checks that the aggregate ID's form an
     * uninterrupted sequence starting from zero, and that each
     * aggregate is a connected component.
     * @param graph The graph
     * @param aggregation The node labeling to verify
     * @param verbose Prints more info if true
     * @return True if valid aggregation false otherwise
     */
    bool IsValidAggregation(Types::Graph_d &graph,
      Types::IntVector_d &aggregation,
      bool verbose);
    Types::Graph_d* GetAggregateMap(Types::IntVector_d& aggregation);
    //	double GetEdgeCutRatio(Graph_d &graph, IntVector_d& aggregation);
  }
}

#endif	/* AGGMIS_AGGREGATION_GPU_H */

