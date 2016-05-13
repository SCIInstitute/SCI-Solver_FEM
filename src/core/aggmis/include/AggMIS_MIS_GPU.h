/*
 * File:   AggMIS_MIS_GPU.h
 * Author: T. James Lewis
 *
 * Created on April 17, 2013, 12:49 PM
 */

#ifndef AGGMIS_MIS_GPU_H
#define	AGGMIS_MIS_GPU_H
#include "AggMIS_Types.h"
namespace AggMIS {
  namespace MIS {
    namespace Kernels {
      __global__ void GenerateRandoms(int size,
        int iterations,
        unsigned int *randoms,
        unsigned int *seeds);
      __global__ void PreInitialize(int size,
        unsigned int *randoms,
        int *bestSeen,
        int *origin,
        int *mis);
      __global__ void Initialize(int size,
        unsigned int *randoms,
        int *bestSeen,
        int *origin,
        int *mis,
        int *incomplete);
      __global__ void Iterate(int size,
        int *originIn,
        int *originOut,
        int *bestSeenIn,
        int *bestSeenOut,
        int *adjIndexes,
        int *adjacency);
      __global__ void Finalize(int size,
        int *originIn,
        int *originOut,
        int *bestSeenIn,
        int *bestSeenOut,
        int *adjIndexes,
        int *adjacency,
        int *mis,
        int *incomplete);
    }
    Types::IntVector_d* RandomizedMIS(int k, Types::Graph_d &graph);
    bool IsValidKMIS(Types::IntVector_d &misIn, 
      Types::Graph_d &graph, int k, bool verbose);
  }
}


#endif	/* AGGMIS_MIS_GPU_H */

