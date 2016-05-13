/*
 * File:   AggMIS_Aggregation_GPU.cu
 * Author: T. James Lewis
 *
 * Created on April 19, 2013, 11:30 AM
 */
#include "AggMIS_Aggregation_GPU.h"
#include "AggMIS_Types.h"
#include "AggMIS_GraphHelpers.h"
namespace AggMIS {
  namespace Aggregation {
    namespace Kernels {
      __global__ void allocateNodesKernel(int size,
        int *adjIndexes,
        int *adjacency,
        int *partIn,
        int *partOut,
        int *aggregated) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
          if (aggregated[idx] == 0)
          {
            int start = adjIndexes[idx];
            int end = adjIndexes[idx + 1];

            // Storage for possible aggregations.
            int candidates[10];
            int candidateCounts[10];
            for (int i = 0; i < 10; i++)
            {
              candidates[i] = -1;
              candidateCounts[i] = 0;
            }

            // Going through neighbors to aggregate:
            for (int i = start; i < end; i++)
            {
              int candidate = partIn[adjacency[i]];
              if (candidate != -1)
              {
                for (int j = 0; j < 10 && candidate != -1; j++)
                {
                  if (candidates[j] == -1)
                  {
                    candidates[j] = candidate;
                    candidateCounts[j] = 1;
                  } else
                  {
                    if (candidates[j] == candidate)
                    {
                      candidateCounts[j] += 1;
                      candidate = -1;
                    }
                  }
                }
              }
            }

            // Finding the most adjacent aggregate and adding node to it:
            int addTo = candidates[0];
            int count = candidateCounts[0];
            for (int i = 1; i < 10; i++)
            {
              if (candidateCounts[i] > count)
              {
                count = candidateCounts[i];
                addTo = candidates[i];
              }
            }
            partOut[idx] = addTo;
            if (addTo != -1)
            {
              aggregated[idx] = 1;
            }
          }
        }
      }
      __global__ void checkAggregationFillAggregates(int size,
        int *adjIndices,
        int *adjacency,
        int* aggregation,
        int* valuesIn,
        int* valuesOut,
        int* incomplete) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
          // Find the currently marked distance
          int currentVal = valuesIn[idx];
          int currentAgg = aggregation[idx];

          // Checking if any neighbors have a better value
          int start = adjIndices[idx];
          int end = adjIndices[idx + 1];
          for (int i = start; i < end; i++)
          {
            int neighborAgg = aggregation[adjacency[i]];
            int neighborVal = valuesIn[adjacency[i]];
            if (neighborAgg == currentAgg && neighborVal > currentVal)
            {
              currentVal = neighborVal;
              incomplete[0] = 1;
            }
          }

          // Write out the distance to the output vector:
          valuesOut[idx] = currentVal;
        }
      }
    }

    // Functions
    AggMIS::Types::IntVector_d* AggregateToNearest(AggMIS::Types::Graph_d &graph,
      AggMIS::Types::IntVector_d &roots) {
      // Create temp vectors to work with
      int size = graph.Size();
      AggMIS::Types::IntVector_d *aggregated = new AggMIS::Types::IntVector_d(roots);
      AggMIS::Types::IntVector_d *partIn = new AggMIS::Types::IntVector_d(roots);


      // Prefix sum to number aggregate roots:
      thrust::inclusive_scan(partIn->begin(), partIn->end(), partIn->begin());

      // Transform non root nodes to -1
      thrust::transform(partIn->begin(), partIn->end(), aggregated->begin(), partIn->begin(), Functors::NumberParts());
      AggMIS::Types::IntVector_d *partOut = new AggMIS::Types::IntVector_d(*partIn);

      // Preparing to call aggregate kernel:   
      int *partIn_d = thrust::raw_pointer_cast(partIn->data());               // Pointer to partIn vector
      int *partOut_d = thrust::raw_pointer_cast(partOut->data());             // Pointer to partOut vector
      int *adjIndexes_d = thrust::raw_pointer_cast(graph.indices->data()); // Pointer to adjacency indexes
      int *adjacency_d = thrust::raw_pointer_cast(graph.adjacency->data());   // Pointer to adjacency
      int *aggregated_d = thrust::raw_pointer_cast(aggregated->data());       // Pointer to aggregated
      bool complete = false;      // Indicates whether all nodes are aggregated

      // Figuring out block sizes for kernel call:
      int blockSize = 256;
      int nBlocks = size / blockSize + (size%blockSize == 0 ? 0 : 1);

      while (!complete)
      {
        // Allocating nodes
        Kernels::allocateNodesKernel << < nBlocks, blockSize >> > (size, adjIndexes_d, adjacency_d, partIn_d, partOut_d, aggregated_d);

        // Copying partOut to partIn
        thrust::copy(partOut->begin(), partOut->end(), partIn->begin());

        // Checking if done
        int unallocatedNodes = thrust::count(aggregated->begin(), aggregated->end(), 0);
        complete = unallocatedNodes == 0;
      }

      // Cleaning up
      aggregated->clear();
      partOut->clear();
      delete aggregated;
      delete partOut;

      return partIn;
    }
    bool IsValidAggregation(AggMIS::Types::Graph_d &graph,
      AggMIS::Types::IntVector_d &aggregation,
      bool verbose) {
      // Counter for number of errors found
      int errors = 0;

      // Check to make sure that the aggregate id's are sequential
      AggMIS::Types::IntVector_d scratch(aggregation);
      thrust::sort(scratch.begin(), scratch.end());
      int newLength = thrust::unique(scratch.begin(), scratch.end()) - scratch.begin();
      scratch.resize(newLength);

      if (scratch[0] != 0 || scratch[scratch.size() - 1] != scratch.size() - 1)
      {
        if (verbose) {
          printf("Error found in aggregation: improper aggregate indices:\n");
          int firstId = scratch[0];
          int lastId = scratch[scratch.size() - 1];
          int count = scratch.size();
          printf("\tFirst index is %d, last index is %d, there are %d unique id's\n", firstId, lastId, count);
        }
        errors++;
        return false;
      }

      // Check to make sure each aggregate is a connected component
      AggMIS::Types::IntVector_d *valuesIn = GraphHelpers::GetIndicesVector(aggregation.size());
      AggMIS::Types::IntVector_d valuesOut(aggregation.size());
      AggMIS::Types::IntVector_d incomplete(1, 1);

      // Figuring out block sizes for kernel call:
      int size = graph.Size();
      int blockSize = 256;
      int nBlocks = size / blockSize + (size%blockSize == 0 ? 0 : 1);

      // Getting raw pointers
      int *valuesIn_d = thrust::raw_pointer_cast(valuesIn->data());
      int *valuesOut_d = thrust::raw_pointer_cast(&valuesOut[0]);
      int *incomplete_d = thrust::raw_pointer_cast(&incomplete[0]);
      int *adjacency_d = thrust::raw_pointer_cast(graph.adjacency->data());
      int *adjIndices_d = thrust::raw_pointer_cast(graph.indices->data());
      int *aggregation_d = thrust::raw_pointer_cast(&aggregation[0]);

      // Flood filling within each aggregate
      int *originalOut = valuesIn_d;
      while (incomplete[0] == 1)
      {
        incomplete[0] = 0;
        Kernels::checkAggregationFillAggregates << < nBlocks, blockSize >> >
          (size, adjIndices_d, adjacency_d, aggregation_d, valuesIn_d, valuesOut_d, incomplete_d);
        int *temp = valuesIn_d;
        valuesIn_d = valuesOut_d;
        valuesOut_d = temp;
      }

      if (originalOut != valuesOut_d)
        valuesOut.assign(valuesIn->begin(), valuesIn->end());
      valuesIn->assign(aggregation.begin(), aggregation.end());

      // 
      int correctLength = newLength;
      thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(valuesIn->begin(), valuesOut.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(valuesIn->end(), valuesOut.end())));
      newLength = thrust::unique(thrust::make_zip_iterator(thrust::make_tuple(valuesOut.begin(), valuesIn->begin())),
        thrust::make_zip_iterator(thrust::make_tuple(valuesOut.end(), valuesIn->end())))
        - thrust::make_zip_iterator(thrust::make_tuple(valuesOut.begin(), valuesIn->begin()));

      valuesIn->resize(newLength);
      valuesOut.resize(newLength);

      if (newLength != correctLength)
      {
        if (verbose)
          printf("Error: there were %d connected components found and %d aggregates\n", newLength, correctLength);
        errors++;

        AggMIS::Types::IntVector_h aggIds(*valuesIn);
        AggMIS::Types::IntVector_h nodeIds(valuesOut);
        for (int i = 0; i < valuesOut.size() - 1; i++)
        {
          int currentAgg = aggIds[i];
          int nextAgg = aggIds[i + 1];
          if (currentAgg == nextAgg && verbose)
            printf("Aggregate %d was filled from %d and %d\n", currentAgg, nodeIds[i], nodeIds[i + 1]);
        }
      }

      // Clean up 
      scratch.resize(0);
      valuesIn->resize(0);
      delete valuesIn;
      incomplete.resize(0);

      return errors == 0;
    }
    AggMIS::Types::Graph_d* GetAggregateMap(AggMIS::Types::IntVector_d& aggregation) {
      AggMIS::Types::Graph_d* output = new AggMIS::Types::Graph_d();
      // Setting adjacency of output to be indices 
      GraphHelpers::SetToIndicesVector(aggregation.size(), *(output->adjacency));
      AggMIS::Types::IntVector_d aggLabels(aggregation.begin(), aggregation.end());

      // Sorting by key to get node id's grouped by aggregates
      thrust::sort_by_key(aggLabels.begin(), aggLabels.end(), output->adjacency->begin());

      // Resizing the indices to aggregate count
      int maxAggregate = aggLabels[aggLabels.size() - 1];
      output->indices->resize(maxAggregate + 2, 0);

      // Figuring out block sizes for kernel call:
      int size = aggregation.size();
      int blockSize = 256;
      int nBlocks = size / blockSize + (size%blockSize == 0 ? 0 : 1);

      // Calling kernel to find indices for each part:
      GraphHelpers::Kernels::findPartIndicesKernel << < nBlocks, blockSize >> >
        (size,
        AggMIS::Types::StartOf(aggLabels),
        output->indStart());

      // Cleaning up
      aggLabels.clear();

      return output;
    }
  }
}
