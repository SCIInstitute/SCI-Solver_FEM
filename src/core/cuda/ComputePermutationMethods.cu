#include <smoothedMG/aggregators/misHelpers.h>
extern "C" {
#include "metis.h"
}
#include "AggMIS_Types.h"
#include "AggMIS_Aggregation_GPU.h"
#include "AggMIS_MIS_GPU.h"
#include "AggMIS_MIS_CPU.h"
#include "AggMIS_MergeSplitConditioner.h"
#include "AggMIS_MergeSplitConditioner_CPU.h"
#include "AggMIS_GraphHelpers.h"
#include <smoothedMG/aggregators/Timer.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "Logger.h"
namespace misHelpers {
   double totalAggregationTime = 0;
   int totalAggregationCalls = 0;
   namespace CP {
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
            bool verbose) {

         int numNodesIn = adjIndexesIn.size() - 1; // Size of input graph
         int fineDepth = parameters % 100; // The MIS depth for the first aggregation
         int coarseDepth = (parameters / 100) % 100; // The MIS depth for the second aggregation
         int minAggregateSize = (parameters / 10000) % 10; // The minimum acceptable size for an aggregate

         IdxVector_d fineAggregate(numNodesIn, 0); // The partition label for the fine partition
         IdxVector_d fineAggregateSort; // The copy of the fine partition label that is sorted
         IdxVector_d finePartSizes; // Vector with sizes of fine partitions
         IdxVector_d neighborCountsIn; // Vector to hold the sizes for each nodes adjacency
         IdxVector_d permutedAdjIndexesIn; // Vector to hold the indices for the initial adjacency permutation
         IdxVector_d permutedAdjacencyIn; // Holds the permuted initial adjacency
         IdxVector_d neighborCountsOut; // Holds the counts of neighbors for the induced graph
         IdxVector_d coarseAggregate; // Holds the partition label for the coarse partition
         IdxVector_d aggregateRemapId; // Holds the current id of each aggregate
         IdxVector_d iAggregateRemapId; // Holds the permutation to remap the aggregate id'
         IdxVector_d aggregateRemapIndex; // Holds the start index of each aggregate
         IdxVector_d inducedNodeWeights; // Holds the sizes of the fine aggregates
         AggMIS::Types::JTimer timmy;
         if (verbose)
            std::cout << "Finished initializing IdxVector_d elements." << std::endl;

         partitionLabel = IdxVector_d(numNodesIn, 2); // Holds the partition each vertex is located in
         if (verbose)
            std::cout << "Finished with partitionLabel creation." << std::endl;
         timmy.start();

         misHelpers::aggregateGraph(minAggregateSize, fineDepth, adjIndexesIn, adjacencyIn, fineAggregate, verbose);
         if (verbose)
            std::cout << "Finished with aggregateGraph." << std::endl;
         timmy.stop();
         totalAggregationTime += timmy.getElapsedTimeInSec(true);
         if (verbose)
            printf("Fine conditioning time: %3.3fs\n", timmy.getElapsedTimeInSec(true));
         totalAggregationCalls++;

         Help::RecordAllStats(adjIndexesIn, adjacencyIn, fineAggregate, "Fine Aggregation");

         // Setting the permutation array to have values equal to element indices
         permutation = IdxVector_d(numNodesIn);
         misHelpers::fillWithIndex(permutation);

         // Sorting arrays together:
         fineAggregateSort = fineAggregate;
         thrust::sort_by_key(fineAggregateSort.begin(), fineAggregateSort.end(), permutation.begin());
         if (verbose)
            std::cout << "Finished with fineAggregateSort." << std::endl;

         // Building the permutation array:
         misHelpers::getInversePermutation(permutation, ipermutation);
         if (verbose)
            std::cout << "Got permutation array." << std::endl;

         // Getting the aggregate indices and node weights for the induced graph
         misHelpers::getPartSizes(fineAggregateSort, inducedNodeWeights, aggregateIdx);
         if (verbose)
            std::cout << "Got partition sizes." << std::endl;

         // Getting the induced graph:
         misHelpers::getInducedGraph(adjIndexesIn, adjacencyIn, fineAggregate, adjIndexesOut, adjacencyOut);
         if (verbose)
            std::cout << "Got induced graph." << std::endl;

         // Doing the coarse aggregation:
         int maxSize = part_max_size;
         int fullSize = adjIndexesIn.size() - 1;
         coarseAggregate = IdxVector_d(fullSize, 1);
         int inducedGraphSize = adjIndexesOut.size() - 1;

         timmy.start();
         misHelpers::aggregateWeightedGraph(maxSize, fullSize, coarseDepth, adjIndexesOut, adjacencyOut, coarseAggregate, inducedNodeWeights, verbose);
         timmy.stop();
         if (verbose)
            std::cout << "Finished aggregateWeightedGraph." << std::endl;

         Help::RecordAllStats(adjIndexesOut,
               adjacencyOut,
               coarseAggregate,
               inducedNodeWeights,
               "Coarse Aggregation");

         // Performing new version of getting induced graph
         misHelpers::remapInducedGraph(adjIndexesOut, adjacencyOut, coarseAggregate);
         if (verbose)
            std::cout << "Finished remapInducedGraph." << std::endl;

         // Filling in the partitionLabel:
         misHelpers::fillPartitionLabel(coarseAggregate, fineAggregateSort, partitionLabel);
         if (verbose)
            std::cout << "Finished fillPartitionLabel." << std::endl;

         // Do a stable sort by key with the partitionLabel as the key:
         thrust::stable_sort_by_key(partitionLabel.begin(), partitionLabel.end(), thrust::make_zip_iterator(thrust::make_tuple(fineAggregateSort.begin(), permutation.begin())));
         if (verbose)
            std::cout << "Finished thrust::stable_sort_by_key." << std::endl;

         // Remapping the aggregate id's:
         aggregateRemapId = IdxVector_d(aggregateIdx.size() - 1, 0);
         aggregateRemapIndex = IdxVector_d(aggregateIdx.size() - 1, 0);
         misHelpers::fillWithIndex(aggregateRemapId);
         misHelpers::getAggregateStartIndices(fineAggregateSort, aggregateRemapIndex);
         thrust::stable_sort_by_key(aggregateRemapIndex.begin(), aggregateRemapIndex.end(), aggregateRemapId.begin());
         misHelpers::getInversePermutation(aggregateRemapId, iAggregateRemapId);
         misHelpers::remapAggregateIdx(fineAggregateSort, iAggregateRemapId);
         misHelpers::remapAggregateIdx(fineAggregate, iAggregateRemapId);

         // Sort the coarseAggregate for indices and permutation:
         thrust::sort(coarseAggregate.begin(), coarseAggregate.end());
         misHelpers::getPartIndices(coarseAggregate, partitionIdx);

         // Get indices for the fine aggregates
         misHelpers::getPartIndices(fineAggregateSort, aggregateIdx);

         // Putting in the right permutation vectors for the output:
         ipermutation = permutation;
         misHelpers::getInversePermutation(ipermutation, permutation);
      }
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
            bool verbose) {
         int numNodesIn = adjIndexesIn.size() - 1; // Size of input graph

         IdxVector_d fineAggregate(numNodesIn, 0); // The partition label for the fine partition
         IdxVector_d fineAggregateSort; // The copy of the fine partition label that is sorted
         IdxVector_d finePartSizes; // Vector with sizes of fine partitions
         IdxVector_d neighborCountsIn; // Vector to hold the sizes for each nodes adjacency
         IdxVector_d permutedAdjIndexesIn; // Vector to hold the indices for the initial adjacency permutation
         IdxVector_d permutedAdjacencyIn; // Holds the permuted initial adjacency
         IdxVector_d neighborCountsOut; // Holds the counts of neighbors for the induced graph
         IdxVector_d coarseAggregate; // Holds the partition label for the coarse partition
         IdxVector_d aggregateRemapId; // Holds the current id of each aggregate
         IdxVector_d iAggregateRemapId; // Holds the permutation to remap the aggregate id'
         IdxVector_d aggregateRemapIndex; // Holds the start index of each aggregate
         IdxVector_d inducedNodeWeights; // Holds the sizes of the fine aggregates

         partitionLabel = IdxVector_d(numNodesIn, 2); // Holds the partition each vertex is located in
         int fineSize, coarseSize;
         coarseSize = part_max_size % 1000;
         fineSize = (part_max_size / 1000) % 1000;


         // Getting the fine aggregation with Metis
         AT::IntVector_h indices(adjIndexesIn.size());
         thrust::copy(adjIndexesIn.begin(), adjIndexesIn.end(), indices.begin());
         AT::IntVector_h adjacency(adjacencyIn.size());
         thrust::copy(adjacencyIn.begin(), adjacencyIn.end(), adjacency.begin());
         AT::IntVector_h result(numNodesIn);
         Help::GetMetisAggregation(indices, adjacency, result, fineSize);
         thrust::copy(result.begin(), result.end(), fineAggregate.begin());

         Help::RecordAllStats(adjIndexesIn,
               adjacencyIn,
               fineAggregate,
               "Fine Aggregation");

         // Setting the permutation array to have values equal to element indices
         permutation = IdxVector_d(numNodesIn);
         misHelpers::fillWithIndex(permutation);

         // Sorting arrays together:
         fineAggregateSort = fineAggregate;
         thrust::sort_by_key(fineAggregateSort.begin(), fineAggregateSort.end(), permutation.begin());

         // Building the permutation array:
         misHelpers::getInversePermutation(permutation, ipermutation);

         // Getting the aggregate indices and node weights for the induced graph
         misHelpers::getPartSizes(fineAggregateSort, inducedNodeWeights, aggregateIdx);

         // Getting the induced graph:
         misHelpers::getInducedGraph(adjIndexesIn, adjacencyIn, fineAggregate, adjIndexesOut, adjacencyOut);

         int inducedGraphSize = adjIndexesOut.size() - 1;

         // Doing the coarse aggregation (assuming Metis parts are close enough in size to ignore weighting)
         indices.assign(adjIndexesOut.begin(), adjIndexesOut.end());
         adjacency.assign(adjacencyOut.begin(), adjacencyOut.end());
         Help::GetMetisAggregation(indices, adjacency, result, coarseSize);
         coarseAggregate.assign(result.begin(), result.end());

         Help::RecordAllStats(adjIndexesOut,
               adjacencyOut,
               coarseAggregate,
               inducedNodeWeights,
               "Coarse Aggregation");

         // Performing new version of getting induced graph
         misHelpers::remapInducedGraph(adjIndexesOut, adjacencyOut, coarseAggregate);

         // Filling in the partitionLabel:
         misHelpers::fillPartitionLabel(coarseAggregate, fineAggregateSort, partitionLabel);

         // Do a stable sort by key with the partitionLabel as the key:
         thrust::stable_sort_by_key(partitionLabel.begin(), partitionLabel.end(), thrust::make_zip_iterator(thrust::make_tuple(fineAggregateSort.begin(), permutation.begin())));

         // Remapping the aggregate id's:
         aggregateRemapId = IdxVector_d(aggregateIdx.size() - 1, 0);
         aggregateRemapIndex = IdxVector_d(aggregateIdx.size() - 1, 0);
         misHelpers::fillWithIndex(aggregateRemapId);
         misHelpers::getAggregateStartIndices(fineAggregateSort, aggregateRemapIndex);
         thrust::stable_sort_by_key(aggregateRemapIndex.begin(), aggregateRemapIndex.end(), aggregateRemapId.begin());
         misHelpers::getInversePermutation(aggregateRemapId, iAggregateRemapId);
         misHelpers::remapAggregateIdx(fineAggregateSort, iAggregateRemapId);
         misHelpers::remapAggregateIdx(fineAggregate, iAggregateRemapId);

         // Sort the coarseAggregate for indices and permutation:
         thrust::sort(coarseAggregate.begin(), coarseAggregate.end());
         misHelpers::getPartIndices(coarseAggregate, partitionIdx);

         // Get indices for the fine aggregates
         misHelpers::getPartIndices(fineAggregateSort, aggregateIdx);

         // Putting in the right permutation vectors for the output:
         ipermutation = permutation;
         misHelpers::getInversePermutation(ipermutation, permutation);

         // Clean up temp vectors
         indices.clear();
         adjacency.clear();
         result.clear();
         if (verbose)
            printf("Total aggregation time (Metis): %3.4fs for %d calls\n", totalAggregationTime, totalAggregationCalls);
      }
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
            bool verbose) {
         int numNodesIn = adjIndexesIn.size() - 1; // Size of input graph
         int fineDepth = parameters % 100; // The MIS depth for the first aggregation
         int coarseDepth = (parameters / 100) % 100; // The MIS depth for the second aggregation
         int minAggregateSize = (parameters / 10000) % 10; // The minimum acceptable size for an aggregate
         IdxVector_d fineAggregate(numNodesIn, 0); // The partition label for the fine partition
         IdxVector_d fineAggregateSort; // The copy of the fine partition label that is sorted
         IdxVector_d finePartSizes; // Vector with sizes of fine partitions
         IdxVector_d neighborCountsIn; // Vector to hold the sizes for each nodes adjacency
         IdxVector_d permutedAdjIndexesIn; // Vector to hold the indices for the initial adjacency permutation
         IdxVector_d permutedAdjacencyIn; // Holds the permuted initial adjacency
         IdxVector_d neighborCountsOut; // Holds the counts of neighbors for the induced graph
         IdxVector_d coarseAggregate; // Holds the partition label for the coarse partition
         IdxVector_d aggregateRemapId; // Holds the current id of each aggregate
         IdxVector_d iAggregateRemapId; // Holds the permutation to remap the aggregate id'
         IdxVector_d aggregateRemapIndex; // Holds the start index of each aggregate
         IdxVector_d inducedNodeWeights; // Holds the sizes of the fine aggregates

         partitionLabel = IdxVector_d(numNodesIn, 2); // Holds the partition each vertex is located in
         misHelpers::aggregateGraph(minAggregateSize, fineDepth, adjIndexesIn, adjacencyIn, fineAggregate, verbose);

         // Setting the permutation array to have values equal to element indices
         permutation = IdxVector_d(numNodesIn);
         misHelpers::fillWithIndex(permutation);

         // Sorting arrays together:
         fineAggregateSort = fineAggregate;
         thrust::sort_by_key(fineAggregateSort.begin(), fineAggregateSort.end(), permutation.begin());

         // Building the permutation array:
         misHelpers::getInversePermutation(permutation, ipermutation);

         // Getting the aggregate indices and node weights for the induced graph
         misHelpers::getPartSizes(fineAggregateSort, inducedNodeWeights, aggregateIdx);

         // Getting the induced graph:
         misHelpers::getInducedGraph(adjIndexesIn, adjacencyIn, fineAggregate, adjIndexesOut, adjacencyOut);

         // Doing the coarse aggregation:
         int maxSize = part_max_size; //400;
         int fullSize = adjIndexesIn.size() - 1;
         coarseAggregate = IdxVector_d(fullSize, 1);
         misHelpers::aggregateWeightedGraph(maxSize, fullSize, coarseDepth, adjIndexesOut, adjacencyOut, coarseAggregate, inducedNodeWeights, verbose);

         // Performing new version of getting induced graph
         misHelpers::remapInducedGraph(adjIndexesOut, adjacencyOut, coarseAggregate);

         // Filling in the partitionLabel:
         misHelpers::fillPartitionLabel(coarseAggregate, fineAggregateSort, partitionLabel);

         // Do a stable sort by key with the partitionLabel as the key:
         thrust::stable_sort_by_key(partitionLabel.begin(), partitionLabel.end(), thrust::make_zip_iterator(thrust::make_tuple(fineAggregateSort.begin(), permutation.begin())));

         // Remapping the aggregate id's:
         aggregateRemapId = IdxVector_d(aggregateIdx.size() - 1, 0);
         aggregateRemapIndex = IdxVector_d(aggregateIdx.size() - 1, 0);
         misHelpers::fillWithIndex(aggregateRemapId);
         misHelpers::getAggregateStartIndices(fineAggregateSort, aggregateRemapIndex);
         thrust::stable_sort_by_key(aggregateRemapIndex.begin(), aggregateRemapIndex.end(), aggregateRemapId.begin());
         misHelpers::getInversePermutation(aggregateRemapId, iAggregateRemapId);
         misHelpers::remapAggregateIdx(fineAggregateSort, iAggregateRemapId);
         misHelpers::remapAggregateIdx(fineAggregate, iAggregateRemapId);

         // Sort the coarseAggregate for indices and permutation:
         thrust::sort(coarseAggregate.begin(), coarseAggregate.end());
         misHelpers::getPartIndices(coarseAggregate, partitionIdx);

         // Get indices for the fine aggregates
         misHelpers::getPartIndices(fineAggregateSort, aggregateIdx);

         // Putting in the right permutation vectors for the output:
         ipermutation = permutation;
         misHelpers::getInversePermutation(ipermutation, permutation);
      }
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
            bool verbose) {
         int numNodesIn = adjIndexesIn.size() - 1; // Size of input graph
         int fineDepth = parameters % 100; // The MIS depth for the first aggregation
         int coarseDepth = (parameters / 100) % 100;
         IdxVector_d fineAggregate(numNodesIn, 0); // The partition label for the fine partition
         IdxVector_d fineAggregateSort; // The copy of the fine partition label that is sorted
         IdxVector_d finePartSizes; // Vector with sizes of fine partitions
         IdxVector_d neighborCountsIn; // Vector to hold the sizes for each nodes adjacency
         IdxVector_d permutedAdjIndexesIn; // Vector to hold the indices for the initial adjacency permutation
         IdxVector_d permutedAdjacencyIn; // Holds the permuted initial adjacency
         IdxVector_d neighborCountsOut; // Holds the counts of neighbors for the induced graph
         IdxVector_d coarseAggregate; // Holds the partition label for the coarse partition
         IdxVector_d aggregateRemapId; // Holds the current id of each aggregate
         IdxVector_d iAggregateRemapId; // Holds the permutation to remap the aggregate id'
         IdxVector_d aggregateRemapIndex; // Holds the start index of each aggregate
         IdxVector_d inducedNodeWeights; // Holds the sizes of the fine aggregates

         AggMIS::Types::JTimer jimmy;
         AggMIS::Types::JTimer iTime;
         int fineMin, fineMax, coarseMin, coarseMax;
         fineMax = parameters % 1000;
         fineMin = (parameters / 1000) % 1000;
         coarseMax = part_max_size % 1000;
         coarseMin = (part_max_size / 1000) % 1000;
         coarseDepth = (parameters / 1000000) % 10;
         fineDepth = (parameters / 10000000) % 10;


         partitionLabel = IdxVector_d(numNodesIn, 2); // Holds the partition each vertex is located in

         // First transfer in the graph
         AT::Graph_d fineGraph;
         fineGraph.indices->swap(adjIndexesIn);
         fineGraph.adjacency->swap(adjacencyIn);
         jimmy.start();
         // Now get an MIS of the graph
         iTime.start();
         IntVector_d *fineMIS = AggMIS::MIS::RandomizedMIS(fineDepth, fineGraph);
         iTime.stop();
         // Aggregate to nearest
         iTime.start();
         IntVector_d *fineAgg = AggMIS::Aggregation::AggregateToNearest(fineGraph, *fineMIS);
         iTime.stop();

         // Getting a conditioner
         iTime.start();
         AggMIS::MergeSplitGPU::MergeSplitConditionerGPU fineConditioner(fineGraph, *fineAgg);
         int desiredSize = (fineMin + fineMax) / 2;
         fineConditioner.SetSizeBounds(fineMin, fineMax);
         fineConditioner.Condition(desiredSize, true, .1, .1, 10);
         iTime.stop();
         jimmy.stop();

         // Getting the count of the MIS
         int misCount = thrust::count(fineMIS->begin(), fineMIS->end(), 1);
         //         DataRecorder::Add("Fine MIS Count", misCount);
         fineGraph.indices->swap(adjIndexesIn);
         fineGraph.adjacency->swap(adjacencyIn);
         fineAgg->swap(fineAggregate);

         // Record initial aggregation stats
         Help::RecordAllStats(adjIndexesIn,
               adjacencyIn,
               fineAggregate,
               "Initial Fine Aggregation");
         fineAgg->swap(fineAggregate);

         // Swap out the aggregation and graph
         fineConditioner.GetAggregation()->swap(fineAggregate);

         // Record final aggregation stats
         Help::RecordAllStats(adjIndexesIn,
               adjacencyIn,
               fineAggregate,
               "Fine Aggregation");

         // Clear temp stuff
         fineMIS->clear();
         delete fineMIS;
         fineAgg->clear();
         delete fineAgg;

         // Setting the permutation array to have values equal to element indices
         permutation = IdxVector_d(numNodesIn);
         misHelpers::fillWithIndex(permutation);

         // Sorting arrays together:
         fineAggregateSort = fineAggregate;
         thrust::sort_by_key(fineAggregateSort.begin(),
               fineAggregateSort.end(),
               permutation.begin());

         // Building the permutation array:
         misHelpers::getInversePermutation(permutation, ipermutation);

         // Getting the aggregate indices and node weights for the induced graph
         misHelpers::getPartSizes(fineAggregateSort,
               inducedNodeWeights,
               aggregateIdx);

         // Getting the induced graph:
         misHelpers::getInducedGraph(adjIndexesIn,
               adjacencyIn,
               fineAggregate,
               adjIndexesOut,
               adjacencyOut);

         int inducedGraphSize = adjIndexesOut.size() - 1;

         // Doing the coarse aggregation with AggMIS
         // Swapping in the graph data and weights
         AT::Graph_d coarseGraph;
         coarseGraph.indices->swap(adjIndexesOut);
         coarseGraph.adjacency->swap(adjacencyOut);
         IntVector_d nodeWeights;
         nodeWeights.swap(inducedNodeWeights);

         jimmy.start();
         // Getting an MIS
         iTime.start();
         IntVector_d *coarseMIS = AggMIS::MIS::RandomizedMIS(coarseDepth, coarseGraph);
         iTime.stop();

         // Getting initial aggregation
         iTime.start();
         IntVector_d *coarseAgg = AggMIS::Aggregation::AggregateToNearest(coarseGraph, *coarseMIS);
         iTime.stop();

         // Getting a conditioner
         iTime.start();
         AggMIS::MergeSplitGPU::MergeSplitConditionerGPU coarseConditioner(coarseGraph, *coarseAgg);
         coarseConditioner.SetNodeWeights(nodeWeights);
         coarseConditioner.SetSizeBounds(coarseMin, coarseMax);
         desiredSize = (coarseMin + coarseMax) / 2;
         coarseConditioner.Condition(desiredSize, true, .1, .1, 10);
         jimmy.stop();
         iTime.stop();
         misCount = thrust::count(coarseMIS->begin(), coarseMIS->end(), 1);

         // Swap out the aggregation, graph, and node weights
         coarseGraph.indices->swap(adjIndexesOut);
         coarseGraph.adjacency->swap(adjacencyOut);
         coarseConditioner.GetNodeWeights()->swap(inducedNodeWeights);
         coarseAgg->swap(coarseAggregate);

         // Record initial aggregation stats
         Help::RecordAllStats(adjIndexesOut,
               adjacencyOut,
               coarseAggregate,
               inducedNodeWeights,
               "Initial Coarse Aggregation");
         coarseAgg->swap(coarseAggregate);
         coarseConditioner.GetAggregation()->swap(coarseAggregate);

         // Record final aggregation stats
         Help::RecordAllStats(adjIndexesOut,
               adjacencyOut,
               coarseAggregate,
               inducedNodeWeights,
               "Coarse Aggregation");

         // Clear temp stuff
         coarseMIS->clear();
         delete coarseMIS;
         coarseAgg->clear();
         delete coarseAgg;

         // Performing new version of getting induced graph
         misHelpers::remapInducedGraph(adjIndexesOut, adjacencyOut, coarseAggregate);

         // Filling in the partitionLabel:
         misHelpers::fillPartitionLabel(coarseAggregate, fineAggregateSort, partitionLabel);

         // Do a stable sort by key with the partitionLabel as the key:
         thrust::stable_sort_by_key(partitionLabel.begin(), partitionLabel.end(), thrust::make_zip_iterator(thrust::make_tuple(fineAggregateSort.begin(), permutation.begin())));

         // Remapping the aggregate id's:
         aggregateRemapId = IdxVector_d(aggregateIdx.size() - 1, 0);
         aggregateRemapIndex = IdxVector_d(aggregateIdx.size() - 1, 0);
         misHelpers::fillWithIndex(aggregateRemapId);

         misHelpers::getAggregateStartIndices(fineAggregateSort, aggregateRemapIndex);

         thrust::stable_sort_by_key(aggregateRemapIndex.begin(), aggregateRemapIndex.end(), aggregateRemapId.begin());

         misHelpers::getInversePermutation(aggregateRemapId, iAggregateRemapId);

         misHelpers::remapAggregateIdx(fineAggregateSort, iAggregateRemapId);

         misHelpers::remapAggregateIdx(fineAggregate, iAggregateRemapId);

         // Sort the coarseAggregate for indices and permutation:
         thrust::sort(coarseAggregate.begin(), coarseAggregate.end());

         misHelpers::getPartIndices(coarseAggregate, partitionIdx);

         // Get indices for the fine aggregates
         misHelpers::getPartIndices(fineAggregateSort, aggregateIdx);

         // Putting in the right permutation vectors for the output:
         ipermutation = permutation;
         misHelpers::getInversePermutation(ipermutation, permutation);
      }
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
            bool verbose) {
         int numNodesIn = adjIndexesIn.size() - 1; // Size of input graph
         IdxVector_d fineAggregate(numNodesIn, 0); // The partition label for the fine partition
         IdxVector_d fineAggregateSort; // The copy of the fine partition label that is sorted
         IdxVector_d finePartSizes; // Vector with sizes of fine partitions
         IdxVector_d neighborCountsIn; // Vector to hold the sizes for each nodes adjacency
         IdxVector_d permutedAdjIndexesIn; // Vector to hold the indices for the initial adjacency permutation
         IdxVector_d permutedAdjacencyIn; // Holds the permuted initial adjacency
         IdxVector_d neighborCountsOut; // Holds the counts of neighbors for the induced graph
         IdxVector_d coarseAggregate; // Holds the partition label for the coarse partition
         IdxVector_d aggregateRemapId; // Holds the current id of each aggregate
         IdxVector_d iAggregateRemapId; // Holds the permutation to remap the aggregate id'
         IdxVector_d aggregateRemapIndex; // Holds the start index of each aggregate
         IdxVector_d inducedNodeWeights; // Holds the sizes of the fine aggregates

         // Creating timer
         AggMIS::Types::JTimer jimmy;
         AggMIS::Types::JTimer iTime;

         // Parsing the parameters
         int fineMin, fineMax, coarseMin, coarseMax, fineDepth, coarseDepth;
         fineMax = parameters % 1000;
         fineMin = (parameters / 1000) % 1000;
         coarseMax = part_max_size % 1000;
         coarseMin = (part_max_size / 1000) % 1000;
         coarseDepth = (parameters / 1000000) % 10;
         fineDepth = (parameters / 10000000) % 10;

         // Initialize the partitionLabel array
         partitionLabel = IdxVector_d(numNodesIn, 2); // Holds the partition each vertex is located in

         // Getting aggregation of graph with AggMIS
         AT::Graph_d fineGraph;
         fineGraph.indices->swap(adjIndexesIn);
         fineGraph.adjacency->swap(adjacencyIn);

         // Getting a host version of the graph
         AT::Graph_h fineGraph_h(fineGraph);
         jimmy.start();

         // Now get an MIS of the graph
         iTime.start();
         AT::IntVector_h *fineMIS = AggMIS::MIS::FloodFillMIS(fineDepth, fineGraph_h);
         iTime.stop();

         // Aggregate to nearest
         iTime.start();
         AT::IntVector_h *fineAgg = AggMIS::Aggregation::AggregateToNearest(fineGraph_h, *fineMIS);
         iTime.stop();

         // Getting a conditioner
         iTime.start();
         AggMIS::MergeSplitCPU::MergeSplitConditionerCPU fineConditioner(fineGraph_h, *fineAgg);

         int desiredSize = (fineMin + fineMax) / 2;
         fineConditioner.SetSizeBounds(fineMin, fineMax);
         fineConditioner.Condition(desiredSize, true, .1, .1, 10);
         jimmy.stop();
         iTime.stop();

         // Getting the count of the MIS
         int misCount = thrust::count(fineMIS->begin(), fineMIS->end(), 1);
         //         DataRecorder::Add("Fine MIS Count", misCount);

         // Swap out the aggregation and graph
         fineGraph.indices->swap(adjIndexesIn);
         fineGraph.adjacency->swap(adjacencyIn);
         fineAggregate.assign(fineAgg->begin(), fineAgg->end());
         Help::RecordAllStats(adjIndexesIn,
               adjacencyIn,
               fineAggregate,
               "Initial Fine Aggregation");
         fineAggregate.assign(fineConditioner.GetAggregation()->begin(),
               fineConditioner.GetAggregation()->end());
         Help::RecordAllStats(adjIndexesIn,
               adjacencyIn,
               fineAggregate,
               "Fine Aggregation");

         // Clear temp stuff
         fineMIS->clear();
         delete fineMIS;
         fineAgg->clear();
         delete fineAgg;

         // Setting the permutation array to have values equal to element indices
         permutation = IdxVector_d(numNodesIn);
         misHelpers::fillWithIndex(permutation);

         // Sorting arrays together:
         fineAggregateSort = fineAggregate;
         thrust::sort_by_key(fineAggregateSort.begin(),
               fineAggregateSort.end(),
               permutation.begin());

         // Building the permutation array:
         misHelpers::getInversePermutation(permutation, ipermutation);

         // Getting the aggregate indices and node weights for the induced graph
         misHelpers::getPartSizes(fineAggregateSort,
               inducedNodeWeights,
               aggregateIdx);

         // Getting the induced graph:
         misHelpers::getInducedGraph(adjIndexesIn,
               adjacencyIn,
               fineAggregate,
               adjIndexesOut,
               adjacencyOut);

         int inducedGraphSize = adjIndexesOut.size() - 1;

         // Doing the coarse aggregation with AggMIS
         // Swapping in the graph data and weights
         AT::Graph_d coarseGraph;
         coarseGraph.indices->swap(adjIndexesOut);
         coarseGraph.adjacency->swap(adjacencyOut);

         AT::Graph_h coarseGraph_h(coarseGraph);
         AT::IntVector_h nodeWeights_h(inducedNodeWeights.begin(),
               inducedNodeWeights.end());
         jimmy.start();

         // Getting an MIS
         iTime.start();
         AT::IntVector_h *coarseMIS = AggMIS::MIS::FloodFillMIS(coarseDepth, coarseGraph_h);
         iTime.stop();

         // Getting initial aggregation
         iTime.start();
         AT::IntVector_h *coarseAgg = AggMIS::Aggregation::AggregateToNearest(coarseGraph_h, *coarseMIS);
         iTime.stop();

         // Getting a conditioner
         iTime.start();
         AggMIS::MergeSplitCPU::MergeSplitConditionerCPU coarseConditioner(coarseGraph_h, *coarseAgg);
         coarseConditioner.SetNodeWeights(nodeWeights_h);
         coarseConditioner.SetSizeBounds(coarseMin, coarseMax);
         desiredSize = (coarseMin + coarseMax) / 2;
         coarseConditioner.Condition(desiredSize, true, .1, .1, 10);
         jimmy.stop();
         iTime.stop();
         misCount = thrust::count(coarseMIS->begin(), coarseMIS->end(), 1);

         // Swap out the aggregation, graph, and node weights
         coarseGraph.indices->swap(adjIndexesOut);
         coarseGraph.adjacency->swap(adjacencyOut);
         coarseAggregate.assign(coarseAgg->begin(), coarseAgg->end());
         Help::RecordAllStats(adjIndexesOut,
               adjacencyOut,
               coarseAggregate,
               inducedNodeWeights,
               "Initial Coarse Aggregation");
         coarseAggregate.assign(coarseConditioner.GetAggregation()->begin(),
               coarseConditioner.GetAggregation()->end());
         Help::RecordAllStats(adjIndexesOut,
               adjacencyOut,
               coarseAggregate,
               inducedNodeWeights,
               "Coarse Aggregation");

         // Clear temp stuff
         coarseMIS->clear();
         delete coarseMIS;
         coarseAgg->clear();
         delete coarseAgg;

         // Performing new version of getting induced graph
         misHelpers::remapInducedGraph(adjIndexesOut, adjacencyOut, coarseAggregate);

         // Filling in the partitionLabel:
         misHelpers::fillPartitionLabel(coarseAggregate, fineAggregateSort, partitionLabel);

         // Do a stable sort by key with the partitionLabel as the key:
         thrust::stable_sort_by_key(partitionLabel.begin(), partitionLabel.end(), thrust::make_zip_iterator(thrust::make_tuple(fineAggregateSort.begin(), permutation.begin())));

         // Remapping the aggregate id's:
         aggregateRemapId = IdxVector_d(aggregateIdx.size() - 1, 0);
         aggregateRemapIndex = IdxVector_d(aggregateIdx.size() - 1, 0);
         misHelpers::fillWithIndex(aggregateRemapId);
         misHelpers::getAggregateStartIndices(fineAggregateSort, aggregateRemapIndex);
         thrust::stable_sort_by_key(aggregateRemapIndex.begin(), aggregateRemapIndex.end(), aggregateRemapId.begin());
         misHelpers::getInversePermutation(aggregateRemapId, iAggregateRemapId);
         misHelpers::remapAggregateIdx(fineAggregateSort, iAggregateRemapId);
         misHelpers::remapAggregateIdx(fineAggregate, iAggregateRemapId);

         // Sort the coarseAggregate for indices and permutation:
         thrust::sort(coarseAggregate.begin(), coarseAggregate.end());
         misHelpers::getPartIndices(coarseAggregate, partitionIdx);

         // Get indices for the fine aggregates
         misHelpers::getPartIndices(fineAggregateSort, aggregateIdx);

         // Putting in the right permutation vectors for the output:
         ipermutation = permutation;
         misHelpers::getInversePermutation(ipermutation, permutation);
         if (verbose)
            printf("Total aggregation time (Conditioned MIS CPU): %3.4fs for %d calls\n", totalAggregationTime, totalAggregationCalls);
      }
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
            bool verbose) {
         int numNodesIn = adjIndexesIn.size() - 1; // Size of input graph
         IdxVector_d fineAggregate(numNodesIn, 0); // The partition label for the fine partition
         IdxVector_d fineAggregateSort; // The copy of the fine partition label that is sorted
         IdxVector_d finePartSizes; // Vector with sizes of fine partitions
         IdxVector_d neighborCountsIn; // Vector to hold the sizes for each nodes adjacency
         IdxVector_d permutedAdjIndexesIn; // Vector to hold the indices for the initial adjacency permutation
         IdxVector_d permutedAdjacencyIn; // Holds the permuted initial adjacency
         IdxVector_d neighborCountsOut; // Holds the counts of neighbors for the induced graph
         IdxVector_d coarseAggregate; // Holds the partition label for the coarse partition
         IdxVector_d aggregateRemapId; // Holds the current id of each aggregate
         IdxVector_d iAggregateRemapId; // Holds the permutation to remap the aggregate id'
         IdxVector_d aggregateRemapIndex; // Holds the start index of each aggregate
         IdxVector_d inducedNodeWeights; // Holds the sizes of the fine aggregates

         // Creating timer
         AggMIS::Types::JTimer jimmy;
         AggMIS::Types::JTimer iTime;

         // Parsing the parameters
         int fineDepth, coarseDepth;
         int maxPart = parameters % 1000;
         coarseDepth = (parameters / 1000000) % 10;
         fineDepth = (parameters / 10000000) % 10;


         // Initialize the partitionLabel array
         partitionLabel = IdxVector_d(numNodesIn, 2); // Holds the partition each vertex is located in

         // Getting aggregation of graph with AggMIS
         AT::Graph_d fineGraph;
         fineGraph.indices->swap(adjIndexesIn);
         fineGraph.adjacency->swap(adjacencyIn);

         // Getting a host version of the graph
         AT::Graph_h fineGraph_h(fineGraph);
         jimmy.start();

         // Now get an MIS of the graph
         iTime.start();
         AT::IntVector_h *fineMIS = AggMIS::MIS::FloodFillMIS(fineDepth, fineGraph_h);
         iTime.stop();

         // Aggregate to nearest
         iTime.start();
         AT::IntVector_h *fineAgg = AggMIS::Aggregation::AggregateToNearest(fineGraph_h, *fineMIS);
         iTime.stop();

         // Getting a conditioner
         iTime.start();
         AggMIS::MergeSplitCPU::MergeSplitConditionerCPU fineConditioner(fineGraph_h, *fineAgg);

         jimmy.stop();
         iTime.stop();

         // Getting the count of the MIS
         int misCount = thrust::count(fineMIS->begin(), fineMIS->end(), 1);

         // Swap out the aggregation and graph
         fineGraph.indices->swap(adjIndexesIn);
         fineGraph.adjacency->swap(adjacencyIn);
         fineAggregate.assign(fineAgg->begin(), fineAgg->end());
         Help::RecordAllStats(adjIndexesIn,
               adjacencyIn,
               fineAggregate,
               "Initial Fine Aggregation");
         fineAggregate.assign(fineConditioner.GetAggregation()->begin(),
               fineConditioner.GetAggregation()->end());
         Help::RecordAllStats(adjIndexesIn,
               adjacencyIn,
               fineAggregate,
               "Fine Aggregation");

         // Clear temp stuff
         fineMIS->clear();
         delete fineMIS;
         fineAgg->clear();
         delete fineAgg;

         // Setting the permutation array to have values equal to element indices
         permutation = IdxVector_d(numNodesIn);
         misHelpers::fillWithIndex(permutation);

         // Sorting arrays together:
         fineAggregateSort = fineAggregate;
         thrust::sort_by_key(fineAggregateSort.begin(),
               fineAggregateSort.end(),
               permutation.begin());

         // Building the permutation array:
         misHelpers::getInversePermutation(permutation, ipermutation);

         // Getting the aggregate indices and node weights for the induced graph
         //            finePartCount = fineAggregateSort[fineAggregateSort.size() - 1];
         misHelpers::getPartSizes(fineAggregateSort,
               inducedNodeWeights,
               aggregateIdx);

         // Getting the induced graph:
         misHelpers::getInducedGraph(adjIndexesIn,
               adjacencyIn,
               fineAggregate,
               adjIndexesOut,
               adjacencyOut);

         int inducedGraphSize = adjIndexesOut.size() - 1;

         // Doing the coarse aggregation with AggMIS
         // Swapping in the graph data and weights
         AT::Graph_d coarseGraph;
         coarseGraph.indices->swap(adjIndexesOut);
         coarseGraph.adjacency->swap(adjacencyOut);

         AT::Graph_h coarseGraph_h(coarseGraph);
         AT::IntVector_h nodeWeights_h(inducedNodeWeights.begin(),
               inducedNodeWeights.end());
         jimmy.start();

         // Getting an MIS
         iTime.start();
         AT::IntVector_h *coarseMIS = AggMIS::MIS::FloodFillMIS(coarseDepth, coarseGraph_h);
         iTime.stop();

         // Getting initial aggregation
         iTime.start();
         AT::IntVector_h *coarseAgg = AggMIS::Aggregation::AggregateToNearest(coarseGraph_h, *coarseMIS);
         iTime.stop();

         // Getting a conditioner
         iTime.start();
         AggMIS::MergeSplitCPU::MergeSplitConditionerCPU coarseConditioner(coarseGraph_h, *coarseAgg);
         coarseConditioner.SetNodeWeights(nodeWeights_h);
         coarseConditioner.SetSizeBounds(maxPart / 2, maxPart);
         coarseConditioner.CycleSplits(true);
         coarseConditioner.CycleMerges(false);
         jimmy.stop();
         iTime.stop();
         misCount = thrust::count(coarseMIS->begin(), coarseMIS->end(), 1);

         // Swap out the aggregation, graph, and node weights
         coarseGraph.indices->swap(adjIndexesOut);
         coarseGraph.adjacency->swap(adjacencyOut);
         coarseAggregate.assign(coarseAgg->begin(), coarseAgg->end());
         Help::RecordAllStats(adjIndexesOut,
               adjacencyOut,
               coarseAggregate,
               inducedNodeWeights,
               "Initial Coarse Aggregation");
         coarseAggregate.assign(coarseConditioner.GetAggregation()->begin(),
               coarseConditioner.GetAggregation()->end());
         Help::RecordAllStats(adjIndexesOut,
               adjacencyOut,
               coarseAggregate,
               inducedNodeWeights,
               "Coarse Aggregation");

         // Clear temp stuff
         coarseMIS->clear();
         delete coarseMIS;
         coarseAgg->clear();
         delete coarseAgg;

         // Performing new version of getting induced graph
         misHelpers::remapInducedGraph(adjIndexesOut, adjacencyOut, coarseAggregate);

         // Filling in the partitionLabel:
         misHelpers::fillPartitionLabel(coarseAggregate,
               fineAggregateSort,
               partitionLabel);

         // Do a stable sort by key with the partitionLabel as the key:
         thrust::stable_sort_by_key(partitionLabel.begin(),
               partitionLabel.end(),
               thrust::make_zip_iterator(
                  thrust::make_tuple(
                     fineAggregateSort.begin(),
                     permutation.begin())));

         // Remapping the aggregate id's:
         aggregateRemapId = IdxVector_d(aggregateIdx.size() - 1, 0);
         aggregateRemapIndex = IdxVector_d(aggregateIdx.size() - 1, 0);
         misHelpers::fillWithIndex(aggregateRemapId);
         misHelpers::getAggregateStartIndices(fineAggregateSort, aggregateRemapIndex);
         thrust::stable_sort_by_key(aggregateRemapIndex.begin(), aggregateRemapIndex.end(), aggregateRemapId.begin());
         misHelpers::getInversePermutation(aggregateRemapId, iAggregateRemapId);
         misHelpers::remapAggregateIdx(fineAggregateSort, iAggregateRemapId);
         misHelpers::remapAggregateIdx(fineAggregate, iAggregateRemapId);

         // Sort the coarseAggregate for indices and permutation:
         thrust::sort(coarseAggregate.begin(), coarseAggregate.end());
         misHelpers::getPartIndices(coarseAggregate, partitionIdx);

         // Get indices for the fine aggregates
         misHelpers::getPartIndices(fineAggregateSort, aggregateIdx);

         // Putting in the right permutation vectors for the output:
         ipermutation = permutation;
         misHelpers::getInversePermutation(ipermutation, permutation);
      }
   }
   namespace Help {
      namespace AT = AggMIS::Types;
      int GetMetisAggregation(AT::IntVector_h &indices,
            AT::IntVector_h &adjacency,
            AT::IntVector_h &result,
            int partSize,
            bool verbose) {
         // Getting size of graph
         int graphSize = indices.size() - 1;

         // Making sure result is sized correctly
         result.resize(graphSize);

         // Setting up for Metis call:
         int nparts, edgecut;
         int *npart = &result[0];
         nparts = (graphSize / partSize);
         if (nparts < 8192) {
            if (nparts < 2)
               nparts = 2;
            int options[10], pnumflag = 0, wgtflag = 0;
            for (int i = 0; i < 10; i++)
               options[i] = 0;

            AggMIS::Types::JTimer jimmy;
            jimmy.start();

            METIS_PartGraphKway(&graphSize,
                  &indices[0],
                  &adjacency[0],
                  NULL,
                  NULL,
                  &wgtflag,
                  &pnumflag,
                  &nparts,
                  options,
                  &edgecut,
                  npart);
            jimmy.stop();
            if (verbose)
               printf("Metis call for graph of %d nodes into parts of size %d took %3.4fs (host) %3.4fs (cuda)\n", graphSize, partSize, jimmy.getElapsedTimeInSec(true), jimmy.getElapsedTimeInSec(false));

            totalAggregationTime += jimmy.getElapsedTimeInSec(true);
            totalAggregationCalls++;

            // Output timing to file
            std::ofstream outputFile;
            outputFile.open("TimingResults.csv", std::ofstream::app);
            if (totalAggregationCalls == 1)
               outputFile << "\n\nNote,Metis Aggregation Time, Total Calls, Total Time\n";
            outputFile << "Metis call on graph with " << graphSize << " nodes into parts of size " << partSize << ",";
            outputFile << jimmy.getElapsedTimeInSec(true) <<
               "," << totalAggregationCalls << "," << totalAggregationTime << "\n";

            return EnsureConnectedAndNonEmpty(indices, adjacency, result);
         }
         else {
            int count = GetMetisAggregation_Large(indices, adjacency, result, partSize);
            return count;
         }
      }
      int GetMetisAggregation_Large(AT::IntVector_h &indices,
            AT::IntVector_h &adjacency,
            AT::IntVector_h &result,
            int partSize,
            bool verbose) {
         // Getting size of graph
         int graphSize = indices.size() - 1;

         // Getting a partitioning with four parts to create subgraphs
         int subGraphSize = graphSize / 4;
         GetMetisAggregation(indices, adjacency, result, subGraphSize);

         // Getting the subgraphs from the partitioning
         AT::IntVector_h_ptr subIndices, subAdjacencies, subNodeMaps;
         GetSubGraphs(indices, adjacency, result, subIndices, subAdjacencies, subNodeMaps);
         int subGraphCount = subIndices.size();

         // Getting aggregation of each subgraph and mapping to original graph
         int offset = 0;
         for (int i = 0; i < subGraphCount; i++) {
            AT::IntVector_h agg;
            AT::IntVector_h &nodeMap = *(subNodeMaps[i]);
            AT::IntVector_h &ind = *(subIndices[i]);
            AT::IntVector_h &adj = *(subAdjacencies[i]);

            int aggCount = GetMetisAggregation(ind, adj, agg, partSize);
            for (int n = 0; n < agg.size(); n++) {
               // Look up original node Id
               int node = nodeMap[n];
               // Set the aggregate to the subgraph's plus current offset
               result[node] = agg[n] + offset;
            }
            offset += aggCount;
            agg.clear();
            ind.clear();
            adj.clear();
            nodeMap.clear();
         }

         // Cleaning up
         subIndices.clear();
         subAdjacencies.clear();
         subNodeMaps.clear();

         // Return number of aggregates
         return offset;
      }
      void GetSubGraphs(AT::IntVector_h &indices,
            AT::IntVector_h &adjacency,
            AT::IntVector_h &partition,
            AT::IntVector_h_ptr &newIndices,
            AT::IntVector_h_ptr &newAdjacencies,
            AT::IntVector_h_ptr &nodeMaps,
            bool verbose) {

         // Getting a map from old graph id to subgraph id
         AT::IntVector_h mapToSubGraphId(adjacency.size() - 1);

         // Getting separate node maps for all partitions
         nodeMaps.clear();
         int minPart = partition[0];
         int maxPart = partition[0];
         for (int i = 0; i < partition.size(); i++) {
            int partId = partition[i];
            minPart = std::min(minPart, partId);
            maxPart = std::max(maxPart, partId);
            while (partId + 1 > nodeMaps.size())
               nodeMaps.push_back(new AT::IntVector_h());
            nodeMaps[partId]->push_back(i);
            mapToSubGraphId[i] = nodeMaps[partId]->size() - 1;
         }
         int graphCount = nodeMaps.size();

         // Creating the new subgraph indices and adjacency vectors
         newIndices.resize(graphCount);
         newAdjacencies.resize(graphCount);
         for (int i = 0; i < graphCount; i++) {
            newIndices[i] = new AT::IntVector_h(nodeMaps[i]->size() + 1);

            AT::IntVector_h *ptr = newIndices[i];

            newAdjacencies[i] = new AT::IntVector_h();
         }

         // Filling the subgraphs in
         for (int i = 0; i < graphCount; i++) {
            AT::IntVector_h &nodes = *nodeMaps[i];
            AT::IntVector_h &ind = *newIndices[i];
            AT::IntVector_h &adj = *newAdjacencies[i];
            int insertAt = 0;
            (*newIndices[0])[0] = 0;
            for (int nIt = 0; nIt < nodes.size(); nIt++) {
               int node = nodes[nIt];

               if (partition[node] != i) {
                  int p = partition[node];
                  if (verbose)
                     printf("Node %d found in node list %d but marked as in partition %d\n",
                           node, i, p);
                  std::cin >> p;
               }

               int start = indices[node];
               int end = indices[node + 1];
               for (int n = start; n < end; n++) {
                  int neighbor = adjacency[n];
                  if (partition[neighbor] == i) {
                     newAdjacencies[i]->push_back(mapToSubGraphId[neighbor]);
                     insertAt++;
                  }
               }
               ind[nIt + 1] = insertAt;
            }
         }

         // Cleaning up
         mapToSubGraphId.clear();
      }
      int EnsureConnectedAndNonEmpty(AT::IntVector_h &indices,
            AT::IntVector_h &adjacency,
            AT::IntVector_h &aggregation) {
         AT::IntVector_h temp(aggregation.size());

         // Flood fill aggregates with node indices
         for (int i = 0; i < temp.size(); i++)
            temp[i] = i;
         bool changed = true;
         while (changed) {
            changed = false;
            for (int root = 0; root < aggregation.size(); root++) {
               int rootValue = temp[root];
               int rootAggregate = aggregation[root];
               int start = indices[root];
               int end = indices[root + 1];
               for (int nIt = start; nIt < end; nIt++) {
                  int neighbor = adjacency[nIt];
                  int neighborAggregate = aggregation[neighbor];
                  int neighborValue = temp[neighbor];
                  if (rootAggregate == neighborAggregate && neighborValue > rootValue)
                     rootValue = neighborValue;
               }
               if (rootValue > temp[root]) {
                  temp[root] = rootValue;
                  changed = true;
               }
            }
         }

         // Making a copy of the filled aggregation
         AT::IntVector_h mapping(temp.size());
         thrust::copy(temp.begin(), temp.end(), mapping.begin());


         // Sort the values
         thrust::sort(mapping.begin(), mapping.end());

         // Get just unique values
         int newSize = thrust::unique(mapping.begin(), mapping.end()) - mapping.begin();
         mapping.resize(newSize);

         // Remap aggregation
         for (int i = 0; i < aggregation.size(); i++)
            aggregation[i] = BinarySearch(temp[i], mapping);

         // Get rid of temporary vectors
         mapping.clear();
         temp.clear();

         // Return count of aggregates
         return newSize;
      }
      int BinarySearch(int value,
            AT::IntVector_h &array) {
         int imin = 0;
         int imax = array.size() - 1;
         while (imin < imax) {
            int imid = (imax + imin) / 2;
            if (array[imid] < value)
               imin = imid + 1;
            else
               imax = imid;
         }
         if (imax == imin && array[imin] == value)
            return imin;
         else
            return -1;
      }
      void RecordAllStats(IdxVector_d& indices,
            IdxVector_d& adjacency,
            IdxVector_d& aggregation,
            std::string prefix) {
         IdxVector_d dummy;
         RecordAllStats(indices,
               adjacency,
               aggregation,
               dummy,
               prefix);
      }
      void RecordAllStats(IdxVector_d& indices,
            IdxVector_d& adjacency,
            IdxVector_d& aggregation,
            IdxVector_d& nodeWeights,
            std::string prefix) {
         // Recording aggregation stats
         if (nodeWeights.size() == 0)
            RecordAggregationStats(aggregation, prefix + ":Parts");
         else
            RecordAggregationStats(aggregation, nodeWeights, prefix + ":Parts");

         // Recording Valence stats
         RecordValenceStats(indices, adjacency, prefix + ":Valence");

         // Recording Edge cut ratio
         RecordEdgeCut(indices, adjacency, aggregation, prefix);
      }
      void RecordAggregationStats(IdxVector_d& aggregation,
            std::string prefix) {
         AT::IntVector_d agg;
         agg.swap(aggregation);
         AT::IntVector_d partSizes;

         // Get the part sizes
         AggMIS::GraphHelpers::getPartSizes(agg, partSizes);

         // Find the largest and smallest parts
         thrust::sort(partSizes.begin(), partSizes.end());
         int smallest = partSizes[0];
         int largest = partSizes.back();

         // Get the mean, median, and std deviation
         double meanSize = (double)agg.size() / partSizes.size();
         int medianSize = partSizes[(partSizes.size() - 1) / 2];
         double std = thrust::transform_reduce(partSizes.begin(),
               partSizes.end(),
               AggMIS::MergeSplitGPU::Functors::SquaredDifference(meanSize),
               0.0,
               thrust::plus<double>());
         std = sqrt(std / partSizes.size());

         agg.swap(aggregation);
      }
      void RecordAggregationStats(IdxVector_d& aggregation,
            IdxVector_d& nodeWeights,
            std::string prefix) {
         AT::IntVector_d agg;
         agg.swap(aggregation);
         AT::IntVector_d nw;
         nw.swap(nodeWeights);
         AT::IntVector_d partSizes;

         // Get the part sizes
         AggMIS::GraphHelpers::getPartSizes(agg, partSizes, nw);


         // Find the largest and smallest parts
         thrust::sort(partSizes.begin(), partSizes.end());
         int smallest = partSizes[0];
         int largest = partSizes.back();

         // Get the mean, median, and std deviation
         int totalWeight = thrust::reduce(nw.begin(), nw.end());
         int medianSize = partSizes[(partSizes.size() - 1) / 2];
         double meanSize = (double)totalWeight / partSizes.size();
         double std = thrust::transform_reduce(partSizes.begin(),
               partSizes.end(),
               AggMIS::MergeSplitGPU::Functors::SquaredDifference(meanSize),
               0.0,
               thrust::plus<double>());
         std = sqrt(std / partSizes.size());

         agg.swap(aggregation);
         nw.swap(nodeWeights);
      }
      void RecordValenceStats(IdxVector_d& indices,
            IdxVector_d& adjacency,
            std::string prefix) {
         // Get a graph object to use
         AT::Graph_d g;
         g.adjacency->swap(adjacency);
         g.indices->swap(indices);

         // Get the valences from the graph
         AT::IntVector_d* valences = AggMIS::GraphHelpers::GetValences(g);

         // Compute the stats
         thrust::sort(valences->begin(), valences->end());
         int smallest = valences->data()[0];
         int largest = valences->back();

         // Get the mean, median, and std deviation
         int totalValence = thrust::reduce(valences->begin(),
               valences->end());
         double meanSize = (double)totalValence / valences->size();
         int medianSize = valences->data()[(valences->size() - 1) / 2];
         double std = thrust::transform_reduce(valences->begin(),
               valences->end(),
               AggMIS::MergeSplitGPU::Functors::SquaredDifference(meanSize),
               0.0,
               thrust::plus<double>());
         std = sqrt(std / valences->size());

         g.adjacency->swap(adjacency);
         g.indices->swap(indices);
      }
      void RecordEdgeCut(IdxVector_d& indices,
            IdxVector_d& adjacency,
            IdxVector_d& aggregation,
            std::string prefix) {
         // Get a graph
         AT::Graph_d g;
         g.indices->swap(indices);
         g.adjacency->swap(adjacency);

         // Get an IntVector for aggregation
         AT::IntVector_d agg;
         agg.swap(aggregation);

         // Swapping back data
         g.indices->swap(indices);
         g.adjacency->swap(adjacency);
         agg.swap(aggregation);
      }
   }
}
