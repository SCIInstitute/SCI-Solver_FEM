#include <smoothedMG/aggregators/mis.h>
#include <algorithm>
#include <queue>
#include <time.h>
#include <smoothedMG/aggregators/misHelpers.h>
#include <AggMIS_Aggregation_GPU.h>
#include <AggMIS_MergeSplitConditioner.h>
#include <AggMIS_MIS_GPU.h>
using namespace std;

template <class Matrix, class Vector>
void RandMIS_Aggregator<Matrix, Vector>::computePermutation(TetMesh* meshPtr, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitionlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize)
{
   // Getting the neighbors for the mesh
   meshPtr->need_neighbors();

   // Vertex count:
   int nn = meshPtr->vertices.size();

   // Counting up edges for adjacency:
   int edgeCount = 0;
   for(int vIt = 0; vIt < nn; vIt++)
   {
      edgeCount += meshPtr->neighbors[vIt].size();
   }

   //Allocating storage for array values of adjacency
   int* xadj = new int[nn + 1];
   int* adjncy = new int[edgeCount];

   // filling the arrays:
   xadj[0] = 0;
   int idx = 0;

   // Populating the arrays:
   for(int i = 1; i < nn + 1; i++)
   {
      xadj[i] = xadj[i - 1] + meshPtr->neighbors[i - 1].size();
      for(int j = 0; j < meshPtr->neighbors[i - 1].size(); j++)
      {
         adjncy[idx++] = meshPtr->neighbors[i - 1][j];
      }
   }

   // Calling the other override to finish:
   computePermutation(nn, xadj, adjncy, permutation, ipermutation, aggregateIdx, partitionIdx, partitionlabel, nnout, xadjout, adjncyout, metissize);

   // Freeing up memories:
   delete [] xadj;
   delete [] adjncy;

}

template <class Matrix, class Vector>
void RandMIS_Aggregator<Matrix, Vector>::computePermutation(TriMesh* meshPtr, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitionlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize)
{
   // Getting the neighbors for the mesh
   meshPtr->need_neighbors();

   // Vertex count:
   int nn = meshPtr->vertices.size();

   // Counting up edges for adjacency:
   int edgeCount = 0;
   for(int vIt = 0; vIt < nn; vIt++)
   {
      edgeCount += meshPtr->neighbors[vIt].size();
   }

   //Allocating storage for array values of adjacency
   int* xadj = new int[nn + 1];
   int* adjncy = new int[edgeCount];



   // filling the arrays:
   xadj[0] = 0;
   int idx = 0;

   // Populating the arrays:
   for(int i = 1; i < nn + 1; i++)
   {
      xadj[i] = xadj[i - 1] + meshPtr->neighbors[i - 1].size();
      for(int j = 0; j < meshPtr->neighbors[i - 1].size(); j++)
      {
         adjncy[idx++] = meshPtr->neighbors[i - 1][j];
      }
   }

   // Calling the other override to finish:
   computePermutation(nn, xadj, adjncy, permutation, ipermutation, aggregateIdx, partitionIdx, partitionlabel, nnout, xadjout, adjncyout, metissize);

   // Freeing up memories:
   delete [] xadj;
   delete [] adjncy;
}

template <class Matrix, class Vector>
void RandMIS_Aggregator<Matrix, Vector>::computePermutation(int nn, int* xadj, int* adjncy, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitionlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize)
{
   // Starting off by finding a fine aggregation of the mesh
   int *fineAggregate = new int[nn];
   // get the initial depth from parameter
   int depth = metissize / 10;
   int finePartCount;
   extendedMIS(nn, depth, xadj, adjncy, fineAggregate, &finePartCount);

   // Building the next level graph
   int notNeighbor = finePartCount + 1;
   int *permutedFullAdjacency = new int[xadj[nn]];
   int *initialPermutationOldToNew = new int[nn];
   int *initialPermutationNewToOld = new int[nn];
   int *permutationCheck = new int[nn];

   int *nextAggregateOffset = new int[finePartCount + 1];
   int *tempAggregateOffset = new int[finePartCount + 1];
   int *aggregateVertexCounts = new int[finePartCount];
   int *vertexNeighborCounts = new int[nn];
   int *aggregateNeighborCounts = new int[finePartCount];
   int *vertexAdjacencyOffsets = new int[nn + 1];
   int *aggregateAdjacency = new int[finePartCount];
   int *newAdjacencyIndexes = new int[finePartCount + 1];
   int *newAdjacency;

   // Clearing aggregate counts
   for(int i = 0; i < finePartCount; i++)
   {
      aggregateVertexCounts[i] = 0;
      aggregateNeighborCounts[i] = 0;
   }

   // Counting vertices in each aggregate, and total neighbors
   for(int vIt = 0; vIt < nn; vIt++)
   {
      aggregateVertexCounts[fineAggregate[vIt]]++;
      aggregateNeighborCounts[fineAggregate[vIt]] += xadj[vIt + 1] - xadj[vIt];
   }

   // Finding min/max aggregates:
   int min = aggregateVertexCounts[0];
   int max = aggregateVertexCounts[0];
   for(int i = 0; i < finePartCount; i++)
   {
      if(aggregateVertexCounts[i] > max)
         max = aggregateVertexCounts[i];
      if(aggregateVertexCounts[i] < min)
         min = aggregateVertexCounts[i];
   }
   //printf("There are: %d aggregates, minimum size: %d maximum size: %d\n", finePartCount, min, max);

   // Calculating the new offsets of each aggregate
   nextAggregateOffset[0] = 0;
   tempAggregateOffset[0] = 0;
   for(int aIt = 1; aIt < finePartCount + 1; aIt++)
   {
      // Doing a prefix sum:
      nextAggregateOffset[aIt] = nextAggregateOffset[aIt - 1] + aggregateVertexCounts[aIt - 1];
      tempAggregateOffset[aIt] = nextAggregateOffset[aIt - 1] + aggregateVertexCounts[aIt - 1];
   }

   // Filling in the initialPermutation array:
   for(int vIt = 0; vIt < nn; vIt++)
   {
      int aggID = fineAggregate[vIt];
      initialPermutationOldToNew[vIt] = tempAggregateOffset[aggID];
      initialPermutationNewToOld[tempAggregateOffset[aggID]] = vIt;
      tempAggregateOffset[aggID]++;
   }

   // For testing check the permutation array for consistency
   for(int vIt = 0; vIt < nn; vIt++)
   {
      permutationCheck[vIt] = initialPermutationOldToNew[initialPermutationNewToOld[vIt]];
   }


   // Counting neighbors of each (permuted) node
   for(int vIt = 0; vIt < nn; vIt++)
   {
      int oldNodeID = initialPermutationNewToOld[vIt];
      vertexNeighborCounts[vIt] = xadj[oldNodeID + 1] - xadj[oldNodeID];
   }

   // Calculating the new vertex offsets:
   vertexAdjacencyOffsets[0] = 0;
   for(int vIt = 1; vIt < nn + 1; vIt++)
   {
      vertexAdjacencyOffsets[vIt] = vertexAdjacencyOffsets[vIt - 1] + vertexNeighborCounts[vIt - 1];
   }

   // Filling in the permutedFullAdjacency
   for(int vIt = 0; vIt < nn; vIt++)
   {
      int permIdx = initialPermutationOldToNew[vIt];
      int currentPart = fineAggregate[vIt];

      int newOffset = vertexAdjacencyOffsets[permIdx];
      int oldOffset = xadj[vIt];
      for(int nIt = 0; nIt < vertexNeighborCounts[permIdx]; nIt++)
      {
         int partID = fineAggregate[adjncy[oldOffset + nIt]];

         if(partID == currentPart)
            permutedFullAdjacency[newOffset + nIt] = notNeighbor;
         else
            permutedFullAdjacency[newOffset + nIt] = partID;
         //permutedFullAdjacency[newOffset + nIt] = adjacency[oldOffset + nIt];
      }
   }

   // Sorting each aggregates neighbors (with duplicates)
   for(int aIt = 0; aIt < finePartCount; aIt++)
   {
      int beginAddr = vertexAdjacencyOffsets[ nextAggregateOffset[aIt] ];
      int endAddr = vertexAdjacencyOffsets[ nextAggregateOffset[aIt + 1] ];
      sort(permutedFullAdjacency + beginAddr, permutedFullAdjacency + endAddr);
   }

   // Setting counts to zero
   for(int aIt = 0; aIt < finePartCount; aIt++)
   {
      aggregateAdjacency[aIt] = 1;
   }

   // Counting the distinct neighbors of each aggregate:
   for(int aIt = 0; aIt < finePartCount; aIt++)
   {
      int begin = vertexAdjacencyOffsets[ nextAggregateOffset[aIt] ];
      int end = vertexAdjacencyOffsets[ nextAggregateOffset[aIt + 1] ];
      for(int i = begin + 1; i < end; i++)
      {
         if(permutedFullAdjacency[i] < notNeighbor && permutedFullAdjacency[i - 1] != permutedFullAdjacency[i])
         {
            permutedFullAdjacency[begin + aggregateAdjacency[aIt]] = permutedFullAdjacency[i];
            aggregateAdjacency[aIt]++;
         }
      }
   }

   // Finding the offsets for the aggregate adjacency
   newAdjacencyIndexes[0] = 0;
   for(int aIt = 1; aIt < finePartCount + 1; aIt++)
   {
      newAdjacencyIndexes[aIt] = newAdjacencyIndexes[aIt - 1] + aggregateAdjacency[aIt - 1];
   }

   // Allocating the adjacency array
   newAdjacency = new int[newAdjacencyIndexes[finePartCount]];

   // Writing the new adjacency to the list:
   for(int aIt = 0; aIt < finePartCount; aIt++)
   {
      int oldOffset = vertexAdjacencyOffsets[ nextAggregateOffset[aIt] ];
      int newOffset = newAdjacencyIndexes[aIt];
      for(int i = 0; i < aggregateAdjacency[aIt]; i++)
      {
         newAdjacency[newOffset + i] = permutedFullAdjacency[oldOffset + i];
      }
   }

   // Allocating an array for the block partition:
   int *blockPartition = new int[finePartCount];
   int blockCount;

   // Setting the depth from parameter:
   depth = metissize % 10;

   // Calling extendedMIS to get the block partition:
   extendedMIS(finePartCount, depth, newAdjacencyIndexes, newAdjacency, blockPartition, &blockCount);

   // Allocating block level arrays:
   int *blockAggregateCounts = new int[blockCount];
   int *blockNeighborCounts = new int[blockCount];
   int *nextBlockOffset = new int[blockCount + 1];
   int *tempBlockOffset = new int[blockCount + 1];
   int *blockPermutationOldToNew = new int[finePartCount];
   int *blockPermutationNewToOld = new int[finePartCount];
   int *permutedBlockAdjacency = new int[newAdjacencyIndexes[finePartCount]];
   int *newAggregateNeighborCounts = new int [finePartCount];
   int *aggregateAdjacencyOffsets = new int[finePartCount + 1];
   partitionIdx.resize(blockCount + 1);

   // Clearing block counts
   for(int i = 0; i < blockCount; i++)
   {
      blockAggregateCounts[i] = 0;
      blockNeighborCounts[i] = 0;
   }

   // Counting aggregates in each block, and total neighbors
   for(int aIt = 0; aIt < finePartCount; aIt++)
   {
      blockAggregateCounts[blockPartition[aIt]]++;
      blockNeighborCounts[blockPartition[aIt]] += newAdjacencyIndexes[aIt + 1] - newAdjacencyIndexes[aIt];
   }

   // Calculating the new offsets of each block
   nextBlockOffset[0] = 0;
   tempBlockOffset[0] = 0;
   for(int bIt = 1; bIt < blockCount + 1; bIt++)
   {
      // Doing a prefix sum:
      nextBlockOffset[bIt] = nextBlockOffset[bIt - 1] + blockAggregateCounts[bIt - 1];
      tempBlockOffset[bIt] = nextBlockOffset[bIt - 1] + blockAggregateCounts[bIt - 1];
      partitionIdx[bIt] = nextBlockOffset[bIt - 1] + blockAggregateCounts[bIt - 1];
   }

   // Filling in the blockPermutation array:
   for(int aIt = 0; aIt < finePartCount; aIt++)
   {
      int blockID = blockPartition[aIt];
      blockPermutationOldToNew[aIt] = tempBlockOffset[blockID];
      blockPermutationNewToOld[tempBlockOffset[blockID]] = aIt;
      tempBlockOffset[blockID]++;
   }

   // Counting neighbors of each (permuted) aggregate
   for(int aIt = 0; aIt < finePartCount; aIt++)
   {
      int oldNodeID = blockPermutationNewToOld[aIt];
      newAggregateNeighborCounts[aIt] = newAdjacencyIndexes[oldNodeID + 1] - newAdjacencyIndexes[oldNodeID];
   }

   // Calculating the new aggregate offsets:
   aggregateAdjacencyOffsets[0] = 0;
   for(int aIt = 1; aIt < finePartCount + 1; aIt++)
   {
      aggregateAdjacencyOffsets[aIt] = aggregateAdjacencyOffsets[aIt - 1] + newAggregateNeighborCounts[aIt - 1];
   }

   // Filling in the permutedBlockAdjacency
   for(int aIt = 0; aIt < finePartCount; aIt++)
   {
      int permIdx = blockPermutationOldToNew[aIt];
      int newOffset = aggregateAdjacencyOffsets[permIdx];
      int oldOffset = newAdjacencyIndexes[aIt];
      for(int nIt = 0; nIt < newAggregateNeighborCounts[permIdx]; nIt++)
      {
         // Permute the neighbor's index and write to new location in array.
         permutedBlockAdjacency[newOffset + nIt] = blockPermutationOldToNew[newAdjacency[oldOffset + nIt]];
      }
   }
   // Finding the new permutation on the original vertices
   // Reusing the original arrays as they are no longer needed.
   permutation.resize(nn);
   ipermutation.resize(nn);
   int *permutedFineAggregate = new int[nn];
   aggregateIdx.resize(finePartCount + 1);

   // filling the permutedFineAggregate array:
   for(int vIt = 0; vIt < nn; vIt++)
   {
      permutedFineAggregate[vIt] = blockPermutationOldToNew[ fineAggregate[vIt] ];
   }

   // Clearing aggregate counts
   for(int i = 0; i < finePartCount; i++)
   {
      aggregateVertexCounts[i] = 0;
      aggregateNeighborCounts[i] = 0;
   }

   // Counting vertices in each aggregate, and total neighbors
   for(int vIt = 0; vIt < nn; vIt++)
   {
      aggregateVertexCounts[permutedFineAggregate[vIt]]++;
   }

   // Calculating the new offsets of each aggregate
   tempAggregateOffset[0] = 0;
   aggregateIdx[0] = 0;
   for(int aIt = 1; aIt < finePartCount + 1; aIt++)
   {
      // Doing a prefix sum:
      tempAggregateOffset[aIt] = tempAggregateOffset[aIt - 1] + aggregateVertexCounts[aIt - 1];
      aggregateIdx[aIt] = tempAggregateOffset[aIt - 1] + aggregateVertexCounts[aIt - 1];
   }

   // Filling in the finalPermutation array:
   for(int vIt = 0; vIt < nn; vIt++)
   {
      int aggID = permutedFineAggregate[vIt];
      permutation[vIt] = tempAggregateOffset[aggID];
      ipermutation[tempAggregateOffset[aggID]] = vIt;
      tempAggregateOffset[aggID]++;
   }


   // Setting values for return:
   *nnout = finePartCount;
   xadjout = aggregateAdjacencyOffsets;
   adjncyout = permutedBlockAdjacency;

   // Setting the partitionlabel:
   for(int i = 0; i < blockCount; i++)
   {
      int startAt = aggregateIdx[partitionIdx[i]];
      int nextBlockAt = aggregateIdx[partitionIdx[i + 1]];
      for(int j = startAt; j < nextBlockAt; j++)
         partitionlabel[j] = i;
   }
   // Deleting the temporary arrays:
   delete[] fineAggregate;
   delete[] permutedFineAggregate;
   delete[] blockAggregateCounts;
   delete[] blockNeighborCounts;
   delete[] nextBlockOffset;
   delete[] tempBlockOffset;
   delete[] blockPermutationOldToNew;
   delete[] blockPermutationNewToOld;
   delete[] newAggregateNeighborCounts;
   delete[] permutedFullAdjacency;
   delete[] initialPermutationOldToNew;
   delete[] initialPermutationNewToOld;
   delete[] permutationCheck;
   delete[] nextAggregateOffset;
   delete[] tempAggregateOffset;
   delete[] aggregateVertexCounts;
   delete[] vertexNeighborCounts;
   delete[] aggregateNeighborCounts;
   delete[] vertexAdjacencyOffsets;
   delete[] aggregateAdjacency;
   delete[] newAdjacencyIndexes;
   delete[] newAdjacency;

   // And Done.
   return;
}

template <class Matrix, class Vector>
void RandMIS_Aggregator<Matrix, Vector>::computePermutation_d(IdxVector_d &adjIndexesIn,
      IdxVector_d &adjacencyIn,
      IdxVector_d &permutation,
      IdxVector_d &ipermutation,
      IdxVector_d &aggregateIdx,
      IdxVector_d &partitionIdx,
      IdxVector_d &partitionLabel,
      IdxVector_d &adjIndexesOut,
      IdxVector_d &adjacencyOut,
      int agg_type,
      int parameters,
      int part_max_size,
      bool verbose) {
  if (agg_type == 0)
   {
      if (verbose)
         printf("Calling Old MIS Aggregation method.\n");
      misHelpers::CP::OldMIS(adjIndexesIn,
            adjacencyIn,
            permutation,
            ipermutation,
            aggregateIdx,
            partitionIdx,
            partitionLabel,
            adjIndexesOut,
            adjacencyOut,
            parameters,
            part_max_size,
            verbose);
   }
  else if (agg_type == 1)
   {
      if (verbose)
         printf("Calling Metis bottom up method\n");
      misHelpers::CP::MetisBottomUp(adjIndexesIn,
            adjacencyIn,
            permutation,
            ipermutation,
            aggregateIdx,
            partitionIdx,
            partitionLabel,
            adjIndexesOut,
            adjacencyOut,
            parameters,
            part_max_size,
            verbose);
   }
  else if (agg_type == 2)
   {
      if (verbose)
         printf("Calling Metis top down method\n");
      misHelpers::CP::MetisTopDown(adjIndexesIn,
            adjacencyIn,
            permutation,
            ipermutation,
            aggregateIdx,
            partitionIdx,
            partitionLabel,
            adjIndexesOut,
            adjacencyOut,
            parameters,
            part_max_size,
            verbose);
   }
  else if (agg_type == 3)
   {
      if (verbose)
         printf("Calling AggMIS GPU method\n");
      misHelpers::CP::NewMIS(adjIndexesIn,
            adjacencyIn,
            permutation,
            ipermutation,
            aggregateIdx,
            partitionIdx,
            partitionLabel,
            adjIndexesOut,
            adjacencyOut,
            parameters,
            part_max_size,
            verbose);
   }
  else if (agg_type == 4)
   {
      if (verbose)
         printf("Calling AggMIS CPU method\n");
      misHelpers::CP::NewMIS_CPU(adjIndexesIn,
            adjacencyIn,
            permutation,
            ipermutation,
            aggregateIdx,
            partitionIdx,
            partitionLabel,
            adjIndexesOut,
            adjacencyOut,
            parameters,
            part_max_size,
            verbose);
   }
  else if (agg_type == 5)
   {
      if (verbose)
         printf("Calling AggMIS Light CPU method\n");
      misHelpers::CP::LightMIS_CPU(adjIndexesIn,
            adjacencyIn,
            permutation,
            ipermutation,
            aggregateIdx,
            partitionIdx,
            partitionLabel,
            adjIndexesOut,
            adjacencyOut,
            parameters,
            part_max_size,
            verbose);
   }
   else if (verbose)
     printf("Aggregation method %d not recognized!\n", agg_type);

   if (verbose)
      std::cout << "Finished with RandMIS_Aggregator::computePermutation_d" << std::endl;
}

template <class Matrix, class Vector>
void RandMIS_Aggregator<Matrix, Vector>::computePermutation_d(TriMesh *meshPtr,
  IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx,
  IdxVector_d &partitionIdx, IdxVector_d &partitionLabel, IdxVector_d &adjIndexesOut, 
  IdxVector_d &adjacencyOut, int aggregation_type, int parameters, int part_max_size, bool verbose)
{
   IdxVector_d adjIndexesIn, adjacencyIn;
   misHelpers::getAdjacency(meshPtr, adjIndexesIn, adjacencyIn);
   computePermutation_d(adjIndexesIn, adjacencyIn, permutation, ipermutation, 
     aggregateIdx, partitionIdx, partitionLabel, adjIndexesOut,
     adjacencyOut, aggregation_type, parameters, part_max_size, verbose);
}

template <class Matrix, class Vector>
void RandMIS_Aggregator<Matrix, Vector>::computePermutation_d(TetMesh *meshPtr, 
  IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx,
  IdxVector_d &partitionIdx, IdxVector_d &partitionLabel, IdxVector_d &adjIndexesOut,
  IdxVector_d &adjacencyOut, int aggregation_type, int parameters, int part_max_size, bool verbose)
{
   IdxVector_d adjIndexesIn, adjacencyIn;
   misHelpers::getAdjacency(meshPtr, adjIndexesIn, adjacencyIn);
   computePermutation_d(adjIndexesIn, adjacencyIn, permutation, ipermutation,
     aggregateIdx, partitionIdx, partitionLabel, adjIndexesOut, adjacencyOut,
     aggregation_type, parameters, part_max_size, verbose);
}

template <class Matrix, class Vector>
void RandMIS_Aggregator<Matrix, Vector>::extendedMIS(int n, int partSize, int *adjIndexes, int *adjacency, int *partition, int *partCount, bool verbose)
{
   // If the input is too small just return a single partition
   if (n < 32)
   {
      for (int i = 0; i < n; i++)
         partition[i] = 0;
      *partCount = 1;
      return;
   }
   clock_t starttime, endtime;
   starttime = clock();
   if (verbose)
      printf("Beginning extended MIS call %d nodes\n", n);
   // Creating a graph with edges for every distinct path less than k
   vector< vector <int> > inducedAdj(n);


   vector<int> clusterCounts(n);
   // Every vertex
   for(int vIt = 0; vIt < n; vIt++)
   {
      vector< vector <int> > nodeRings(partSize);
      // Add neighbors to nodeRing 0
      for(int nIt = adjIndexes[vIt]; nIt < adjIndexes[vIt + 1]; nIt++)
      {
         nodeRings[0].push_back(adjacency[nIt]);
      }

      // For every level of nodeRings
      for(int level = 1; level < nodeRings.size(); level++)
      {
         // Every node at the previous level:
         for(int lowerNode = 0; lowerNode < nodeRings[level - 1].size(); lowerNode++)
         {
            // Every neighbor of lower nodes
            int currentNode = nodeRings[level - 1][lowerNode];
            for(int nIt = adjIndexes[currentNode]; nIt < adjIndexes[currentNode + 1]; nIt++)
            {
               int candidate = adjacency[nIt];

               // Checking the candidate is not the root...
               if(candidate != vIt)
               {
                  // If the node is not present in nodeRings add to current level
                  for(int i = 0; i <= level && candidate != -1; i++)
                  {
                     if(nodeRings[i].size() == 0)
                        nodeRings[i].push_back(candidate);

                     for(int j = 0; j < nodeRings[i].size(); j++)
                        if(nodeRings[i][j] == candidate)
                           candidate = -1;
                  }

                  if(candidate != -1)
                  {
                     nodeRings[level].push_back(candidate);
                  }
               }
            }
         }
      }

      // Now that nodeRings are populated add edges to all nodes in upper level (k-path's)
      int clusterCount = 1;
      for(int i = 0; i < nodeRings.size(); i++)
      {
         for(int j = 0; j < nodeRings[i].size(); j++)
         {
            inducedAdj[vIt].push_back(nodeRings[i][j]);
            clusterCount++;
         }
      }
      clusterCounts[vIt] = clusterCount;
   }

   if (verbose)
      printf("Finished generating induced graph.\n");

   // Calculating average cluster count to determine random weighting
   int totalClusterCount = 0;
   int maxDegree = clusterCounts[0];
   for(int i = 0; i < clusterCounts.size(); i++)
   {
      totalClusterCount += clusterCounts[i];
      if(maxDegree < clusterCounts[i])
         maxDegree = clusterCounts[i];
   }
   double averageClusterSize = (double)totalClusterCount / (double)clusterCounts.size();
   double probPositive = (1.0 / (averageClusterSize + 1.0));

   if (verbose)
      printf("ProbPositive = %f\n", probPositive);

   //printf("Random Weight is: %f\n", randWeight);

   // Clearing partitions:
   for(int i = 0; i < n; i++)
   {
      partition[i] = -1;
   }

   // Finding a maximal independent set randomly:
   vector<int> MIS(n, -1);
   vector<double> RandValues(n);
   vector<double> randThreshold(n);
   vector<int> rootDistance(n, -1);

   // Setting probability thresholds for each node based on degree
   for(int vIt = 0; vIt < n; vIt++)
   {
      // The degreeFactor is the percent difference between this degree and average
      //        double degreeFactor = clusterCounts[vIt] - averageClusterSize;
      //        if (degreeFactor < 0)
      //            degreeFactor *= - 1;

      double degreeFactor = (double)clusterCounts[vIt] / (double)maxDegree;
      degreeFactor /= averageClusterSize;
      degreeFactor = 1 - degreeFactor;

      if(degreeFactor < .1)
         printf("Low degreeFactor: %f degree: %d average degree: %f\n", degreeFactor, clusterCounts[vIt], averageClusterSize);

      // The threshold value is the probPositive times the degreeFactor
      randThreshold[vIt] = degreeFactor * probPositive;

      if(randThreshold[vIt] > 1 || randThreshold[vIt] < 0)
         printf("Random threshold out of range: %f degreeFactor = %f probPositive = %f!\n", randThreshold[vIt], degreeFactor, probPositive);
   }

   if (verbose)
      printf("Finished generation of random thresholds.\n");

   bool incomplete = true;
   //  srand(time(NULL));
   srand(0);
   int iterations = 0;
   while(incomplete)
   {
      iterations++;
      if(iterations > 10000)
      {
         printf("Something seems to be going wrong with the random assignments!\n");
         for(int i = 0; i < n; i++)
            partition[i] = 0;
         *partCount = 1;
         return;
      }
      // Maybe we are done?
      incomplete = false;

      // Independent for loop
      for(int i = 0; i < n; i++)
      {
         if(MIS[i] == -1)
         {
            // This should assign to a random value between 0 and 1
            double randValue = (double)rand() / (double)(RAND_MAX);

            // If the value is below the randThreshold than 1 else -1
            if(randValue < randThreshold[i])
               RandValues[i] = 1;
            else
               RandValues[i] = -1;

            // There is still work to do
            incomplete = true;
         }
         else if(MIS[i] == 1)
         {
            RandValues[i] = 1;
         }
         else
         {
            RandValues[i] = -1;
         }
      }

      // Independent for loop
      for(int i = 0; i < n && incomplete; i++)
      {
         if(RandValues[i] > 0)
         {
            bool negativeNeighbors = true;
            for(int j = 0; j < inducedAdj[i].size(); j++)
            {
               if(RandValues[inducedAdj[i][j]] > 0)
                  negativeNeighbors = false;
            }
            if(negativeNeighbors)
            {
               // Mark the node as in MIS
               MIS[i] = 1;
               rootDistance[i] = 0;

               // Mark all neighbors as out
               for(int j = 0; j < inducedAdj[i].size(); j++)
               {
                  MIS[inducedAdj[i][j]] = 0;
                  //rootDistance[i] = 1;
               }
            }
         }
      }
   }

   if (verbose)
      printf("Found a MIS of the graph in %d iterations.\n", iterations);


   // Setting each member of the independent set to be the root of a partition
   vector<int> rootNodes;
   int curPart = 0;
   for(int i = 0; i < n; i++)
   {
      if(MIS[i] == 1)
      {
         partition[i] = curPart;
         rootNodes.push_back(i);
         curPart++;
      }
   }

   // Setting the partCount
   *partCount = curPart;

   // An array to hold partition assignments to apply
   int *newPartition = new int[n];
   for(int i = 0; i < n; i++)
      newPartition[i] = partition[i];
   vector<int> partSizes(curPart, 1);

   // Adding unpartitioned nodes to best partition for them:
   incomplete = true;
   int its = 0;

   // new rootDistance array
   int *newRootDist = new int[n];
   for(int i = 0; i < n; i++)
      newRootDist[i] = rootDistance[i];

   while(incomplete)
   {
      incomplete = false;
      its++;

      // If this has been going on too long:
      if (its > 2 * n)
      {
         printf("There was an error in the node allocation section: Too many iterations!\n");
         for (int i = 0; i < n; i++)
            partition[i] = 0;
         *partCount = 1;
         return;
      }

      for(int i = 0; i < n; i++)
      {
         if(partition[i] == -1)
         {
            int adjSize = adjIndexes[i + 1] - adjIndexes[i];
            //        printf("adjSize is:  %d\n", adjSize);
            int *adjParts = new int[adjSize];
            int *adjRootDist = new int[adjSize];
            int *adjSizes = new int[adjSize];
            int *adjCounts = new int[adjSize];
            int *adjScore = new int[adjSize];

            // Getting adjacent partitions:
            for(int j = 0; j < adjSize; j++)
            {
               int adjacentNodePart = partition[ adjacency[adjIndexes[i] + j] ];
               adjParts[j] = adjacentNodePart;

               // Getting the size of the aggregate
               adjSizes[j] = partSizes[adjacentNodePart];

               // Getting the distance of the adjacent node to the root of its partition:
               if(adjacentNodePart == -1)
               {
                  adjRootDist[j] = 1000;
               }
               else
               {
                  adjRootDist[j] = rootDistance[adjacency[adjIndexes[i] + j] ];
               }
            }

            // Finding the smallest partition distance:
            int smallestDistance = 1000;
            int largestDistance = 0;
            int largestSize = adjSizes[0];
            int smallestSize = adjSizes[0];
            for(int j = 0; j < adjSize; j++)
               adjCounts[j] = 1;
            for(int j = 0; j < adjSize; j++)
            {
               if(adjParts[j] != -1)
               {
                  if(smallestDistance > adjRootDist[j])
                  {
                     smallestDistance = adjRootDist[j];
                  }
                  if(smallestSize > adjSizes[j])
                  {
                     smallestSize = adjSizes[j];
                  }
                  if(adjRootDist[j] < 1000 && largestDistance < adjRootDist[j])
                  {
                     largestDistance = adjRootDist[j];
                  }
                  if(largestSize > adjSizes[j])
                  {
                     largestSize = adjSizes[j];
                  }

                  for(int jj = j + 1; jj < adjSize; jj++)
                  {
                     if(adjParts[j] == adjParts[jj] && adjCounts[j] < 3)
                     {
                        adjCounts[j]++;
                        adjCounts[jj]++;
                     }
                  }
               }
            }

            // Calculating score factor for each entry:
            double highestScore = -1;
            int scoringPart = -1;
            for(int j = 0; j < adjSize; j++)
            {
               if(adjParts[j] != -1)
               {
                  double sizeScore = 1.0 / (adjSizes[j] - smallestSize + 1);
                  double distScore = ((double)smallestDistance + 1) / (adjRootDist[j] + 1);
                  double adjScore = std::pow(0.75, 4. - static_cast<double>(adjCounts[j]));
                  double totalScore = (sizeScore + distScore) * adjScore;
                  if(totalScore > highestScore)
                  {
                     highestScore = totalScore;
                     scoringPart = adjParts[j];
                  }
               }
            }

            // Adding the node to best part found:
            newPartition[i] = scoringPart;
            if(scoringPart >= 0)
            {
               partSizes[scoringPart]++;
               newRootDist[i] = smallestDistance + 1;
            }
            else
               incomplete = true;


            delete [] adjParts;
            delete [] adjRootDist;
            delete [] adjSizes;
            delete [] adjCounts;
            delete [] adjScore;
         }
      }

      // Write changes to partition:
      for(int i = 0; i < n; i++)
      {
         partition[i] = newPartition[i];
         rootDistance[i] = newRootDist[i];
      }

      if(!incomplete)
      {
         // To store the parts that are too small:
         vector<int> partsToRemove;

         // Check for too small partitions
         for(int i = 0; i < partSizes.size(); i++)
         {
            if(partSizes[i] < 6)
            {
               partsToRemove.push_back(i);
               incomplete = true;
            }
         }

         if (partsToRemove.size() != partSizes.size())
         {
            if (verbose)
               printf("Starting removal of runty parts.\n");
            int originalPartCount = *partCount;
            // Removing runty aggregates:
            for(int i = partsToRemove.size() - 1; i > -1; i--)
            {
               // Unmark the partition label and rootnode dist
               for(int j = 0; j < n; j++)
               {
                  if(partition[j] == partsToRemove[i])
                  {
                     partition[j] = -1;
                     newPartition[j] = -1;
                     rootDistance[j] = -1;
                     newRootDist[j] = -1;
                  }
                  if(partition[j] > partsToRemove[i])
                  {
                     partition[j] = partition[j] - 1;
                     newPartition[j] = newPartition[j] - 1;
                  }

               }

               // Remove the entry from the partSizes array
               *partCount = *partCount - 1;
               partSizes.erase(partSizes.begin() + partsToRemove[i]);
            }
            if (verbose)
               printf("Removed %lu undersized aggregates out of %d total. Leaving %d\n", partsToRemove.size(), originalPartCount, *partCount);
         }
      }


      if(!incomplete && verbose)
      {
         endtime = clock();
         double duration = (double)(endtime - starttime) * 1000 / CLOCKS_PER_SEC;
         printf("Finished with a call to extendedMIS in %f ms\n", duration);
         printf("\t%d nodes aggregated to depth: %d \n\n", n, partSize);
      }
   }

   delete [] newRootDist;
   if (verbose) printf("GoodBye.\n");
}


/****************************************
 * Explicit instantiations
 ***************************************/
template class RandMIS_Aggregator<Matrix_h, Vector_h>;
template class RandMIS_Aggregator<Matrix_d, Vector_d>;
