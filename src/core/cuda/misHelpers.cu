#include <smoothedMG/aggregators/misHelpers.h>
//#include "thrust/detail/device_ptr.inl"

__global__ void findAdjacencySizesKernel(int size, int *adjIndexes, int *output)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      output[idx] = adjIndexes[idx + 1] - adjIndexes[idx];
   }
}

__global__ void allocateNodesKernel(int size, int *adjIndexes, int *adjacency, int *partIn, int *partOut, int *aggregated)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      if(aggregated[idx] == 0)
      {
         int start = adjIndexes[idx];
         int end = adjIndexes[idx + 1];

         // Storage for possible aggregations.
         int candidates[10];
         int candidateCounts[10];
         for(int i = 0; i < 10; i++)
         {
            candidates[i] = -1;
            candidateCounts[i] = 0;
         }

         // Going through neighbors to aggregate:
         for(int i = start; i < end; i++)
         {
            int candidate = partIn[ adjacency[i] ];
            if(candidate != -1)
            {
               for(int j = 0; j < 10 && candidate != -1; j++)
               {
                  if(candidates[j] == -1)
                  {
                     candidates[j] = candidate;
                     candidateCounts[j] = 1;
                  }
                  else
                  {
                     if(candidates[j] == candidate)
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
         for(int i = 1; i < 10; i++)
         {
            if(candidateCounts[i] > count)
            {
               count = candidateCounts[i];
               addTo = candidates[i];
            }
         }
         partOut[idx] = addTo;
         if(addTo != -1)
         {
            aggregated[idx] = 1;
         }
      }
   }
}

__global__ void findPartIndicesKernel(int size, int *array, int *partIndices)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < size)
   {
      int value = array[idx];
      int nextValue = (idx != size - 1) ? array[idx + 1] : -1;
      if (value != nextValue)
      {
         partIndices[value + 1] = idx + 1;
      }
   }
}

__global__ void findPartIndicesNegStartKernel(int size, int *array, int *partIndices)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
   if(idx < size)
   {
      int value = array[idx];
      int nextValue = array[idx + 1];
      if(value != nextValue)
         partIndices[value + 1] = idx;
   }
}

__global__ void fillWithIndexKernel(int size, int *array)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      array[idx] = idx;
   }
}

__global__ void getInversePermutationKernel(int size, int *original, int *inverse)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      inverse[original[idx]] = idx;
   }
}

__global__ void permuteInitialAdjacencyKernel(int size, int *adjIndexesIn, int *adjacencyIn, int *permutedAdjIndexesIn, int *permutedAdjacencyIn, int *ipermutation, int *fineAggregate)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      int oldBegin = adjIndexesIn[ipermutation[idx]];
      int oldEnd = adjIndexesIn[ipermutation[idx] + 1];
      int runSize = oldEnd - oldBegin;
      int newBegin = permutedAdjIndexesIn[idx];
      //int newEnd = permutedAdjIndexesIn[idx + 1];
      //int newRunSize = newEnd - newBegin;

      //printf("Thread %d is copying from %d through %d into %d through %d\n", idx, oldBegin, oldEnd, newBegin, newEnd);

      // Transfer old adjacency into new, while changing node id's with partition id's
      for(int i = 0; i < runSize; i++)
      {
         permutedAdjacencyIn[newBegin + i] = fineAggregate[ adjacencyIn[oldBegin + i] ];
      }
   }
}

__global__ void getInducedGraphNeighborCountsKernel(int size, int *aggregateIdx, int *adjIndexesOut, int *permutedAdjIndexes, int *permutedAdjacencyIn)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      int Begin = permutedAdjIndexes[ aggregateIdx[idx] ];
      int End = permutedAdjIndexes[ aggregateIdx[idx + 1] ];

      // Sort each section of the adjacency:
      for(int i = Begin; i < End - 1; i++)
      {
         for(int ii = i + 1; ii < End; ii++)
         {
            if(permutedAdjacencyIn[i] < permutedAdjacencyIn[ii])
            {
               int temp = permutedAdjacencyIn[i];
               permutedAdjacencyIn[i] = permutedAdjacencyIn[ii];
               permutedAdjacencyIn[ii] = temp;
            }
         }
      }

      // Scan through the sorted adjacency to get the condensed adjacency:
      int neighborCount = 1;
      if(permutedAdjacencyIn[Begin] == idx)
         neighborCount = 0;

      for(int i = Begin + 1; i < End; i++)
      {
         if(permutedAdjacencyIn[i] != permutedAdjacencyIn[i - 1] && permutedAdjacencyIn[i] != idx)
         {
            permutedAdjacencyIn[neighborCount + Begin] = permutedAdjacencyIn[i];
            neighborCount++;
         }
      }

      // Store the size
      adjIndexesOut[idx] = neighborCount;
   }
}

__global__ void fillCondensedAdjacencyKernel(int size, int *aggregateIdx, int *adjIndexesOut, int *adjacencyOut, int *permutedAdjIndexesIn, int *permutedAdjacencyIn)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      int oldBegin = permutedAdjIndexesIn[ aggregateIdx[idx] ];
      int newBegin = adjIndexesOut[idx];
      int runSize = adjIndexesOut[idx + 1] - newBegin;

      // Copy adjacency over
      for(int i = 0; i < runSize; i++)
      {
         adjacencyOut[newBegin + i] = permutedAdjacencyIn[oldBegin + i];
      }
   }
}

__global__ void fillPartitionLabelKernel(int size, int *coarseAggregate, int *fineAggregateSort, int *partitionLabel)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      partitionLabel[idx] = coarseAggregate[ fineAggregateSort[idx] ];
   }
}

__global__ void getAggregateStartIndicesKernel(int size, int *fineAggregateSort, int *aggregateRemapIndex)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      if(idx == 0 || fineAggregateSort[idx] != fineAggregateSort[idx - 1])
      {
         aggregateRemapIndex[fineAggregateSort[idx]] = idx;
      }
   }
}

__global__ void remapAggregateIdxKernel(int size, int *fineAggregateSort, int *aggregateRemapId)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      fineAggregateSort[idx] = aggregateRemapId[fineAggregateSort[idx]];
   }
}

__global__ void mapAdjacencyToBlockKernel(int size, int *adjIndexes, int *adjacency, int *adjacencyBlockLabel, int *blockMappedAdjacency, int *fineAggregate)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      int begin = adjIndexes[idx];
      int end = adjIndexes[idx + 1];
      int thisBlock = fineAggregate[idx];

      // Fill block labeled adjacency and block mapped adjacency vectors
      for(int i = begin; i < end; i++)
      {
         int neighbor = fineAggregate[adjacency[i]];

         if(thisBlock == neighbor)
         {
            adjacencyBlockLabel[i] = -1;
            blockMappedAdjacency[i] = -1;
         }
         else
         {
            adjacencyBlockLabel[i] = thisBlock;
            blockMappedAdjacency[i] = neighbor;
         }
      }
   }
}

__global__ void removeRuntyPartsKernel(int size, int *partition, int *removeStencil, int *subtractions)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      int currentNode = partition[idx];
      if(removeStencil[currentNode] == 1)
         partition[idx] = -1;
      else
         partition[idx] -= subtractions[currentNode];
   }
}

__global__ void accumulatedPartSizesKernel(int size, int *part, int *weights, int *accumulatedSize)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx == size - 1)
      accumulatedSize[part[idx]] = weights[idx];
   if(idx < size - 1)
   {
      int thisPart = part[idx];
      if(thisPart != part[idx + 1])
         accumulatedSize[thisPart] = weights[idx];
   }
}

__global__ void unaccumulatedPartSizesKernel(int size, int *accumulatedSize, int *sizes)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx == 0)
      sizes[idx] = accumulatedSize[0];

   else if(idx < size)
   {
      sizes[idx] = accumulatedSize[idx] - accumulatedSize[idx - 1];
   }
}

__global__ void findDesirabilityKernel(int size, int optimalSize, int *adjIndexes, int *adjacency, int *partition, int *partSizes, int *nodeWeights, int *swap_to, int *swap_from, int *swap_index, float *desirability)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < size)
   {
      int currentPart = partition[idx];
      int currentPartSize = partSizes[currentPart];
      int nodeSize = nodeWeights[idx];
      int selfAdjacency = 0;
      int addTo = -1;
      float bestDesirability = 0;

      // The currentWeightFactor is higher the farther the count is from average
      float currentWeightFactor = (float)abs(currentPartSize - optimalSize) / optimalSize;
      // The self improvement is a measure of how much better this partitions size will be if the node is gone.
      float selfImprovement = (abs(currentPartSize - optimalSize) - abs((currentPartSize - nodeSize) - optimalSize)) * currentWeightFactor;
      if(selfImprovement > 0)
      {
         int start = adjIndexes[idx];
         int end = adjIndexes[idx + 1];

         // Arrays to store info about neighboring aggregates
         int candidates[10];
         int candidateCounts[10];
         for(int i = 0; i < 10; i++)
         {
            candidates[i] = -1;
            candidateCounts[i] = 0;
         }

         // Going through the neighbors:
         for(int i = start; i < end; i++)
         {
            int candidate = partition[ adjacency[i] ];
            if(candidate == currentPart)
               selfAdjacency++;
            else
               for(int j = 0; j < 10; j++)
               {
                  if(candidate != -1 && candidates[j] == -1)
                  {
                     candidates[j] = candidate;
                     candidateCounts[j] = 1;
                     candidate = -1;
                  }
                  else if(candidates[j] == candidate)
                  {
                     candidateCounts[j] += 1;
                     candidate = -1;
                  }
               }
         }

         // Finding the best possible swap:
         for(int i = 1; i < 10; i++)
         {
            if(candidates[i] != -1)
            {
               int neighborPart = candidates[i];
               int neighborPartSize = partSizes[neighborPart];
               float neighborWeightFactor = (float)abs(neighborPartSize - optimalSize) / optimalSize;
               float neighborImprovement = ((float)(abs(neighborPartSize - optimalSize) - abs((neighborPartSize + nodeSize) - optimalSize))) * neighborWeightFactor;
               // Combining with self improvement to get net
               neighborImprovement += selfImprovement;
               // Multiplying by adjacency factor
               neighborImprovement *= (float)candidateCounts[i] / selfAdjacency;

               if(neighborImprovement > bestDesirability)
               {
                  addTo = neighborPart;
                  bestDesirability = neighborImprovement;
               }
            }
         }
      }

      swap_from[idx] = currentPart;
      swap_index[idx] = idx;
      swap_to[idx] = addTo;
      desirability[idx] = bestDesirability;
   }
}

__global__ void makeSwapsKernel(int size, int *partition, int *partSizes, int *nodeWeights, int *swap_to, int *swap_from, int *swap_index, float *desirability)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx == size - 1)
   {
      if(desirability[idx] > .1)
      {
         int swapTo = swap_to[idx];
         int swapFrom = swap_from[idx];
         int swapIndex = swap_index[idx];
         int nodeWeight = nodeWeights[swapIndex];
         partition[swapIndex] = swapTo;
         atomicAdd(&partSizes[swapTo], nodeWeight);
         atomicAdd(&partSizes[swapFrom], -nodeWeight);
         //printf("Swapping node: %d, %d from part: %d, %d to part: %d, %d desirability: %f\n", swapIndex, nodeWeight, swapFrom, partSizes[swapFrom], swapTo, partSizes[swapTo], desirability[idx]);
      }
   }

   else if(idx < size - 1)
   {
      if(desirability[idx] > .1 && swap_from[idx] != swap_from[idx + 1])
      {
         int swapTo = swap_to[idx];
         int swapFrom = swap_from[idx];
         int swapIndex = swap_index[idx];
         int nodeWeight = nodeWeights[swapIndex];
         partition[swapIndex] = swapTo;
         atomicAdd(&partSizes[swapTo], nodeWeight);
         atomicAdd(&partSizes[swapFrom], -nodeWeight);
         //printf("Swapping node: %d, %d from part: %d, %d to part: %d, %d desirability: %f\n", swapIndex, nodeWeight, swapFrom, partSizes[swapFrom], swapTo, partSizes[swapTo], desirability[idx]);
      }
   }
}

void misHelpers::getMIS(IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &misStencil, int depth)
{
   IdxVector_d mtxValues_d(adjacency.size(), 1);
   int vSize = adjIndexes.size() - 1;
   //    IdxVector_h tmp = misStencil;
   //    for(int i=0; i<10; i++)
   //    {
   //      printf("%d\n", tmp[i]);
   //    }

   // Creating a matrix with the vectors supplied:
   //    devMtx graphy(vSize, vSize, adjacency.size());
   //    graphy.column_indices = adjacency;
   //    cusp::detail::offsets_to_indices(adjIndexes , graphy.row_indices);
   //    graphy.values = mtxValues_d;
   //    cusp::print(graphy);
   cusp::csr_matrix<int, int, cusp::device_memory> graphy(vSize, vSize, adjacency.size());
   graphy.column_indices = adjacency;
   graphy.row_offsets = adjIndexes;

   cusp::graph::maximal_independent_set(graphy, misStencil, depth);
   //    tmp = misStencil;
   //    for(int i=0; i<50; i++)
   //    {
   //      printf("%d\n", tmp[i]);
   //    }
   graphy.resize(0, 0, 0);
}

void misHelpers::getAdjacency(TriMesh *meshPtr, IdxVector_d &adjIndexes, IdxVector_d &adjacency)
{
   int vSize = meshPtr->vertices.size();
   meshPtr->need_neighbors();

   // Finding total size of adjacency list:
   int adjacencySize = 0;
   for(int i = 0; i < vSize; i++)
   {
      adjacencySize += meshPtr->neighbors[i].size();
   }

   // Vectors to hold the adjacency:
   IdxVector_h adjIndexes_h(vSize + 1);
   IdxVector_h adjacency_h(adjacencySize);

   // Populating adjacency
   adjIndexes_h[0] = 0;
   ;

   int nextOffset = 0;
   for(int i = 0; i < vSize; i++)
   {
      for(int j = 0; j < meshPtr->neighbors[i].size(); j++)
         adjacency_h[nextOffset + j] = meshPtr->neighbors[i][j];

      nextOffset += meshPtr->neighbors[i].size();
      adjIndexes_h[i + 1] = nextOffset;
   }

   // Copying to device vectors
   adjIndexes = adjIndexes_h;
   adjacency = adjacency_h;
}

void misHelpers::getAdjacency(TetMesh *meshPtr, IdxVector_d &adjIndexes, IdxVector_d &adjacency)
{
   int vSize = meshPtr->vertices.size();
   meshPtr->need_neighbors();

   // Finding total size of adjacency list:
   int adjacencySize = 0;
   for(int i = 0; i < vSize; i++)
   {
      adjacencySize += meshPtr->neighbors[i].size();
   }

   // Vectors to hold the adjacency:
   IdxVector_h adjIndexes_h(vSize + 1);
   IdxVector_h adjacency_h(adjacencySize);

   // Populating adjacency
   adjIndexes_h[0] = 0;
   ;

   int nextOffset = 0;
   for(int i = 0; i < vSize; i++)
   {
      for(int j = 0; j < meshPtr->neighbors[i].size(); j++)
         adjacency_h[nextOffset + j] = meshPtr->neighbors[i][j];

      nextOffset += meshPtr->neighbors[i].size();
      adjIndexes_h[i + 1] = nextOffset;
   }

   // Copying to device vectors
   adjIndexes = adjIndexes_h;
   adjacency = adjacency_h;
}

void misHelpers::aggregateGraph(int minSize, int depth, IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &partIn)
{
   int size = adjIndexes.size() - 1;

   // Get an MIS for the graph:
   //  getMIS(adjIndexes, adjacency, partIn, depth);
   randomizedMIS(adjIndexes, adjacency, partIn, depth);

   IdxVector_d aggregated = partIn;
   IdxVector_d partOut;

   // Prefix sum to number aggregate roots:
   thrust::inclusive_scan(partIn.begin(), partIn.end(), partIn.begin());

   int misCount = partIn.back();
   //  DataRecorder::Add("Fine MIS Count", misCount);

   // Transform non root nodes to -1
   thrust::transform(partIn.begin(), partIn.end(), aggregated.begin(), partIn.begin(), ifLabelOne());
   partOut = partIn;

   // Preparing to call aggregate kernel:
   int *partIn_d; // Pointer to partIn vector
   int *partOut_d; // Pointer to partOut vector
   int *adjIndexes_d; // Pointer to adjacency indexes
   int *adjacency_d; // Pointer to adjacency
   int *aggregated_d; // Pointer to aggregated
   bool complete = false; // Indicates whether all nodes are aggregated

   partIn_d = thrust::raw_pointer_cast(&partIn[0]);
   partOut_d = thrust::raw_pointer_cast(&partOut[0]);
   adjIndexes_d = thrust::raw_pointer_cast(&adjIndexes[0]);
   adjacency_d = thrust::raw_pointer_cast(&adjacency[0]);
   aggregated_d = thrust::raw_pointer_cast(&aggregated[0]);

   // Figuring out block sizes for kernel call:
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   while(!complete)
   {
      // Allocating nodes
      allocateNodesKernel <<< nBlocks, blockSize >>> (size, adjIndexes_d, adjacency_d, partIn_d, partOut_d, aggregated_d);

      // Copying partOut to partIn
      partIn = partOut;

      // Checking if done
      int unallocatedNodes = thrust::count(aggregated.begin(), aggregated.end(), 0);
      if(unallocatedNodes == 0)
      {
         // Trying to remove parts below minSize
         complete = removeRuntyParts(minSize, partIn);

         // If stuff was removed get the aggregated labeling again
         if(!complete)
         {
            thrust::transform(partIn.begin(), partIn.end(), aggregated.begin(), findAggregated());
            partOut = partIn;
         }
      }
   }
}

void misHelpers::aggregateWeightedGraph(int maxSize, int fullSize, int depth, IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &partIn, IdxVector_d &nodeWeights, bool verbose)
{
   int size = adjIndexes.size() - 1;

   // Get an MIS for the graph:
   //  getMIS(adjIndexes, adjacency, partIn, depth);
   randomizedMIS(adjIndexes, adjacency, partIn, depth);

   IdxVector_d aggregated = partIn;
   IdxVector_d partOut;

   // Prefix sum to number aggregate roots:
   thrust::inclusive_scan(partIn.begin(), partIn.end(), partIn.begin());

   int misCount = partIn.back();
   //  DataRecorder::Add("Coarse MIS Count", misCount);

   // Transform non root nodes to -1
   thrust::transform(partIn.begin(), partIn.end(), aggregated.begin(), partIn.begin(), ifLabelOne());
   partOut = partIn;

   // Preparing to call aggregate kernel:
   int *partIn_d; // Pointer to partIn vector
   int *partOut_d; // Pointer to partOut vector
   int *adjIndexes_d; // Pointer to adjacency indexes
   int *adjacency_d; // Pointer to adjacency
   int *aggregated_d; // Pointer to aggregated
   bool complete = false; // Indicates whether all nodes are aggregated

   partIn_d = thrust::raw_pointer_cast(&partIn[0]);
   partOut_d = thrust::raw_pointer_cast(&partOut[0]);
   adjIndexes_d = thrust::raw_pointer_cast(&adjIndexes[0]);
   adjacency_d = thrust::raw_pointer_cast(&adjacency[0]);
   aggregated_d = thrust::raw_pointer_cast(&aggregated[0]);

   // Figuring out block sizes for kernel call:
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);
   bool firstTime = true;
   int counter = 0;

   while(!complete)
   {
      counter++;
      // Allocating nodes
      allocateNodesKernel << < nBlocks, blockSize >> > (size, adjIndexes_d, adjacency_d, partIn_d, partOut_d, aggregated_d);

      // Copying partOut to partIn
      partIn = partOut;

      // Checking if done
      int unallocatedNodes = thrust::count(aggregated.begin(), aggregated.end(), 0);
      if (verbose)
         printf("unallocatedNodes = %d\n", unallocatedNodes);
      if(unallocatedNodes == 0)
      {
         // Removing small partitions:
         if(!firstTime || misCount < 10)
         {
            // Making sure there are no oversized partitions
            restrictPartitionSize(maxSize, fullSize, adjIndexes, adjacency, partIn, nodeWeights);
            complete = true;
         }
         else
         {
            firstTime = false;
            removeRuntyPartitions(fullSize, partIn, nodeWeights, verbose);
            thrust::transform(partIn.begin(), partIn.end(), aggregated.begin(), findAggregated());
            partOut = partIn;
         }
      }
   }
   cudaThreadSynchronize();
}

void misHelpers::restrictPartitionSize(int maxSize, int fullSize, IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &partition, IdxVector_d &nodeWeights, bool verbose)
{
   int size = partition.size();
   IntVector_d partSizes, swap_to(size), swap_from(size), swap_index(size);
   FloatVector_d desirability(size);

   // Finding the weighted sizes of each partition
   getWeightedPartSizes(partition, nodeWeights, partSizes);

   // Finding the average size:
   int averageSize = fullSize / partSizes.size();

   // Finding largest part size:
   int largestPart = thrust::reduce(partSizes.begin(), partSizes.end(), (int)0, thrust::maximum<int>());
   while(largestPart > maxSize)
   {
      if (verbose)
         printf("largestPart = %d\n", largestPart);
      // Calculating the desirability of the nodes:
      int *adjIndexes_d = thrust::raw_pointer_cast(&adjIndexes[0]);
      int *adjacency_d = thrust::raw_pointer_cast(&adjacency[0]);
      int *partition_d = thrust::raw_pointer_cast(&partition[0]);
      int *partSizes_d = thrust::raw_pointer_cast(&partSizes[0]);
      int *swap_to_d = thrust::raw_pointer_cast(&swap_to[0]);
      int *swap_from_d = thrust::raw_pointer_cast(&swap_from[0]);
      int *nodeWeights_d = thrust::raw_pointer_cast(&nodeWeights[0]);
      int *swap_index_d = thrust::raw_pointer_cast(&swap_index[0]);
      float *desirability_d = thrust::raw_pointer_cast(&desirability[0]);
      int blockSize = 256;
      int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

      findDesirabilityKernel << < nBlocks, blockSize >> >(size, averageSize, adjIndexes_d, adjacency_d, partition_d, partSizes_d, nodeWeights_d, swap_to_d, swap_from_d, swap_index_d, desirability_d);

      // Sort the results with (swap_from, desirability) as the key
      thrust::sort_by_key(thrust::make_zip_iterator(
               thrust::make_tuple(swap_from.begin(), desirability.begin())),
            thrust::make_zip_iterator(
               thrust::make_tuple(swap_from.end(), desirability.end())),
            thrust::make_zip_iterator(
               thrust::make_tuple(swap_to.begin(), swap_index.begin())));

      // Perform good swaps
      makeSwapsKernel << < nBlocks, blockSize >> >(size, partition_d, partSizes_d, nodeWeights_d, swap_to_d, swap_from_d, swap_index_d, desirability_d);

      // Repeat until no overlarge aggregates are found
      largestPart = thrust::reduce(partSizes.begin(), partSizes.end(), (int)0, thrust::maximum<int>());

   }
}

void misHelpers::getSizes(IdxVector_d &adjIndexes, IdxVector_d &sizes)
{
   int size = adjIndexes.size() - 1;
   sizes.resize(size, 0);
   int *adjIndexes_d = thrust::raw_pointer_cast(&adjIndexes[0]);
   int *sizes_d = thrust::raw_pointer_cast(&sizes[0]);

   // Figuring out block sizes for kernel call:
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Calling kernel to find sizes:
   findAdjacencySizesKernel << < nBlocks, blockSize >> > (size, adjIndexes_d, sizes_d);
}

bool misHelpers::removeRuntyParts(int minSize, IdxVector_d &partition)
{
   // Getting the part sizes:
   IdxVector_d partSizes;
   getPartSizes(partition, partSizes);

   // Converting part sizes to a removeStencil
   thrust::device_vector<int> removeStencil(partSizes.size());
   thrust::transform(partSizes.begin(), partSizes.end(), removeStencil.begin(), labelLessThan(minSize));

   // Checking if anything will be removed:
   int removed = thrust::count(removeStencil.begin(), removeStencil.end(), 1);

   //  DataRecorder::Add("Runty parts Removed", removed);

   // If nothing to remove, just return.
   if(removed == 0)
      return true;

   // Getting a vector with how much to subtract from non-removed aggregates
   thrust::device_vector<int> subtractions(partSizes.size());
   thrust::inclusive_scan(removeStencil.begin(), removeStencil.end(), subtractions.begin());

   // Figuring out block sizes for kernel call:
   int size = partition.size();
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Getting pointers for the call:
   int *partition_d = thrust::raw_pointer_cast(&partition[0]);
   int *removeStencil_d = thrust::raw_pointer_cast(&removeStencil[0]);
   int *subtractions_d = thrust::raw_pointer_cast(&subtractions[0]);

   // Calling kernel to find sizes:
   removeRuntyPartsKernel <<< nBlocks, blockSize >>> (size, partition_d, removeStencil_d, subtractions_d);

   return false;
}

bool misHelpers::removeRuntyPartitions(int fullSize, IdxVector_d &partition, IdxVector_d &nodeWeights, bool verbose)
{
   // Getting the part sizes:
   IntVector_d partSizes;
   getWeightedPartSizes(partition, nodeWeights, partSizes);

   // Figuring out the appropriate removal size
   double averageSize = (double)fullSize / partSizes.size();
   if (verbose)
      printf("Partition average size is %f\n", averageSize);
   int threshold = (int)(averageSize * .7);

   // Converting part sizes to a removeStencil
   thrust::device_vector<int> removeStencil(partSizes.size());
   thrust::transform(partSizes.begin(), partSizes.end(), removeStencil.begin(), labelLessThan(threshold));

   // Checking if anything will be removed:
   int removed = thrust::count(removeStencil.begin(), removeStencil.end(), 1);

   // Getting a vector with how much to subtract from non-removed aggregates
   thrust::device_vector<int> subtractions(partSizes.size());
   thrust::inclusive_scan(removeStencil.begin(), removeStencil.end(), subtractions.begin());

   // Figuring out block sizes for kernel call:
   int size = partition.size();
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Getting pointers for the call:
   int *partition_d = thrust::raw_pointer_cast(&partition[0]);
   int *removeStencil_d = thrust::raw_pointer_cast(&removeStencil[0]);
   int *subtractions_d = thrust::raw_pointer_cast(&subtractions[0]);

   // Calling kernel to find sizes:
   removeRuntyPartsKernel << < nBlocks, blockSize >> > (size, partition_d, removeStencil_d, subtractions_d);

   return false;
}

void misHelpers::getPartSizes(IdxVector_d &partition, IdxVector_d &partSizes)
{
   // Make a copy of the partition vector to mess with:
   IdxVector_d temp = partition;

   // Sort the copy and find largest element
   thrust::sort(temp.begin(), temp.end());
   int maxPart = temp[temp.size() - 1];

   // Creating a new array size
   IdxVector_d partIndices(maxPart + 2, 0);

   // Figuring out block sizes for kernel call:
   int size = partition.size();
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Getting pointers
   int *temp_d = thrust::raw_pointer_cast(&temp[0]);
   int *partIndices_d = thrust::raw_pointer_cast(&partIndices[0]);

   // Calling kernel to find indices for each part:
   findPartIndicesKernel << < nBlocks, blockSize >> > (size, temp_d, partIndices_d);

   // Getting the sizes:
   getSizes(partIndices, partSizes);
}

void misHelpers::getPartSizes(IdxVector_d &partition, IdxVector_d &partSizes, IdxVector_d &partIndices)
{
   // Make a copy of the partition vector to mess with:
   IdxVector_d temp = partition;

   // Sort the copy and find largest element
   thrust::sort(temp.begin(), temp.end());
   int maxPart = temp[temp.size() - 1];

   // Creating a new array size
   partIndices.resize(maxPart + 2, 0);

   // Figuring out block sizes for kernel call:
   int size = partition.size();
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Getting pointers
   int *temp_d = thrust::raw_pointer_cast(&temp[0]);
   int *partIndices_d = thrust::raw_pointer_cast(&partIndices[0]);

   // Calling kernel to find indices for each part:
   findPartIndicesKernel << < nBlocks, blockSize >> > (size, temp_d, partIndices_d);

   // Getting the sizes:
   getSizes(partIndices, partSizes);
}

void misHelpers::getPartIndices(IdxVector_d& sortedPartition, IdxVector_d& partIndices)
{
   // Sizing the array:
   int maxPart = sortedPartition[sortedPartition.size() - 1];
   partIndices.resize(maxPart + 2);
   thrust::fill(partIndices.begin(), partIndices.end(), 0);

   // Figuring out block sizes for kernel call:
   int size = sortedPartition.size();
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Getting pointers
   int *sortedPartition_d = thrust::raw_pointer_cast(&sortedPartition[0]);
   int *partIndices_d = thrust::raw_pointer_cast(&partIndices[0]);

   // Calling kernel to find indices for each part:
   findPartIndicesKernel << < nBlocks, blockSize >> > (size, sortedPartition_d, partIndices_d);
   partIndices[partIndices.size() - 1] = size;
}

void misHelpers::getPartIndicesNegStart(IdxVector_d& sortedPartition, IdxVector_d& partIndices)
{
   // Sizing the array:
   int maxPart = sortedPartition[sortedPartition.size() - 1];
   partIndices.resize(maxPart + 2, 0);

   // Figuring out block sizes for kernel call:
   int size = sortedPartition.size();
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Getting pointers
   int *sortedPartition_d = thrust::raw_pointer_cast(&sortedPartition[0]);
   int *partIndices_d = thrust::raw_pointer_cast(&partIndices[0]);

   // Calling kernel to find indices for each part:
   findPartIndicesNegStartKernel << < nBlocks, blockSize >> > (size, sortedPartition_d, partIndices_d);
   partIndices[partIndices.size() - 1] = size - 1;
}

void misHelpers::fillWithIndex(IdxVector_d &tofill)
{
   // Figuring out block sizes for kernel call:
   int size = tofill.size();
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   int *tofill_d = thrust::raw_pointer_cast(&tofill[0]);

   fillWithIndexKernel << < nBlocks, blockSize >> > (size, tofill_d);
}

void misHelpers::getInversePermutation(IdxVector_d &original, IdxVector_d &inverse)
{
   int size = original.size();
   inverse.resize(size, -1);

   // Get pointers:
   int *original_d = thrust::raw_pointer_cast(&original[0]);
   int *inverse_d = thrust::raw_pointer_cast(&inverse[0]);

   // Figuring out block sizes for kernel call:
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Calling kernel:
   getInversePermutationKernel << < nBlocks, blockSize >> > (size, original_d, inverse_d);
}

void misHelpers::permuteInitialAdjacency(IdxVector_d &adjIndexesIn, IdxVector_d &adjacencyIn, IdxVector_d &permutedAdjIndexesIn, IdxVector_d &permutedAdjacencyIn, IdxVector_d &ipermutation, IdxVector_d &fineAggregate)
{
   int size = adjIndexesIn.size() - 1;

   // Get pointers:adjacencyIn
   int *adjIndexesIn_d = thrust::raw_pointer_cast(&adjIndexesIn[0]);
   int *adjacencyIn_d = thrust::raw_pointer_cast(&adjacencyIn[0]);
   int *permutedAdjIndexesIn_d = thrust::raw_pointer_cast(&permutedAdjIndexesIn[0]);
   int *permutedAdjacencyIn_d = thrust::raw_pointer_cast(&permutedAdjacencyIn[0]);
   int *ipermutation_d = thrust::raw_pointer_cast(&ipermutation[0]);
   int *fineAggregate_d = thrust::raw_pointer_cast(&fineAggregate[0]);

   // Figuring out block sizes for kernel call:
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Calling kernel:
   permuteInitialAdjacencyKernel << < nBlocks, blockSize >> > (size, adjIndexesIn_d, adjacencyIn_d, permutedAdjIndexesIn_d, permutedAdjacencyIn_d, ipermutation_d, fineAggregate_d);
}

void misHelpers::getInducedGraphNeighborCounts(IdxVector_d &aggregateIdx,
      IdxVector_d &adjIndexesOut,
      IdxVector_d &permutedAdjIndexesIn,
      IdxVector_d &permutedAdjacencyIn) {
   int size = aggregateIdx.size() - 1;

   // Get pointers:adjacencyIn
   int *aggregateIdx_d = thrust::raw_pointer_cast(&aggregateIdx[0]);
   int *adjIndexesOut_d = thrust::raw_pointer_cast(&adjIndexesOut[0]);
   int *permutedAdjIndexesIn_d = thrust::raw_pointer_cast(&permutedAdjIndexesIn[0]);
   int *permutedAdjacencyIn_d = thrust::raw_pointer_cast(&permutedAdjacencyIn[0]);

   // Figuring out block sizes for kernel call:
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Calling kernel:
   getInducedGraphNeighborCountsKernel << < nBlocks, blockSize >> > (size, aggregateIdx_d, adjIndexesOut_d, permutedAdjIndexesIn_d, permutedAdjacencyIn_d);

}

void misHelpers::fillCondensedAdjacency(IdxVector_d& aggregateIdx, IdxVector_d& adjIndexesOut, IdxVector_d& adjacencyOut, IdxVector_d& permutedAdjIndexesIn, IdxVector_d& permutedAdjacencyIn)
{
   int size = adjIndexesOut.size() - 1;

   // Get pointers:adjacencyIn
   int *aggregateIdx_d = thrust::raw_pointer_cast(&aggregateIdx[0]);
   int *adjIndexesOut_d = thrust::raw_pointer_cast(&adjIndexesOut[0]);
   int *adjacencyOut_d = thrust::raw_pointer_cast(&adjacencyOut[0]);
   int *permutedAdjIndexesIn_d = thrust::raw_pointer_cast(&permutedAdjIndexesIn[0]);
   int *permutedAdjacencyIn_d = thrust::raw_pointer_cast(&permutedAdjacencyIn[0]);

   // Figuring out block sizes for kernel call:
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Calling kernel:
   fillCondensedAdjacencyKernel << < nBlocks, blockSize >> > (size, aggregateIdx_d, adjIndexesOut_d, adjacencyOut_d, permutedAdjIndexesIn_d, permutedAdjacencyIn_d);

}

void misHelpers::fillPartitionLabel(IdxVector_d& coarseAggregate, IdxVector_d& fineAggregateSort, IdxVector_d& partitionLabel)
{
   int size = partitionLabel.size();

   // Get pointers:adjacencyIn
   int *coarseAggregate_d = thrust::raw_pointer_cast(&coarseAggregate[0]);
   int *fineAggregateSort_d = thrust::raw_pointer_cast(&fineAggregateSort[0]);
   int *partitionLabel_d = thrust::raw_pointer_cast(&partitionLabel[0]);

   // Figuring out block sizes for kernel call:
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Calling kernel:
   fillPartitionLabelKernel << < nBlocks, blockSize >> > (size, coarseAggregate_d, fineAggregateSort_d, partitionLabel_d);

}

void misHelpers::getAggregateStartIndices(IdxVector_d& fineAggregateSort, IdxVector_d& aggregateRemapIndex)
{
   int size = fineAggregateSort.size();

   // Get pointers:adjacencyIn
   int *fineAggregateSort_d = thrust::raw_pointer_cast(&fineAggregateSort[0]);
   int *aggregateRemapIndex_d = thrust::raw_pointer_cast(&aggregateRemapIndex[0]);

   // Figuring out block sizes for kernel call:
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Calling kernel:
   getAggregateStartIndicesKernel << < nBlocks, blockSize >> > (size, fineAggregateSort_d, aggregateRemapIndex_d);
}

void misHelpers::remapAggregateIdx(IdxVector_d& fineAggregateSort, IdxVector_d& aggregateRemapId)
{
   int size = fineAggregateSort.size();

   // Get pointers:adjacencyIn
   int *fineAggregateSort_d = thrust::raw_pointer_cast(&fineAggregateSort[0]);
   int *aggregateRemapId_d = thrust::raw_pointer_cast(&aggregateRemapId[0]);

   // Figuring out block sizes for kernel call:
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Calling kernel:
   remapAggregateIdxKernel << < nBlocks, blockSize >> > (size, fineAggregateSort_d, aggregateRemapId_d);
}

void misHelpers::mapAdjacencyToBlock(IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &adjacencyBlockLabel, IdxVector_d &blockMappedAdjacency, IdxVector_d &fineAggregate)
{
   int size = adjIndexes.size() - 1;

   // Get pointers:adjacencyIn
   int *adjIndexes_d = thrust::raw_pointer_cast(&adjIndexes[0]);
   int *adjacency_d = thrust::raw_pointer_cast(&adjacency[0]);
   int *adjacencyBlockLabel_d = thrust::raw_pointer_cast(&adjacencyBlockLabel[0]);
   int *blockMappedAdjacency_d = thrust::raw_pointer_cast(&blockMappedAdjacency[0]);
   int *fineAggregate_d = thrust::raw_pointer_cast(&fineAggregate[0]);

   // Figuring out block sizes for kernel call:
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   // Calling kernel:
   mapAdjacencyToBlockKernel << < nBlocks, blockSize >> > (size, adjIndexes_d, adjacency_d, adjacencyBlockLabel_d, blockMappedAdjacency_d, fineAggregate_d);
}

void misHelpers::getInducedGraph(IdxVector_d &adjIndexesIn, IdxVector_d &adjacencyIn, IdxVector_d &partitionLabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut)
{
   // Declaring temporary vectors:
   IdxVector_d adjacencyBlockLabel, blockMappedAdjacency;
   adjacencyBlockLabel.resize(adjacencyIn.size(), 0);
   blockMappedAdjacency.resize(adjacencyIn.size(), 0);

   // Get the blocklabeled adjacency:
   misHelpers::mapAdjacencyToBlock(adjIndexesIn, adjacencyIn, adjacencyBlockLabel, blockMappedAdjacency, partitionLabel);

   // Zip up the block label and block mapped vectors and sort:
   thrust::sort(thrust::make_zip_iterator(
            thrust::make_tuple(adjacencyBlockLabel.begin(), blockMappedAdjacency.begin())),
         thrust::make_zip_iterator(
            thrust::make_tuple(adjacencyBlockLabel.end(), blockMappedAdjacency.end())));

   // Remove Duplicates and resize:
   int newSize = thrust::unique(
         thrust::make_zip_iterator(
            thrust::make_tuple(adjacencyBlockLabel.begin(), blockMappedAdjacency.begin())),
         thrust::make_zip_iterator(
            thrust::make_tuple(adjacencyBlockLabel.end(), blockMappedAdjacency.end()))) -
      thrust::make_zip_iterator(thrust::make_tuple(adjacencyBlockLabel.begin(), blockMappedAdjacency.begin()));

   adjacencyBlockLabel.resize(newSize);
   blockMappedAdjacency.resize(newSize);
   misHelpers::getPartIndicesNegStart(adjacencyBlockLabel, adjIndexesOut);
   adjacencyOut.resize(blockMappedAdjacency.size() - 1);
   thrust::copy(blockMappedAdjacency.begin() + 1, blockMappedAdjacency.end(), adjacencyOut.begin());
}

void misHelpers::getWeightedPartSizes(IdxVector_d &partition, IdxVector_d &nodeWeights, IntVector_d &partSizes)
{
   // Make copies to mess with
   IntVector_d part(partition.begin(), partition.end());
   IntVector_d weights(nodeWeights.begin(), nodeWeights.end());

   // Sorting temp vectors together
   thrust::sort_by_key(part.begin(), part.end(), weights.begin());

   // Getting prefix sum of values
   thrust::inclusive_scan(weights.begin(), weights.end(), weights.begin());

   // Another temp vector for accumulated size at last nodes
   IntVector_d accumulatedSize(part[part.size() - 1] + 1);

   // Preparing to call kernel to fill accumulated size vector
   int size = part.size();
   int *part_d = thrust::raw_pointer_cast(&part[0]);
   int *weights_d = thrust::raw_pointer_cast(&weights[0]);
   int *accumulatedSize_d = thrust::raw_pointer_cast(&accumulatedSize[0]);

   // Figuring out block sizes for kernel call:
   int blockSize = 256;
   int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

   accumulatedPartSizesKernel << < nBlocks, blockSize >> > (size, part_d, weights_d, accumulatedSize_d);

   // Calling kernel to get the unaccumulated part sizes:
   size = accumulatedSize.size();
   nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);
   partSizes.resize(size);
   int *sizes_d = thrust::raw_pointer_cast(&partSizes[0]);
   unaccumulatedPartSizesKernel << < nBlocks, blockSize >> > (size, accumulatedSize_d, sizes_d);
}

void misHelpers::checkPartConnectivity(int partCount, IdxVector_d partition, IdxVector_d adjIndexes, IdxVector_d adjacency, char *message)
{
   // Debugging check on part connectivity:
   std::cout << message << "\n";
   vector<int> nodesToExplore, exploredNodes;
   for(int i = 0; i < partCount; i++)
   {
      nodesToExplore.clear();
      exploredNodes.clear();

      // Find a node in the part
      int rootId = -1;
      for(int j = 0; j < partition.size(); j++)
      {
         if(partition[j] == i)
         {
            rootId = j;
            break;
         }
      }

      // Explore out from the part
      int start = adjIndexes[rootId], end = adjIndexes[rootId + 1];
      for(int n = start; n < end; n++)
      {
         int neighbor = adjacency[n];
         if(partition[neighbor] == i)
            nodesToExplore.push_back(neighbor);
      }
      exploredNodes.push_back(rootId);

      // Iterating through everything:
      while(nodesToExplore.size() > 0)
      {
         // Popping off the last node to explore and checking if it's done
         int node = nodesToExplore.back();
         nodesToExplore.pop_back();

         // Checking if the node has been explored:
         bool exploredAlready = false;
         for(int q = 0; q < exploredNodes.size(); q++)
            if(exploredNodes[q] == node)
               exploredAlready = true;

         if(!exploredAlready)
         {
            int start = adjIndexes[node], end = adjIndexes[node + 1];
            for(int n = start; n < end; n++)
            {
               int neighbor = adjacency[n];
               if(partition[neighbor] == i)
               {
                  nodesToExplore.push_back(neighbor);
                  //printf("\tAdded %d a neighbor of %d to explore list for part %d", neighbor, node, i);
               }
            }
            exploredNodes.push_back(node);
            //printf("\tAdded %d to explored for part %d\n", node, i);
         }

      }

      // Now checking to see if there were any unreachable nodes.
      for(int j = 0; j < partition.size(); j++)
      {
         if(partition[j] == i)
         {
            bool found = false;
            for(int q = 0; q < exploredNodes.size(); q++)
               if(exploredNodes[q] == j)
               {
                  found = true;
                  break;
               }

            if(!found)
            {
               printf("Could not reach node %d in part %d from root %d\n", j, i, rootId);
               printf("\tExplored nodes:");
               for(int g = 0; g < exploredNodes.size(); g++)
                  printf(" %3d", exploredNodes[g]);
               printf("\n");
            }
         }
      }
   }

   // Pausing
   int dummy = 0;
   std::cin >> dummy;

   if(dummy == 1)
   {
      int partToCheck;
      std::cin >> partToCheck;

      for(int i = 0; i < partition.size(); i++)
      {
         if(partition[i] == partToCheck)
         {
            int start = adjIndexes[i], end = adjIndexes[i + 1];
            printf("Node %d is in partition %d\n\t", i, partToCheck);
            for(int j = start; j < end; j++)
            {
               int neighbor = adjacency[j];
               printf(" %4d ", neighbor);
            }
            printf("\n");
         }
      }

   }
}

void misHelpers::remapInducedGraph(IdxVector_d &adjIndexes, IdxVector_d &adjacency, IdxVector_d &partition)
{
   IdxVector_d tempCoarseAggregate = partition;
   IdxVector_d aggregateLabel = adjacency;
   IdxVector_d permutedAdjacency = adjacency;
   IdxVector_d coarsePermutation = partition;
   IdxVector_d coarseIPermutation;

   // Get the inverse permutation for the re-mapping
   misHelpers::fillWithIndex(coarsePermutation);
   thrust::stable_sort_by_key(tempCoarseAggregate.begin(), tempCoarseAggregate.end(), coarsePermutation.begin());
   misHelpers::getInversePermutation(coarsePermutation, coarseIPermutation);

   // Map the adjacency according to the inverse permutation
   misHelpers::mapAdjacencyToBlock(adjIndexes, adjacency, aggregateLabel, permutedAdjacency, coarseIPermutation);
   thrust::sort_by_key(aggregateLabel.begin(), aggregateLabel.end(), permutedAdjacency.begin());

   // Copy from the temp to the real adjacency
   thrust::copy(permutedAdjacency.begin(), permutedAdjacency.end(), adjacency.begin());

   // Find the adjIndexes for the new adjacency
   misHelpers::getPartIndices(aggregateLabel, adjIndexes);
}
