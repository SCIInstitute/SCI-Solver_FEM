/* 
 * File:   AggMIS_MergeSplitConditioner.cu
 * Author: T. James Lewis
 *
 * Created on April 15, 2013, 2:13 PM
 */
#include "AggMIS_MergeSplitConditioner.h"

namespace AggMIS {
    namespace MergeSplitGPU {
        namespace Kernels {
            namespace D {
                __device__ void LoadLocalNeighbors(int *neighbors,
                                                    int *nextNeighbor,
                                                    int aggSize,
                                                    int *nodeIds,
                                                    int *adjIndices,
                                                    int *adjacency) {
                    if (threadIdx.x < aggSize)
                    {
                        int nodeId = nodeIds[threadIdx.x];
                        int start = adjIndices[nodeId];
                        int end = adjIndices[nodeId + 1];
                        for (int i = start; i < end; i++)
                        {
                            int neighborId = adjacency[i];
                            int neighborPlace = D::BinarySearch(neighborId, 
                                                                0, 
                                                                aggSize,
                                                                nodeIds);
                            if (neighborPlace != -1) {
                                neighbors[(*nextNeighbor)++] = neighborPlace;
                            }
                        }
                    }
                    __syncthreads();
                }
                __device__ int BinarySearch(int value,
                                                    int imin,
                                                    int imax,
                                                    int *array) {
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
                __device__ void FloodFillDistanceFrom(int starter, 
                                                    int* array, 
                                                    int nodeCount,
                                                    int *neighbors,
                                                    int neighborCount,
                                                    int *farthestId,
                                                    bool *incomplete) {
                    // Start by marking the starter node with distance zero
                    int myDist = threadIdx.x == starter ? 0 : -1;
                    if (threadIdx.x < nodeCount) {
                        array[threadIdx.x] = myDist;
                    }
                    
                    // Then set the incomplete flag to true
                    *incomplete = true;
                    __syncthreads();

                    while (*incomplete)
                    {
                        // Set the incomplete flag to false
                        *incomplete = false;

                        // Check if a neighbor has a positive distance
                        if (myDist == -1 && threadIdx.x < nodeCount)
                        {
                            for (int i = 0; i < neighborCount; i++)
                            {
                                int neighborDist = array[neighbors[i]];
                                if (neighborDist > -1) {
                                    myDist = neighborDist + 1;
                                    *farthestId = threadIdx.x;
                                }
                            }
                        }
                        __syncthreads();
                        
                        // Writing current value to shared array 
                        array[threadIdx.x] = myDist;
                        if (myDist == -1 && threadIdx.x < nodeCount) {
                            *incomplete = true;
                        }  
                        __syncthreads();
                    }
                }
                __device__ void PrintSharedArray(int size, 
                                                    int *array, 
                                                    const char *note) {
                    if (threadIdx.x == 0 && blockIdx.x == 0) {
                        printf("%s\n   [%3d:%3d]  ", note, 0, 10);
                        for (int i = 0; i < size; i++) {
                            printf("%6d ", array[i]);
                            if ((i + 1) % 10 == 0 && i != size - 1)
                                printf("\n   [%3d:%3d]  ", (i + 1), (i + 11));
                        }
                        printf("\nDone Printing Array.\n");
                    }
                    __syncthreads();
                }
                __device__ void WarpReport(const char* note) {
                    if (threadIdx.x % 32 == 0)
                        printf("Warp %d %s\n", threadIdx.x / 32, note);
                }
                __device__ void SillyTest() {
                    if (threadIdx.x == 0)
                        printf("Silly test worked!");
                }
            }
            __global__ void MakeMerges (int size, 
                                int *mergeWith, 
                                int *offsets, 
                                int *mis) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size)
                {
                    int currentAgg = mis[idx];
                    int newAgg = mergeWith[currentAgg];
                    // If the aggregate is not merging just apply offset
                    if (newAgg == -1)
                    {
                        mis[idx] = currentAgg - offsets[currentAgg];
                    }
                    // The aggregate is merging find offset of aggregate merging with
                    else
                    {
                        mis[idx] = newAgg - offsets[newAgg];
                    }
                }
            }
            __global__ void MakeMerges_MarkSplits(int size, 
                                int* mergeWith, 
                                int* offsets, 
                                int* mis, 
                                int* splitsToMake) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size)
                {
                    int currentAgg = mis[idx];
                    int newAgg = mergeWith[currentAgg];
                    // If the aggregate is not merging just apply offset
                    if (newAgg == -1)
                    {
                        mis[idx] = currentAgg - offsets[currentAgg];
                    }
                    // The aggregate is merging find offset of aggregate merging with
                    else
                    {
                        int newId = newAgg - offsets[newAgg];
                        mis[idx] = newId;
                        splitsToMake[newId] = 1;
                    }
                }    
            }
            __global__ void MakeSplits(int baseAggregateIdx, 
                                int* splitting, 
                                int* aggregation, 
                                int* aggMapAdjIndices, 
                                int* aggMapAdjacency, 
                                int* adjIndices, 
                                int* adjacency) {
                int currentAgg = splitting[blockIdx.x];
                int aggBegin = aggMapAdjIndices[currentAgg];
                int aggSize = aggMapAdjIndices[currentAgg + 1] - aggBegin;
                int newAgg = baseAggregateIdx + blockIdx.x;

                __shared__ int nodeIds[64];
                __shared__ int scratchA[64];
                __shared__ int scratchB[64];
                __shared__ int rootA;
                __shared__ int rootB;
                __shared__ int aCount, bCount;
                __shared__ bool incomplete;
                incomplete = true;
                aCount = 1;
                bCount = 1;

                // Load in the node Id's from the aggregate map to the shared array:
                if (threadIdx.x < aggSize)
                    nodeIds[threadIdx.x] = aggMapAdjacency[aggBegin + threadIdx.x];
                __syncthreads();

                // Each thread loads it's neighbors list into registers, translating into
                // aggregate offsets.
                int neighbors[40];
                int nextNeighbor = 0;
                int nodeId = -1;
                if (threadIdx.x < aggSize)
                {
                    nodeId = nodeIds[threadIdx.x];
                    int start = adjIndices[nodeId];
                    int end = adjIndices[nodeId + 1];
                    for (int i = start; i < end; i++)
                    {
                        int neighborId = adjacency[i];
                        int a = 0, b = aggSize - 1, midpoint;
                        while (a < b)
                        {
                            midpoint = a + ((b - a) / 2);
                            if (nodeIds[midpoint] < neighborId)
                                a = midpoint + 1;
                            else
                                b = midpoint;
                        }

                        if (nodeIds[a] == neighborId)
                        {
                            neighbors[nextNeighbor++] = a;
                        }
                    }
                }
                __syncthreads();
                
                // Find the farthest node from the lowest indexed node (first root point)
                // Start by marking the first node and threads without a node as seen  
                // Mark initial distances in scratch vector
                if (threadIdx.x < aggSize)
                    scratchA[threadIdx.x] = threadIdx.x == 0 ? 0 : -1;
                int myDist = threadIdx.x == 0 ? 0 : -1;
                bool swapped = false;
                incomplete = true;
                __syncthreads(); 

                while (incomplete)
                {   
                    // Set the incomplete flag to false
                    incomplete = false;
                    __syncthreads();

                    // Check if a neighbor has a positive distance
                    if (threadIdx.x < aggSize && myDist == -1)
                    {
                        for (int i = 0; i < nextNeighbor; i++)
                        {
                            int neighborDist = scratchA[neighbors[i]];
                            if (neighborDist > -1)
                                myDist = neighborDist + 1;
                        }
                    }
                    __syncthreads();

                    if (threadIdx.x < aggSize && myDist > 0 && !swapped)
                    {
                        swapped = true;
                        scratchA[threadIdx.x] = myDist;
                        rootA = threadIdx.x;
                        incomplete = true;
                    }  
                    __syncthreads();
                }

                // Find the farthest node from the first root point (second root point)
                // Start by marking the first node and threads without a node as seen  
                // Mark initial distances in scratch vector
                if (threadIdx.x < aggSize)
                    scratchA[threadIdx.x] = threadIdx.x == rootA ? 0 : -1;
                myDist = threadIdx.x == rootA ? 0 : -1;
                swapped = false;
                incomplete = true;
                __syncthreads(); 

                while (incomplete)
                {   
                    // Set the incomplete flag to false
                    incomplete = false;
                    __syncthreads();

                    // Check if a neighbor has a positive distance
                    if (threadIdx.x < aggSize && myDist == -1)
                    {
                        for (int i = 0; i < nextNeighbor; i++)
                        {
                            int neighborDist = scratchA[neighbors[i]];
                            if (neighborDist > -1)
                            {
                                myDist = neighborDist + 1;
                            }
                        }
                    }
                    __syncthreads();

                    if (threadIdx.x < aggSize && myDist > 0 && !swapped)
                    {
                        swapped = true;
                        scratchA[threadIdx.x] = myDist;
                        rootB = threadIdx.x;
                        incomplete = true;
                    }  
                    __syncthreads();
                }

                // Setting an assigned aggregate label (In ScratchA) for every node with the node at 
                // rootA being assigned the current aggregate ID and the node at rootB
                // being assigned the newAgg ID and set initial distances from a root node
                // (In ScratchB) for each node, -1 for unknown and 0 for the roots
                int myAggregate = -1;
                if (threadIdx.x == rootA)
                    myAggregate = currentAgg;
                if (threadIdx.x == rootB)
                    myAggregate = newAgg;
                if (threadIdx.x < aggSize)
                {      
                    scratchA[threadIdx.x] = myAggregate;
                    scratchB[threadIdx.x] = myAggregate > -1 ? 0 : -1;
                }
                incomplete = true;
                __syncthreads();

                // Assign nodes to each aggregate until no unassigned nodes remain.
                while (incomplete)
                {   
                    // Set the incomplete flag to false
                    incomplete = false;
                    __syncthreads();

                    if (threadIdx.x < aggSize && myAggregate == -1)
                    {
                        for (int i = 0; i < nextNeighbor; i++)
                        {
                            int neighborAgg = scratchA[neighbors[i]];
                            if (neighborAgg > -1)
                            {
                                myDist = scratchB[neighbors[i]] + 1;
                                myAggregate = neighborAgg;
                            }
                        }
                        if (myAggregate == -1)
                            incomplete = true;
                        if (myAggregate == newAgg)
                            atomicAdd((unsigned int*)&bCount, (unsigned)1);
                        if (myAggregate == currentAgg)
                            atomicAdd((unsigned int*)&aCount, (unsigned)1);
                    }
                    __syncthreads();

                    if (threadIdx.x < aggSize)
                    {
                        scratchA[threadIdx.x] = myAggregate;
                        scratchB[threadIdx.x] = myDist;
                    }
                    __syncthreads();
                }

                // If the split was uneven try to repair it
                int sizeDifference = aCount > bCount ? aCount - bCount : bCount - aCount;
                bool moveToA = aCount < bCount;
                __shared__ int moved;
                moved = 0;
                int toMove = sizeDifference / 2;
                incomplete = true;
                __syncthreads();

                while (incomplete && moved < toMove)
                {
                    incomplete = false;
                    __syncthreads();
                    bool swapping = false;
                    int newDist = INT_MAX;
                    if (threadIdx.x < aggSize)
                    {
                        bool canSwap = moveToA ? myAggregate == newAgg : myAggregate == currentAgg;
                        bool borderNode = false;

                        // Check if this node has no same aggregate neighbors of higher distance
                        // and on a border
                        for (int i = 0; i < nextNeighbor; i++)
                        {
                            int neighborAgg = scratchA[neighbors[i]];
                            int neighborDist = scratchB[neighbors[i]];
                            if (neighborAgg == myAggregate && neighborDist > myDist)
                                canSwap = false;

                            if (neighborAgg != myAggregate)
                            {
                                if (neighborDist + 1 < newDist)
                                    newDist = neighborDist + 1;
                                borderNode = true;
                            }
                        }

                        // If a node could swap see if it will
                        if (borderNode && canSwap && atomicAdd((unsigned int*)&moved, 1) < toMove)
                        {
                            swapping = true;
                        }
                    }
                    __syncthreads();
                    if (swapping)
                    {
                        int a = moveToA ? 1 : -1;
                        atomicAdd((unsigned int*)&bCount, -a);
                        atomicAdd((unsigned int*)&aCount, a);
                        scratchA[threadIdx.x] = moveToA ? currentAgg : newAgg;
                        scratchB[threadIdx.x] = newDist;
                        incomplete = true;
                    }
                    __syncthreads();
                }

                // Write out the values to the aggregation array
                if (threadIdx.x < aggSize)
                {
                    aggregation[nodeId] = scratchA[threadIdx.x];
                }

            }
            __global__ void MakeSplitsWeighted(int baseAggregateIdx, 
                                int* splitting, 
                                int* aggregation, 
                                int* aggMapAdjIndices, 
                                int* aggMapAdjacency, 
                                int* adjIndices, 
                                int* adjacency,
                                int* weights) {
                int currentAgg = splitting[blockIdx.x];
                int aggBegin = aggMapAdjIndices[currentAgg];
                int aggSize = aggMapAdjIndices[currentAgg + 1] - aggBegin;
                int newAgg = baseAggregateIdx + blockIdx.x;
                
                // Debug
                int iterationCount = 0;
//                if (blockIdx.x == 0 && threadIdx.x == 0) {
//                    printf("Starting MakeSplitsWeighted kernel for aggregate %d with node count %d\n", 
//                            currentAgg, aggSize);
//                }                
                
                __shared__ int nodeIds[64];
                __shared__ int nodeWeights[64];
                __shared__ int scratchA[64];
                __shared__ int scratchB[64];
                __shared__ int rootA, rootB;
                __shared__ int aCount, bCount;
                __shared__ bool incomplete;
                incomplete = true;

                // Load in the node Id's from the aggregate map to the shared array:
                if (threadIdx.x < aggSize) {
                    nodeIds[threadIdx.x] = aggMapAdjacency[aggBegin + threadIdx.x];
                    nodeWeights[threadIdx.x] = weights[nodeIds[threadIdx.x]];
                }
                __syncthreads();
                
                // Each thread loads it's neighbors list into registers, translating into
                // aggregate offsets.
                int neighbors[40];
                int nextNeighbor = 0;
                D::LoadLocalNeighbors(&neighbors[0],
                                        &nextNeighbor,
                                        aggSize,
                                        &nodeIds[0],
                                        adjIndices,
                                        adjacency);
                
                // Flood fill distances from node 0 to find first root node
                D::FloodFillDistanceFrom(0, 
                        &scratchA[0],
                        aggSize,
                        &neighbors[0],
                        nextNeighbor,
                        &rootA,
                        &incomplete);
                
                // Testing templated function call
//                D::PrintSharedArray(aggSize, &scratchA[0], "Before calling transform");
//                T::Transform(aggSize, &scratchA[0], T::AddTo<int>(3));
//                D::PrintSharedArray(aggSize, &scratchA[0], "After calling transform");
                
                // Flood fill distances from rootA to find rootB
                D::FloodFillDistanceFrom(rootA, 
                        &scratchA[0],
                        aggSize,
                        &neighbors[0],
                        nextNeighbor,
                        &rootB,
                        &incomplete);
                
                // Setting an assigned aggregate label (In ScratchA) for every node with the node at 
                // rootA being assigned the current aggregate ID and the node at rootB
                // being assigned the newAgg ID and set initial distances from a root node
                // (In ScratchB) for each node, -1 for unknown and 0 for the roots
                int myAggregate = -1;
                if (threadIdx.x == rootA) {
                    myAggregate = currentAgg;
                    aCount = weights[threadIdx.x];
                }
                if (threadIdx.x == rootB) {
                    myAggregate = newAgg;
                    bCount = weights[threadIdx.x];
                }
                if (threadIdx.x < aggSize)
                {      
                    scratchA[threadIdx.x] = myAggregate;
                    scratchB[threadIdx.x] = myAggregate > -1 ? 0 : -1;
                }
                incomplete = true;
//                D::WarpReport("before sync");
//                __syncthreads();
//                D::WarpReport("after sync");
//
//                D::PrintSharedArray(aggSize, 
//                                    &scratchA[0], 
//                                    "Pre-Initial aggregate assignments");
//                D::PrintSharedArray(aggSize, 
//                                    &scratchB[0], 
//                                    "Pre-Initial distance assignments");
                
                int myDist = threadIdx.x == 0 ? 0 : -1;
                
                // Assign nodes to each aggregate until no unassigned nodes remain.
                iterationCount = 0;
                while (incomplete && iterationCount < 10)
                {
                    iterationCount++;
//                    if (blockIdx.x == 0 && threadIdx.x == 0) {
//                        printf("\tStarting an initial allocation cycle. aCount=%d bCount=%d\n", aCount, bCount);
//                    }
                    // Set the incomplete flag to false
                    __syncthreads();
                    incomplete = false;
                    __syncthreads();

                    if (threadIdx.x < aggSize && myAggregate == -1)
                    {
                        for (int i = 0; i < nextNeighbor; i++)
                        {
                            int neighborAgg = scratchA[neighbors[i]];
                            if (neighborAgg > -1)
                            {
                                myDist = scratchB[neighbors[i]] + 1;
                                myAggregate = neighborAgg;
                            }
                        }
                        if (myAggregate == -1)
                            incomplete = true;
                        if (myAggregate == newAgg)
                            atomicAdd((unsigned int*)&bCount, (unsigned)nodeWeights[threadIdx.x]);
                        if (myAggregate == currentAgg)
                            atomicAdd((unsigned int*)&aCount, (unsigned)nodeWeights[threadIdx.x]);
                    }
                    __syncthreads();

                    if (threadIdx.x < aggSize)
                    {
                        scratchA[threadIdx.x] = myAggregate;
                        scratchB[threadIdx.x] = myDist;
                    }
                    __syncthreads();
                }
                __syncthreads();
                // Printing out the initial aggregate assignments made.
                //D::PrintSharedArray(aggSize, scratchA, "Initial Aggregate Assignments");
//                D::PrintSharedArray(aggSize, 
//                                    &scratchA[0], 
//                                    "Initial aggregate assignments");
//                D::PrintSharedArray(aggSize, 
//                                    &scratchB[0], 
//                                    "Initial distance assignments");
//                                
//                // Printing a message if max iterations exceeded.
//                if (blockIdx.x == 0 && threadIdx.x == 0 && incomplete) {
//                    printf("***********Max Iterations Exceeded*************\n");
//                }
//                __syncthreads();
                
                
                // If the split was uneven try to repair it
                __shared__ int goodSwaps[20];           // The id of nodes that have desirable swaps
                __shared__ int improvement[20];         // How much improvement the swap would make
                __shared__ int insertID;                // The index at which to insert new item
                
                
                incomplete = true;
                __syncthreads();

                iterationCount = 0;
                while (incomplete && iterationCount < 10)
                {
                    iterationCount++;
                    // Reset values
                    int sizeDifference = aCount > bCount ? aCount - bCount : bCount - aCount;
                    bool moveToA = aCount < bCount;
                    
//                    if (blockIdx.x == 0 && threadIdx.x == 0) {
//                        printf("\tStarting a size correction cycle: A=%d B=%d\n", 
//                                currentAgg, newAgg);
//                        printf("\t\taCount:=%d bCount=%d sizeDifference=%d moveToA=%s\n\n", 
//                                aCount, bCount, sizeDifference, (moveToA ? "True" : "False"));
//                    }
                    
                    insertID = 0;
                    __syncthreads();
                    
                    int newDist = INT_MAX;
                    if (threadIdx.x < aggSize)
                    {
                        bool canSwap = moveToA ? myAggregate == newAgg : myAggregate == currentAgg;
                        bool borderNode = false;

                        // Check if this node has no same aggregate neighbors of higher distance
                        // and on a border
                        for (int i = 0; i < nextNeighbor; i++)
                        {
                            int neighborAgg = scratchA[neighbors[i]];
                            int neighborDist = scratchB[neighbors[i]];
                            if (neighborAgg == myAggregate && neighborDist > myDist)
                                canSwap = false;

                            if (neighborAgg != myAggregate)
                            {
                                if (neighborDist + 1 < newDist)
                                    newDist = neighborDist + 1;
                                borderNode = true;
                            }
                        }

                        // If a node could swap see how attractive the swap would be
                        if (borderNode && canSwap)
                        {
                            int newA = moveToA ? 
                                        aCount + nodeWeights[threadIdx.x] : 
                                        aCount - nodeWeights[threadIdx.x];
                            int newB = moveToA ? 
                                        bCount - nodeWeights[threadIdx.x] : 
                                        bCount + nodeWeights[threadIdx.x];
                            int newSizeDifference = newA > newB ? 
                                                        newA - newB : 
                                                        newB - newA;
                            if (newSizeDifference < sizeDifference) {
                                int newID = atomicAdd((int *)&insertID, 1) - 1;
                                if (newID < 20) {
                                    goodSwaps[newID] = threadIdx.x;
                                    improvement[newID] = newSizeDifference;
                                } 
                            }
                        }
                    }
                    __syncthreads();
                    
                    // Now finding the best swap to make and making it
                    if (insertID > 0)
                    {
//                        if (blockIdx.x == 0 && threadIdx.x == 0) {
//                            printf("\tThere are %d possible swaps marked\n", insertID);
//                        }
                        // Have zero thread look through options for best
                        if (threadIdx.x == 0) {
                            // Checking each option found and getting the best one
                            int bestValue = INT_MAX;
                            int swapId = -1;
                            for (int i = 0; i < insertID && i < 20; i++) {
                                
                                // Debug
//                                if (blockIdx.x == 0 && threadIdx.x == 0) {
//                                    printf("\t\tNode %d can swap with improvement %d\n", 
//                                            goodSwaps[i], improvement[i]);
//                                }
                                
                                if (improvement[i] < bestValue) {
                                    bestValue = improvement[i];
                                    swapId = goodSwaps[i];
                                }
                            }
                            insertID = swapId;
                        }
//                        if (blockIdx.x == 0 && threadIdx.x == 0) {
//                            printf("\tNode %d in aggregate %d selected to swap\n", 
//                                    insertID, scratchA[insertID]);
//                        }
                        __syncthreads();
                        
                        // Have the thread belonging to the swap node do the swap
                        if (threadIdx.x == insertID) {
                            myAggregate = moveToA ? currentAgg : newAgg;
                            scratchA[threadIdx.x] = myAggregate;
                            scratchB[threadIdx.x] = newDist;
                            aCount = moveToA ? aCount + nodeWeights[threadIdx.x] : aCount - nodeWeights[threadIdx.x];
                            bCount = moveToA ? bCount - nodeWeights[threadIdx.x] : bCount + nodeWeights[threadIdx.x];
                        }
                        
                        __syncthreads();
                        
                        // Now recompute the distances to make sure things are still 
                        // connected.
                        __shared__ bool changed;
                        scratchB[threadIdx.x] = threadIdx.x == rootA || threadIdx.x == rootB ?
                                                0 : -1;
                        changed = true;
                        __syncthreads();
                        while (changed) {
                            changed = false;
                            
                            // Check if a neighbor has a positive distance
                            if (threadIdx.x < aggSize && scratchB[threadIdx.x] == -1) {
                                for (int i = 0; i < nextNeighbor; i++) {
                                    // If neighbor has a distance and is in the same 
                                    // aggregate fill distance from it.
                                    if (scratchA[neighbors[i]] == scratchA[threadIdx.x] && scratchB[neighbors[i]] > -1) {
                                        scratchB[threadIdx.x] = scratchB[neighbors[i]] + 1;
                                        changed = true;
                                    }
                                }
                            }
                            __syncthreads();
                        }
                        if (threadIdx.x < aggSize && scratchB[threadIdx.x] == -1) {
                            changed = true;
                        }
                        __syncthreads();
                        
                        
//                        if (changed && blockIdx.x == 0 && threadIdx.x == 0) {
//                            printf("\nWhile splitting aggregate %d into %d:\n",
//                                    currentAgg, newAgg);
//                            int problem = nodeIds[insertID];
//                            printf("\tId %d (node %d) broke things when swapped.\n",
//                                    insertID, problem);
//                            printf("\tDetected a disconnect. Reverting and stopping.\n");
//                        }
                        
                        if (changed && threadIdx.x == insertID) {
                            scratchA[threadIdx.x] = scratchA[threadIdx.x] == newAgg ?
                                                    currentAgg : newAgg;
                            incomplete = true;
                        }
                        
                        __syncthreads();
//                        if (changed && blockIdx.x == 0 && threadIdx.x == 0) {
//                            printf("Incomplete flag marked as: %s\n",
//                                    incomplete ? "True" : "False");
//                        }
//                        __syncthreads();
                        
                        
                        
//                        if (blockIdx.x == 0 && threadIdx.x == 0) {
//                            printf("\t\tNode %d now in aggregate %d with distance %d\n", 
//                                    insertID, scratchA[insertID], scratchB[insertID]);
//                        }
                    }
                    else {
                        incomplete = false;
                    }
                    __syncthreads();
                }
//                if (blockIdx.x == 0 && threadIdx.x == 0 && incomplete) {
//                    printf("***********Max Iterations Exceeded*************\n");
//                }
//                
//                D::PrintSharedArray(aggSize, 
//                                    &scratchA[0], 
//                                    "Final aggregate assignments");
//                D::PrintSharedArray(aggSize, 
//                                    &scratchB[0], 
//                                    "Final distance assignments");

                // Write out the values to the aggregation array
                if (threadIdx.x < aggSize)
                {
                    aggregation[nodeIds[threadIdx.x]] = scratchA[threadIdx.x];
                }
            }
            __global__ void MakeSplits_Large(int baseAggregateIdx, 
                                int* splitting, 
                                int* aggregation, 
                                int* aggMapAdjIndices, 
                                int* aggMapAdjacency, 
                                int* adjIndices, 
                                int* adjacency) {
                int currentAgg = splitting[blockIdx.x];
                int aggBegin = aggMapAdjIndices[currentAgg];
                int aggSize = aggMapAdjIndices[currentAgg + 1] - aggBegin;
                int newAgg = baseAggregateIdx + blockIdx.x;

                __shared__ int nodeIds[256];
                __shared__ int scratchA[256];
                __shared__ int scratchB[256];
                __shared__ int rootA;
                __shared__ int rootB;
                __shared__ int aCount, bCount;
                __shared__ bool incomplete;
                incomplete = true;
                aCount = 1;
                bCount = 1;

                // Load in the node Id's from the aggregate map to the shared array:
                if (threadIdx.x < aggSize)
                    nodeIds[threadIdx.x] = aggMapAdjacency[aggBegin + threadIdx.x];
                __syncthreads();

                // Each thread loads it's neighbors list into registers, translating into
                // aggregate offsets.
                int neighbors[40];
                int nextNeighbor = 0;
                int nodeId = -1;
                if (threadIdx.x < aggSize)
                {
                    nodeId = nodeIds[threadIdx.x];
                    int start = adjIndices[nodeId];
                    int end = adjIndices[nodeId + 1];
                    for (int i = start; i < end; i++)
                    {
                        int neighborId = adjacency[i];
                        int a = 0, b = aggSize - 1, midpoint;
                        while (a < b)
                        {
                            midpoint = a + ((b - a) / 2);
                            if (nodeIds[midpoint] < neighborId)
                                a = midpoint + 1;
                            else
                                b = midpoint;
                        }

                        if (nodeIds[a] == neighborId)
                        {
                            neighbors[nextNeighbor++] = a;
                        }
                    }
                }
                __syncthreads();
                
                // Find the farthest node from the lowest indexed node (first root point)
                // Start by marking the first node and threads without a node as seen  
                // Mark initial distances in scratch vector
                if (threadIdx.x < aggSize)
                    scratchA[threadIdx.x] = threadIdx.x == 0 ? 0 : -1;
                int myDist = threadIdx.x == 0 ? 0 : -1;
                bool swapped = false;
                incomplete = true;
                __syncthreads(); 

                while (incomplete)
                {   
                    // Set the incomplete flag to false
                    incomplete = false;
                    __syncthreads();

                    // Check if a neighbor has a positive distance
                    if (threadIdx.x < aggSize && myDist == -1)
                    {
                        for (int i = 0; i < nextNeighbor; i++)
                        {
                            int neighborDist = scratchA[neighbors[i]];
                            if (neighborDist > -1)
                                myDist = neighborDist + 1;
                        }
                    }
                    __syncthreads();

                    if (threadIdx.x < aggSize && myDist > 0 && !swapped)
                    {
                        swapped = true;
                        scratchA[threadIdx.x] = myDist;
                        rootA = threadIdx.x;
                        incomplete = true;
                    }  
                    __syncthreads();
                }

                // Find the farthest node from the first root point (second root point)
                // Start by marking the first node and threads without a node as seen  
                // Mark initial distances in scratch vector
                if (threadIdx.x < aggSize)
                    scratchA[threadIdx.x] = threadIdx.x == rootA ? 0 : -1;
                myDist = threadIdx.x == rootA ? 0 : -1;
                swapped = false;
                incomplete = true;
                __syncthreads(); 

                while (incomplete)
                {   
                    // Set the incomplete flag to false
                    incomplete = false;
                    __syncthreads();

                    // Check if a neighbor has a positive distance
                    if (threadIdx.x < aggSize && myDist == -1)
                    {
                        for (int i = 0; i < nextNeighbor; i++)
                        {
                            int neighborDist = scratchA[neighbors[i]];
                            if (neighborDist > -1)
                            {
                                myDist = neighborDist + 1;
                            }
                        }
                    }
                    __syncthreads();

                    if (threadIdx.x < aggSize && myDist > 0 && !swapped)
                    {
                        swapped = true;
                        scratchA[threadIdx.x] = myDist;
                        rootB = threadIdx.x;
                        incomplete = true;
                    }  
                    __syncthreads();
                }

                // Setting an assigned aggregate label (In ScratchA) for every node with the node at 
                // rootA being assigned the current aggregate ID and the node at rootB
                // being assigned the newAgg ID and set initial distances from a root node
                // (In ScratchB) for each node, -1 for unknown and 0 for the roots
                int myAggregate = -1;
                if (threadIdx.x == rootA)
                    myAggregate = currentAgg;
                if (threadIdx.x == rootB)
                    myAggregate = newAgg;
                if (threadIdx.x < aggSize)
                {      
                    scratchA[threadIdx.x] = myAggregate;
                    scratchB[threadIdx.x] = myAggregate > -1 ? 0 : -1;
                }
                incomplete = true;
                __syncthreads();

                // Assign nodes to each aggregate until no unassigned nodes remain.
                while (incomplete)
                {   
                    // Set the incomplete flag to false
                    incomplete = false;
                    __syncthreads();

                    if (threadIdx.x < aggSize && myAggregate == -1)
                    {
                        for (int i = 0; i < nextNeighbor; i++)
                        {
                            int neighborAgg = scratchA[neighbors[i]];
                            if (neighborAgg > -1)
                            {
                                myDist = scratchB[neighbors[i]] + 1;
                                myAggregate = neighborAgg;
                            }
                        }
                        if (myAggregate == -1)
                            incomplete = true;
                        if (myAggregate == newAgg)
                            atomicAdd((unsigned int*)&bCount, (unsigned)1);
                        if (myAggregate == currentAgg)
                            atomicAdd((unsigned int*)&aCount, (unsigned)1);
                    }
                    __syncthreads();

                    if (threadIdx.x < aggSize)
                    {
                        scratchA[threadIdx.x] = myAggregate;
                        scratchB[threadIdx.x] = myDist;
                    }
                    __syncthreads();
                }

                // If the split was uneven try to repair it
                int sizeDifference = aCount > bCount ? aCount - bCount : bCount - aCount;
                bool moveToA = aCount < bCount;
                __shared__ int moved;
                moved = 0;
                int toMove = sizeDifference / 2;
                incomplete = true;
                __syncthreads();

                while (incomplete && moved < toMove)
                {
                    incomplete = false;
                    __syncthreads();
                    bool swapping = false;
                    int newDist = INT_MAX;
                    if (threadIdx.x < aggSize)
                    {
                        bool canSwap = moveToA ? myAggregate == newAgg : myAggregate == currentAgg;
                        bool borderNode = false;

                        // Check if this node has no same aggregate neighbors of higher distance
                        // and on a border
                        for (int i = 0; i < nextNeighbor; i++)
                        {
                            int neighborAgg = scratchA[neighbors[i]];
                            int neighborDist = scratchB[neighbors[i]];
                            if (neighborAgg == myAggregate && neighborDist > myDist)
                                canSwap = false;

                            if (neighborAgg != myAggregate)
                            {
                                if (neighborDist + 1 < newDist)
                                    newDist = neighborDist + 1;
                                borderNode = true;
                            }
                        }

                        // If a node could swap see if it will
                        if (borderNode && canSwap && atomicAdd((unsigned int*)&moved, 1) < toMove)
                        {
                            swapping = true;
                        }
                    }
                    __syncthreads();
                    if (swapping)
                    {
                        int a = moveToA ? 1 : -1;
                        atomicAdd((unsigned int*)&bCount, -a);
                        atomicAdd((unsigned int*)&aCount, a);
                        scratchA[threadIdx.x] = moveToA ? currentAgg : newAgg;
                        scratchB[threadIdx.x] = newDist;
                        incomplete = true;
                    }
                    __syncthreads();
                }

                // Write out the values to the aggregation array
                if (threadIdx.x < aggSize)
                {
                    aggregation[nodeId] = scratchA[threadIdx.x];
                }

            }
            __global__ void MakeSplitsWeighted_Large(int baseAggregateIdx, 
                                int* splitting, 
                                int* aggregation, 
                                int* aggMapAdjIndices, 
                                int* aggMapAdjacency, 
                                int* adjIndices, 
                                int* adjacency,
                                int* weights) {
                int currentAgg = splitting[blockIdx.x];
                int aggBegin = aggMapAdjIndices[currentAgg];
                int aggSize = aggMapAdjIndices[currentAgg + 1] - aggBegin;
                int newAgg = baseAggregateIdx + blockIdx.x;
                
                // Debug
                int iterationCount = 0;
//                if (blockIdx.x == 0 && threadIdx.x == 0) {
//                    printf("Starting MakeSplitsWeighted kernel for aggregate %d with node count %d\n", 
//                            currentAgg, aggSize);
//                }                
                
                __shared__ int nodeIds[256];
                __shared__ int nodeWeights[256];
                __shared__ int scratchA[256];
                __shared__ int scratchB[256];
                __shared__ int rootA, rootB;
                __shared__ int aCount, bCount;
                __shared__ bool incomplete;
                incomplete = true;

                // Load in the node Id's from the aggregate map to the shared array:
                if (threadIdx.x < aggSize) {
                    nodeIds[threadIdx.x] = aggMapAdjacency[aggBegin + threadIdx.x];
                    nodeWeights[threadIdx.x] = weights[nodeIds[threadIdx.x]];
                }
                __syncthreads();
                
                // Each thread loads it's neighbors list into registers, translating into
                // aggregate offsets.
                int neighbors[40];
                int nextNeighbor = 0;
                D::LoadLocalNeighbors(&neighbors[0],
                                        &nextNeighbor,
                                        aggSize,
                                        &nodeIds[0],
                                        adjIndices,
                                        adjacency);
                
                // Flood fill distances from node 0 to find first root node
                D::FloodFillDistanceFrom(0, 
                        &scratchA[0],
                        aggSize,
                        &neighbors[0],
                        nextNeighbor,
                        &rootA,
                        &incomplete);
                
                // Testing templated function call
//                D::PrintSharedArray(aggSize, &scratchA[0], "Before calling transform");
//                T::Transform(aggSize, &scratchA[0], T::AddTo<int>(3));
//                D::PrintSharedArray(aggSize, &scratchA[0], "After calling transform");
                
                // Flood fill distances from rootA to find rootB
                D::FloodFillDistanceFrom(rootA, 
                        &scratchA[0],
                        aggSize,
                        &neighbors[0],
                        nextNeighbor,
                        &rootB,
                        &incomplete);
                
                // Setting an assigned aggregate label (In ScratchA) for every node with the node at 
                // rootA being assigned the current aggregate ID and the node at rootB
                // being assigned the newAgg ID and set initial distances from a root node
                // (In ScratchB) for each node, -1 for unknown and 0 for the roots
                int myAggregate = -1;
                if (threadIdx.x == rootA) {
                    myAggregate = currentAgg;
                    aCount = weights[threadIdx.x];
                }
                if (threadIdx.x == rootB) {
                    myAggregate = newAgg;
                    bCount = weights[threadIdx.x];
                }
                if (threadIdx.x < aggSize)
                {      
                    scratchA[threadIdx.x] = myAggregate;
                    scratchB[threadIdx.x] = myAggregate > -1 ? 0 : -1;
                }
                incomplete = true;
//                D::WarpReport("before sync");
//                __syncthreads();
//                D::WarpReport("after sync");
//
//                D::PrintSharedArray(aggSize, 
//                                    &scratchA[0], 
//                                    "Pre-Initial aggregate assignments");
//                D::PrintSharedArray(aggSize, 
//                                    &scratchB[0], 
//                                    "Pre-Initial distance assignments");
                
                int myDist = threadIdx.x == 0 ? 0 : -1;
                
                // Assign nodes to each aggregate until no unassigned nodes remain.
                iterationCount = 0;
                while (incomplete && iterationCount < 10)
                {
                    iterationCount++;
//                    if (blockIdx.x == 0 && threadIdx.x == 0) {
//                        printf("\tStarting an initial allocation cycle. aCount=%d bCount=%d\n", aCount, bCount);
//                    }
                    // Set the incomplete flag to false
                    __syncthreads();
                    incomplete = false;
                    __syncthreads();

                    if (threadIdx.x < aggSize && myAggregate == -1)
                    {
                        for (int i = 0; i < nextNeighbor; i++)
                        {
                            int neighborAgg = scratchA[neighbors[i]];
                            if (neighborAgg > -1)
                            {
                                myDist = scratchB[neighbors[i]] + 1;
                                myAggregate = neighborAgg;
                            }
                        }
                        if (myAggregate == -1)
                            incomplete = true;
                        if (myAggregate == newAgg)
                            atomicAdd((unsigned int*)&bCount, (unsigned)nodeWeights[threadIdx.x]);
                        if (myAggregate == currentAgg)
                            atomicAdd((unsigned int*)&aCount, (unsigned)nodeWeights[threadIdx.x]);
                    }
                    __syncthreads();

                    if (threadIdx.x < aggSize)
                    {
                        scratchA[threadIdx.x] = myAggregate;
                        scratchB[threadIdx.x] = myDist;
                    }
                    __syncthreads();
                }
                __syncthreads();
                // Printing out the initial aggregate assignments made.
                //D::PrintSharedArray(aggSize, scratchA, "Initial Aggregate Assignments");
//                D::PrintSharedArray(aggSize, 
//                                    &scratchA[0], 
//                                    "Initial aggregate assignments");
//                D::PrintSharedArray(aggSize, 
//                                    &scratchB[0], 
//                                    "Initial distance assignments");
//                                
//                // Printing a message if max iterations exceeded.
//                if (blockIdx.x == 0 && threadIdx.x == 0 && incomplete) {
//                    printf("***********Max Iterations Exceeded*************\n");
//                }
//                __syncthreads();
                
                
                // If the split was uneven try to repair it
                __shared__ int goodSwaps[20];           // The id of nodes that have desirable swaps
                __shared__ int improvement[20];         // How much improvement the swap would make
                __shared__ int insertID;                // The index at which to insert new item
                
                
                incomplete = true;
                __syncthreads();

                iterationCount = 0;
                while (incomplete && iterationCount < 10)
                {
                    iterationCount++;
                    // Reset values
                    int sizeDifference = aCount > bCount ? aCount - bCount : bCount - aCount;
                    bool moveToA = aCount < bCount;
                    
//                    if (blockIdx.x == 0 && threadIdx.x == 0) {
//                        printf("\tStarting a size correction cycle: A=%d B=%d\n", 
//                                currentAgg, newAgg);
//                        printf("\t\taCount:=%d bCount=%d sizeDifference=%d moveToA=%s\n\n", 
//                                aCount, bCount, sizeDifference, (moveToA ? "True" : "False"));
//                    }
                    
                    insertID = 0;
                    __syncthreads();
                    
                    int newDist = INT_MAX;
                    if (threadIdx.x < aggSize)
                    {
                        bool canSwap = moveToA ? myAggregate == newAgg : myAggregate == currentAgg;
                        bool borderNode = false;

                        // Check if this node has no same aggregate neighbors of higher distance
                        // and on a border
                        for (int i = 0; i < nextNeighbor; i++)
                        {
                            int neighborAgg = scratchA[neighbors[i]];
                            int neighborDist = scratchB[neighbors[i]];
                            if (neighborAgg == myAggregate && neighborDist > myDist)
                                canSwap = false;

                            if (neighborAgg != myAggregate)
                            {
                                if (neighborDist + 1 < newDist)
                                    newDist = neighborDist + 1;
                                borderNode = true;
                            }
                        }

                        // If a node could swap see how attractive the swap would be
                        if (borderNode && canSwap)
                        {
                            int newA = moveToA ? 
                                        aCount + nodeWeights[threadIdx.x] : 
                                        aCount - nodeWeights[threadIdx.x];
                            int newB = moveToA ? 
                                        bCount - nodeWeights[threadIdx.x] : 
                                        bCount + nodeWeights[threadIdx.x];
                            int newSizeDifference = newA > newB ? 
                                                        newA - newB : 
                                                        newB - newA;
                            if (newSizeDifference < sizeDifference) {
                                int newID = atomicAdd((int *)&insertID, 1) - 1;
                                if (newID < 20) {
                                    goodSwaps[newID] = threadIdx.x;
                                    improvement[newID] = newSizeDifference;
                                } 
                            }
                        }
                    }
                    __syncthreads();
                    
                    // Now finding the best swap to make and making it
                    if (insertID > 0)
                    {
//                        if (blockIdx.x == 0 && threadIdx.x == 0) {
//                            printf("\tThere are %d possible swaps marked\n", insertID);
//                        }
                        // Have zero thread look through options for best
                        if (threadIdx.x == 0) {
                            // Checking each option found and getting the best one
                            int bestValue = INT_MAX;
                            int swapId = -1;
                            for (int i = 0; i < insertID && i < 20; i++) {
                                
                                // Debug
//                                if (blockIdx.x == 0 && threadIdx.x == 0) {
//                                    printf("\t\tNode %d can swap with improvement %d\n", 
//                                            goodSwaps[i], improvement[i]);
//                                }
                                
                                if (improvement[i] < bestValue) {
                                    bestValue = improvement[i];
                                    swapId = goodSwaps[i];
                                }
                            }
                            insertID = swapId;
                        }
//                        if (blockIdx.x == 0 && threadIdx.x == 0) {
//                            printf("\tNode %d in aggregate %d selected to swap\n", 
//                                    insertID, scratchA[insertID]);
//                        }
                        __syncthreads();
                        
                        // Have the thread belonging to the swap node do the swap
                        if (threadIdx.x == insertID) {
                            myAggregate = moveToA ? currentAgg : newAgg;
                            scratchA[threadIdx.x] = myAggregate;
                            scratchB[threadIdx.x] = newDist;
                            aCount = moveToA ? aCount + nodeWeights[threadIdx.x] : aCount - nodeWeights[threadIdx.x];
                            bCount = moveToA ? bCount - nodeWeights[threadIdx.x] : bCount + nodeWeights[threadIdx.x];
                        }
                        
                        __syncthreads();
                        
                        // Now recompute the distances to make sure things are still 
                        // connected.
                        __shared__ bool changed;
                        scratchB[threadIdx.x] = threadIdx.x == rootA || threadIdx.x == rootB ?
                                                0 : -1;
                        changed = true;
                        __syncthreads();
                        while (changed) {
                            changed = false;
                            
                            // Check if a neighbor has a positive distance
                            if (threadIdx.x < aggSize && scratchB[threadIdx.x] == -1) {
                                for (int i = 0; i < nextNeighbor; i++) {
                                    // If neighbor has a distance and is in the same 
                                    // aggregate fill distance from it.
                                    if (scratchA[neighbors[i]] == scratchA[threadIdx.x] && scratchB[neighbors[i]] > -1) {
                                        scratchB[threadIdx.x] = scratchB[neighbors[i]] + 1;
                                        changed = true;
                                    }
                                }
                            }
                            __syncthreads();
                        }
                        if (threadIdx.x < aggSize && scratchB[threadIdx.x] == -1) {
                            changed = true;
                        }
                        __syncthreads();
                        
                        
//                        if (changed && blockIdx.x == 0 && threadIdx.x == 0) {
//                            printf("\nWhile splitting aggregate %d into %d:\n",
//                                    currentAgg, newAgg);
//                            int problem = nodeIds[insertID];
//                            printf("\tId %d (node %d) broke things when swapped.\n",
//                                    insertID, problem);
//                            printf("\tDetected a disconnect. Reverting and stopping.\n");
//                        }
                        
                        if (changed && threadIdx.x == insertID) {
                            scratchA[threadIdx.x] = scratchA[threadIdx.x] == newAgg ?
                                                    currentAgg : newAgg;
                            incomplete = true;
                        }
                        
                        __syncthreads();
//                        if (changed && blockIdx.x == 0 && threadIdx.x == 0) {
//                            printf("Incomplete flag marked as: %s\n",
//                                    incomplete ? "True" : "False");
//                        }
//                        __syncthreads();
                        
                        
                        
//                        if (blockIdx.x == 0 && threadIdx.x == 0) {
//                            printf("\t\tNode %d now in aggregate %d with distance %d\n", 
//                                    insertID, scratchA[insertID], scratchB[insertID]);
//                        }
                    }
                    else {
                        incomplete = false;
                    }
                    __syncthreads();
                }
//                if (blockIdx.x == 0 && threadIdx.x == 0 && incomplete) {
//                    printf("***********Max Iterations Exceeded*************\n");
//                }
//                
//                D::PrintSharedArray(aggSize, 
//                                    &scratchA[0], 
//                                    "Final aggregate assignments");
//                D::PrintSharedArray(aggSize, 
//                                    &scratchB[0], 
//                                    "Final distance assignments");

                // Write out the values to the aggregation array
                if (threadIdx.x < aggSize)
                {
                    aggregation[nodeIds[threadIdx.x]] = scratchA[threadIdx.x];
                }
            }
            __global__ void MarkSplits(int size, 
                                bool force, 
                                int minPartSize, 
                                int maxPartSize, 
                                int* partSizes, 
                                int* splitsToMake) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size)
                {
                    int currentSize = partSizes[idx];
                    bool shouldSplit = currentSize > maxPartSize && (force || currentSize > minPartSize * 2);
                    splitsToMake[idx] = shouldSplit ? 1 : 0;
                } 
            }
            __global__ void FindDesirableMerges(int size, 
                                int minSize, 
                                int maxSize, 
                                bool force, 
                                int* adjIndices, 
                                int* adjacency, 
                                int *partSizes, 
                                int* desiredMerges, 
                                int* merging) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size)
                {
                    // Only evaluate if the aggregate is not marked as active (merging
                    // or no possible merges)
                    if (merging[idx] != 1)
                    {
                        // Check through all neighboring aggregates for most desirable
                        int currentSize = partSizes[idx];
                        int checkedNeighbors = 0;
                        float bestDesirability = 0;
                        int mostDesirable = -1;
                        int start = adjIndices[idx];
                        int end = adjIndices[idx + 1];
                        for (int i = start; i < end; i++)
                        {
                            int neighborAgg = adjacency[i];

                            // Only active neighbor aggregates should be looked at:
                            if (merging[neighborAgg] != 1)
                            {
                                checkedNeighbors++;
                                int neighborSize = partSizes[neighborAgg];

                                float desirability = 0;
                                desirability += currentSize < minSize ? minSize - currentSize : 0;
                                desirability += neighborSize < minSize ? minSize - neighborSize : 0;
                                int totalSize = currentSize + neighborSize;
                                if (totalSize > maxSize)
                                    desirability *= force ? 1.0/(totalSize - maxSize) : 0;

                                // If this merge is the most desirable seen mark it
                                if (desirability > bestDesirability)
                                {
                                    bestDesirability = desirability;
                                    mostDesirable = neighborAgg;
                                }
                            }
                        }

                        if (mostDesirable == -1)
                            merging[idx] = 1;

                        if (currentSize < minSize && force && mostDesirable == -1)
                            printf("Aggregate %d is too small but found no merges! %d / %d neighbors checked.\n",idx, checkedNeighbors, end-start);

                        desiredMerges[idx] = mostDesirable;   
                    }
                }    
            }
            __global__ void FindDesirableMergeSplits(int size, 
                                int minSize, 
                                int maxSize, 
                                int desiredSize, 
                                int* adjIndices, 
                                int* adjacency, 
                                int* partSizes, 
                                int* desiredMerges, 
                                int* merging) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size)
                {
                    // Only evaluate if the aggregate is not marked as active (merging
                    // or no possible merges)
                    if (merging[idx] != 1)
                    {
                        // Check through all neighboring aggregates for most desirable
                        int currentSize = partSizes[idx];
                        int checkedNeighbors = 0;
                        bool currentOutSized = currentSize < minSize || currentSize > maxSize;
                        float bestDesirability = 0;
                        int mostDesirable = -1;
                        int start = adjIndices[idx];
                        int end = adjIndices[idx + 1];
                        for (int i = start; i < end; i++)
                        {
                            int neighborAgg = adjacency[i];

                            // Only active neighbor aggregates should be looked at:
                            if (merging[neighborAgg] != 1)
                            {
                                checkedNeighbors++;
                                int neighborSize = partSizes[neighborAgg];
                                bool neighborOutSized = neighborSize < minSize || neighborSize > maxSize;
                                int totalSize = currentSize + neighborSize;
                                bool legalPair = (neighborOutSized || currentOutSized) && totalSize > minSize * 2 && totalSize < maxSize * 2;
                                float desirability = legalPair ? 1.0 / abs(desiredSize - (currentSize + neighborSize)) : 0;

                                // If this merge is the most desirable seen mark it
                                if (desirability > bestDesirability)
                                {
                                    bestDesirability = desirability;
                                    mostDesirable = neighborAgg;
                                }
                            }
                        }

                        if (mostDesirable == -1)
                            merging[idx] = 1;

                        desiredMerges[idx] = mostDesirable;   
                    }
                }    
            }
            __global__ void MarkMerges(int size, 
                                int* desiredMerges, 
                                int* merging, 
                                int* mergesToMake, 
                                int* incomplete) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size)
                {
                    // Find what aggregate this one wants to merge with
                    int desiredMerge = desiredMerges[idx];

                    // If this aggregate has a real potential merger:
                    if (desiredMerge >= 0)
                    {
                        // If the aggregates agree to merge mark as merging
                        if (desiredMerges[desiredMerge] == idx)
                        {
                            // Mark the merge as the higher indexed aggregate merging into lower
                            if (desiredMerge > idx)
                                mergesToMake[desiredMerge] =  idx;
                            else
                                mergesToMake[idx] = desiredMerge;

                            // Mark both aggregates as merging
                            merging[idx] = 1;
                            merging[desiredMerge] = 1;
                        }
                        // Otherwise mark incomplete to check again
                        else
                        {
                            incomplete[0] = 1;
                        }
                    }
                }    
            }
        }
        
        // Public methods for merge split conditioner
        MergeSplitConditionerGPU::MergeSplitConditionerGPU(Graph_d& graph, 
                                            IntVector_d& aggregation) {
            this->graph = &graph;
            this->aggregation.assign(aggregation.begin(), aggregation.end());
            inducedGraph = GraphHelpers::GetInducedGraph(graph, aggregation);
            
            // Getting the sizes of each aggregate:
            GraphHelpers::getPartSizes(aggregation, partSizes);
            
            verbose = false;
            minSize = 20;
            maxSize = 30;
            outsizedParts = 0;
            merges = 0;
            mergeSplits = 0;
            splits = 0;
        }
        void MergeSplitConditionerGPU::SetSizeBounds(int min, 
                                            int max) {
            minSize = min;
            maxSize = max;
        }
        void MergeSplitConditionerGPU::SetVerbose(bool v) {
            verbose = v;
        }
        void MergeSplitConditionerGPU::SetNodeWeights(IntVector_d &input) {
            nodeWeights.swap(input);
            GraphHelpers::getPartSizes(aggregation, weightedSizes, nodeWeights);
        }
        IntVector_d* MergeSplitConditionerGPU::GetAggregation() {
            return &aggregation;
        }
        IntVector_d* MergeSplitConditionerGPU::GetNodeWeights() {
            return &nodeWeights;
        }
        void MergeSplitConditionerGPU::CycleMerges(bool force) {
            int count = 0;
            while (MarkMerges(force))
            {
                MakeMerges(false);
                count++;
            }
            if (verbose)
                printf("Finished cycling merges after %d cycles.\n", count);
        }
        void MergeSplitConditionerGPU::CycleSplits(bool force) {
            int count = 0;
            while (MarkSplits(force) && count < 10)
            {
                MakeSplits();
                count++;
            }
            if (verbose)
                printf("Finished cycling splits after %d cycles.\n", count);
        }
        void MergeSplitConditionerGPU::CycleMergeSplits(float minImprove, 
                                            int desiredSize) {
            // Start with an initial cycle
            bool somethingDone = MakeMergeSplits(desiredSize);
            
            // Choosing which sizes to use:
            IntVector_d *sizes = &partSizes;
            if (nodeWeights.size() > 0)
                sizes = &weightedSizes;
            
            // Check to see how much improvement was made
            int after = thrust::count_if(sizes->begin(), sizes->end(), Functors::isOutSized(minSize, maxSize));
            float improvement = (float)(outsizedParts - after)/outsizedParts;
            outsizedParts = after;
            
            // While good progress is being made continue cycling
            while (improvement > minImprove && somethingDone)
            {
                // Perform Cycle and check improvement
                somethingDone = MakeMergeSplits(desiredSize);
                after = thrust::count_if(sizes->begin(), sizes->end(), Functors::isOutSized(minSize, maxSize));
                improvement = (float)(outsizedParts - after)/outsizedParts;
                outsizedParts = after;
            }            
        }
        bool MergeSplitConditionerGPU::Condition(int desiredSize,
                                            bool respectUpper, 
                                            float tolerance, 
                                            float minImprove, 
                                            int maxCycles) {
            if (verbose)
                PrintProgress(&cout, "Starting conditioning.", true, true, true, true);
            
            // Start by making any optimal merges and splits
            if (verbose)
                printf("Starting to CycleMerges\n");
            CycleMerges(false);
            if (verbose)
                printf("Starting to CycleSplits\n");
            CycleSplits(false);
            
            if (verbose)
                printf("Starting to CycleMergeSplits\n");
            // Cycle MergeSplits too, to make sure outsizedParts has a value
            CycleMergeSplits(minImprove, desiredSize);
            
            // Find improvement ratio from initial cycle
            float currentRatio = (float)outsizedParts / partSizes.size();
            if (verbose)
                printf("Initial outsized ratio is: %f\n", currentRatio);
            
            // Starting main cycle phase
            int counter = 0;
            bool highCycle = false;
            while(currentRatio > tolerance && counter++ < maxCycles)
            {
                if (verbose)
                    printf("Starting %s conditioning cycle %d\n",
                            highCycle ? "high" : "low", counter);
                
                if (highCycle)
                    CycleMerges(true);
                else
                    CycleSplits(true);
                CycleMergeSplits(minImprove, desiredSize);

                // Checking the current improvement ratio
                if ((highCycle && !respectUpper) || (!highCycle && respectUpper))
                    currentRatio = (float)outsizedParts / partSizes.size();

                // Switch cycle type
                highCycle = !highCycle;
                
                if (verbose)
                {
                    stringstream ss;
                    ss << "After condition cycle: " << counter;
                    PrintProgress(&cout, ss.str(), true, true, true, true);
                }
            }

            // Cleaning up
            if (respectUpper)
            {
                CycleSplits(true);
                CycleMerges(false);
            }
            else
                CycleMerges(true);

            // Checking if we match criteria given:
            int undersized = thrust::count_if(partSizes.begin(), partSizes.end(), Functors::lessThan(minSize));
            int oversized = thrust::count_if(partSizes.begin(), partSizes.end(), Functors::greaterThan(maxSize));

//            printf("Checking for CUDA errors.\n");
//            CheckCudaError(cudaDeviceSynchronize(),__FILE__, __LINE__);
            
//            printf("Checking if aggregation is valid.\n");
//            
//            if (Aggregation::IsValidAggregation(*graph, aggregation, true))
//                printf("Aggregation validates after conditioning.\n");
//            else {
//                printf("Aggregation does not validate after conditioning!\n");
//                int t;
//                cin >> t;
//            }
            if (verbose)
                PrintProgress(&cout, "After conditioning completed.", true, true, true, true);
            
            // Checking if the size constraints are met for the return
            if (respectUpper)
                return (oversized == 0 && (float)outsizedParts / partSizes.size() < tolerance);
            else
                return (undersized == 0 && (float)outsizedParts / partSizes.size() < tolerance);
            
        }
        void MergeSplitConditionerGPU::PrintProgress(ostream* output, 
                                            string note,
                                            bool graphStat,
                                            bool progressStat,
                                            bool sizeStat,
                                            bool memStat) {
            *output << "\n------------------- Progress Check ------------------\n";
            *output << "Note: " << note.c_str() << "\n";

            if (graphStat)
                PrintGraphStats(output, false);
            
            if (progressStat)
                PrintProgressStats(output, false);
            
            if (sizeStat)
                PrintSizeStats(output, false);
            
            if (memStat)
                PrintMemoryStats(output, false);

            *output << "-----------------------------------------------------\n\n";
        }
        void MergeSplitConditionerGPU::PrintSizeStats(ostream* output,
                                            bool makeHeader) {
            if (makeHeader)
                *output << "\n--------------------- Size Check --------------------\n";
            
            // Choosing which sizes to use:
            IntVector_d *sizes = &partSizes;
            if (nodeWeights.size() > 0)
                sizes = &weightedSizes;
            
            int undersized = thrust::count_if(sizes->begin(), 
                                                sizes->end(), 
                                                Functors::lessThan(minSize));
            int oversized = thrust::count_if(sizes->begin(), 
                                                sizes->end(), 
                                                Functors::greaterThan(maxSize));
            int largest = thrust::reduce(sizes->begin(), 
                                                sizes->end(), 
                                                0, 
                                                thrust::maximum<int>());
            int smallest = thrust::reduce(sizes->begin(), 
                                                sizes->end(), 
                                                INT_MAX, 
                                                thrust::minimum<int>());
            
            *output << "Aggregate size statistics:";
            *output << "\n\tUndersized(<" << minSize << "): " << undersized << " / " << (partSizes.size()) << " Total";
            *output << "\n\tOversized(>" << maxSize << "): " << oversized << " / " << (partSizes.size()) << " Total";
            *output << "\n\tSmallest: " << smallest;
            *output << "    Largest: " << largest << "\n";
            
            if (nodeWeights.size() > 0)
            {
                largest = thrust::reduce(partSizes.begin(), 
                                                    partSizes.end(), 
                                                    0, 
                                                    thrust::maximum<int>());
                smallest = thrust::reduce(partSizes.begin(), 
                                                    partSizes.end(), 
                                                    INT_MAX, 
                                                    thrust::minimum<int>());
                *output << "\n\tUnweighted: Smallest: " << smallest;
                *output << "    Largest: " << largest << "\n";
            }
            
            if (makeHeader)
                *output << "-----------------------------------------------------\n\n";            
        }
        void MergeSplitConditionerGPU::PrintMemoryStats(ostream* output,
                                            bool makeHeader) {
            if (makeHeader)
                *output << "\n-------------------- Memory Check -------------------\n";
            
            size_t avail;
            size_t total;
            cudaMemGetInfo(&avail, &total);
            avail /= 1000000;
            total /= 1000000;
            *output << "Device Memory Status:";
            *output << "\n\tAvailable: " << (int)avail; 
            *output << "\tTotal: " << (int)total;
            *output << "\tUtilized: " << (int)(total-avail) << "\n";
            
            if (makeHeader)
                *output << "-----------------------------------------------------\n\n";            
        }
        void MergeSplitConditionerGPU::PrintProgressStats(ostream* output,
                                            bool makeHeader) {
            if (makeHeader)
                *output << "\n------------------- Progress Check ------------------\n";
            
            *output << "Processing done:";
            *output << "\n\tMerges: " << merges;
            *output << "\tSplits: " << splits;
            *output << "\tMerge-Splits: " << mergeSplits << "\n";
            
            if (makeHeader)
                *output << "-----------------------------------------------------\n\n";            
        }
        void MergeSplitConditionerGPU::PrintGraphStats(ostream* output,
                                            bool makeHeader) {
            if (makeHeader)
                *output << "\n----------------- Graph Information -----------------\n";
            
            int totalWeight = thrust::reduce(nodeWeights.begin(), nodeWeights.end());
            int minWeight = thrust::reduce(nodeWeights.begin(), 
                                            nodeWeights.end(), 
                                            INT_MAX, 
                                            thrust::minimum<int>());
            int maxWeight = thrust::reduce(nodeWeights.begin(), 
                                            nodeWeights.end(), 
                                            0,
                                            thrust::maximum<int>());
            
            IntVector_d *valences = GraphHelpers::GetValences(*graph);
            int minValence = thrust::reduce(valences->begin(),
                                            valences->end(),
                                            INT_MAX,
                                            thrust::minimum<int>());
            int maxValence = thrust::reduce(valences->begin(),
                                            valences->end(),
                                            0,
                                            thrust::maximum<int>());
            valences->clear();
            delete valences;
            
            *output << "Graph Information:";
            *output << "\n\tNodes: " << graph->Size();
            if (nodeWeights.size() > 0)
                *output << "   Graph is weighted";
            else
                *output << "   Graph is unweighted";
            
            *output << "\n\tMin. Valence: " << minValence;
            *output << "   Max. Valence: " << maxValence;
            *output << "   Avg. Valence: " << ((float)graph->adjacency->size()/graph->Size());
            
            if (nodeWeights.size() > 0) {
                *output << "\n\tTotal Weight: " << totalWeight;
                *output << "   Avg. Weight: " << ((float)totalWeight / graph->Size());
                *output << "   Min. Weight: " << minWeight;
                *output << "   Max. Weight: " << maxWeight;
            }
            *output << "\n";
            
            if (makeHeader)
                *output << "-----------------------------------------------------\n\n";            
        }
        void MergeSplitConditionerGPU::InteractiveConsole(string message) {
            // Start off by printing overall status info and message
            PrintProgress(&cout, message, true, true, true, false);
            
            // Setting needed variables to defaults
            float minImprove = .1;
            int desiredSize = (minSize + maxSize) / 2;
            float tolerance = .1;
            int maxCycles = 10;
            bool cycling = true;
            bool respectUpper = true;
            
            // Starting the main prompt:
            char operation;
            printf("\nIC:");
            cin >> operation;
            while (operation != 'd')
            {
                if (operation == 'o' || operation == 'f')
                {
                    bool force = operation == 'f';
                    cin >> operation;
                    if (operation == 'm')
                    {
                        if (cycling)
                            CycleMerges(force);
                        else {
                            MarkMerges(force);
                            MakeMerges(false);
                        }
                        string msg = force ? "After forced merges" : "After optimal merges";
                        PrintProgress(&cout, msg, false, true, true, false);
                    }
                    if (operation == 's')
                    {
                        if (cycling)
                            CycleSplits(force);
                        else {
                            MarkSplits(force);
                            MakeSplits();
                        }
                        string msg = force ? "After forced splits" : "After optimal splits";
                        PrintProgress(&cout, msg, false, true, true, false);                               
                    }
                    if (operation == 'g')
                    {
                        if (cycling)
                            CycleMergeSplits(minImprove, desiredSize);
                        else 
                            MakeMergeSplits(desiredSize);
                        PrintProgress(&cout, "After merge-splits", false, true, true, false);
                    }
                }
                else if (operation == 's') {
                    // Printing the current values of the variables
                    string cyclingFlag = cycling ? "True" : "False";
                    string respectUpperFlag = respectUpper ? "True" : "False";
                    cout << "\nCurrent values of variables:";
                    cout << "\n\tminSize: " << minSize;
                    cout << "   maxSize: " << maxSize;
                    cout << "   desiredSize: " << desiredSize;
                    cout << "   maxCycles: " << maxCycles;
                    cout << "\n\tminImprove: " << minImprove;
                    cout << "   tolerance: " << tolerance;
                    cout << "   cycling: " << cyclingFlag;
                    cout << "   respectUpper: " << respectUpperFlag;
                    cout << "\n\nEnter new values in same order\nIC:";
                    
                    // Grabbing the new values
                    cin >> minSize;
                    cin >> maxSize;
                    cin >> desiredSize;
                    cin >> maxCycles;
                    cin >> minImprove;
                    cin >> tolerance;
                    cin >> cycling;
                    cin >> respectUpper;
                    
                    // Confirming the entry
                    cyclingFlag = cycling ? "True" : "False";
                    respectUpperFlag = respectUpper ? "True" : "False";
                    cout << "\nNew values of variables:";
                    cout << "\n\tminSize: " << minSize;
                    cout << "   maxSize: " << maxSize;
                    cout << "   desiredSize: " << desiredSize;
                    cout << "   maxCycles: " << maxCycles;
                    cout << "\n\tminImprove: " << minImprove;
                    cout << "   tolerance: " << tolerance;
                    cout << "   cycling: " << cyclingFlag;
                    cout << "   respectUpper: " << respectUpperFlag << "\n\n";
                }
                else if (operation == 'c')
                {
                    Condition(desiredSize, respectUpper, tolerance, minImprove, maxCycles);
                    PrintProgress(&cout, "After conditioning", false, true, true, false);
                }
                else if (operation == 'v') {
                    bool valid = Aggregation::IsValidAggregation(*graph, aggregation, false);
                    if (valid)
                        printf("Aggregation is valid\n");
                    else
                        printf("Aggregation is not valid!\n");
                }
                else if (operation == 'l') {
                    bool v;
                    cin >> v;
                    SetVerbose(v);
                    printf("Set verbose to %s\n", v ? "True" : "False");
                }

                // Printing prompt for another go
                printf("IC:");
                cin >> operation;
            }

        }
        
        // Private methods for merge split conditioner
        bool MergeSplitConditionerGPU::MarkMerges(bool force) {
            CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
            // Making sure the mergesToMake array is sized correctly
            thrust::constant_iterator<int> negOne(-1);
            mergesToMake.assign(negOne, negOne + inducedGraph->Size());
            
            // Choosing which sizes to use:
            IntVector_d *sizes = &partSizes;
            if (nodeWeights.size() > 0)
                sizes = &weightedSizes;

            // Declaring temp arrays
            int size = inducedGraph->Size();
            IntVector_d desiredMerges(size, -1);
            IntVector_d merging(size, 0);
            IntVector_d incomplete(1,1);

            // Figuring out block sizes for kernel call:
            int blockSize = 256;
            int nBlocks = size/blockSize + (size%blockSize == 0?0:1);

            while (incomplete[0] == 1)
            {
//                IntVector_h before(desiredMerges);
                incomplete[0] = 0;
                if (verbose)
                    printf("Calling FindDesirableMerges Kernel <<<%d, %d>>>\n",
                            blockSize, nBlocks);
                Kernels::FindDesirableMerges <<<blockSize, nBlocks>>>
                        (size, 
                        minSize, 
                        maxSize, 
                        force, 
                        inducedGraph->indStart(), 
                        inducedGraph->adjStart(), 
                        StartOf(sizes), 
                        StartOf(desiredMerges), 
                        StartOf(merging));
                if (verbose)
                    printf("Calling MarkMerges Kernel <<<%d, %d>>>\n",
                            blockSize, nBlocks);
                Kernels::MarkMerges <<<blockSize, nBlocks>>>
                        (size, 
                        StartOf(desiredMerges), 
                        StartOf(merging), 
                        StartOf(mergesToMake), 
                        StartOf(incomplete));
            }

            int marked = -1;
            // Cleaning up temp arrays
            try {
                marked = thrust::count_if(mergesToMake.begin(), mergesToMake.end(), Functors::NotNegOne());
                desiredMerges.clear();
                merging.clear();
                incomplete.clear();
            }
            catch (thrust::system::system_error &e)
            {
                printf("Caught exception at end of MarkMerges!\n");
                cerr << e.what() << endl;
                cerr << "Error code: " << e.code() << endl;
                InputHelpers::GetNonEmptyLineCIN();
            }
            
            if (verbose)
                printf("MarkMerges completing with %d merges marked.\n", marked);
            
            CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
            
            return marked > 0;            
        }
        bool MergeSplitConditionerGPU::MarkSplits(bool force) {
            CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
            int size = inducedGraph->Size();
            thrust::constant_iterator<int> zero(0);
            splitsToMake.assign(zero, zero + size);

            // Choosing which sizes to use:
            IntVector_d *sizes = &partSizes;
            if (nodeWeights.size() > 0)
                sizes = &weightedSizes;
            
            // Debug
            if (verbose)
                printf("MarkSplits called. InducedGraph Size: %d partSizes Size: %d\n", size, partSizes.size());
            
            // Figuring out block sizes for kernel call:
            int blockSize = 256;
            int nBlocks = size/blockSize + (size%blockSize == 0?0:1);

              // Debug
            if (verbose)
                printf("Calling MarkSplits kernel. blockSize: %d nBlocks: %d\n", size, blockSize, nBlocks);
            
            // Calling kernel to mark the needed splits
            if (verbose)
                    printf("Calling MarkSplits Kernel <<<%d, %d>>>\n",
                            blockSize, nBlocks);
            Kernels::MarkSplits <<<blockSize, nBlocks>>> 
                    (size, 
                    force, 
                    minSize, 
                    maxSize, 
                    StartOf(sizes), 
                    StartOf(splitsToMake));

            int marked = thrust::count(splitsToMake.begin(), splitsToMake.end(), 1);
            
            if (verbose)
                printf("MarkSplits completed with %d splits marked\n", marked);
            
            CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
            return marked > 0;
        }
        bool MergeSplitConditionerGPU::MarkMergeSplits(int desiredSize) {
            CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
            // Getting the size to use
            int size = inducedGraph->Size();
            
            // Choosing which sizes to use:
            IntVector_d *sizes = &partSizes;
            if (nodeWeights.size() > 0)
                sizes = &weightedSizes;
            
            // Making sure the mergesToMake array is sized correctly
            thrust::constant_iterator<int> negOne(-1);
            mergesToMake.assign(negOne, negOne + size);

            // Declaring temp arrays
            IntVector_d desiredMerges(size, -1);
            IntVector_d merging(size, 0);
            IntVector_d incomplete(1,1);

            // Figuring out block sizes for kernel call:
            int blockSize = 256;
            int nBlocks = size/blockSize + (size%blockSize == 0?0:1);

            while (incomplete[0] == 1)
            {
                incomplete[0] = 0;
                if (verbose)
                    printf("Calling FindDesirableMergeSplits Kernel <<<%d, %d>>>\n",
                            blockSize, nBlocks);
                Kernels::FindDesirableMergeSplits <<<blockSize, nBlocks>>>
                        (size, 
                        minSize, 
                        maxSize, 
                        desiredSize, 
                        inducedGraph->indStart(), 
                        inducedGraph->adjStart(), 
                        StartOf(sizes), 
                        StartOf(desiredMerges), 
                        StartOf(merging));
                if (verbose)
                    printf("Calling MarkMerges Kernel <<<%d, %d>>>\n",
                            blockSize, nBlocks);
                Kernels::MarkMerges <<<blockSize, nBlocks>>>
                        (size, 
                        StartOf(desiredMerges), 
                        StartOf(merging), 
                        StartOf(mergesToMake), 
                        StartOf(incomplete));
            }
            
            // Checking for marked merges
            int marked = thrust::count_if(mergesToMake.begin(), 
                                            mergesToMake.end(), 
                                            Functors::NotNegOne());
            // Cleaning up temp arrays
            desiredMerges.clear();
            merging.clear();
            incomplete.clear();
            
            CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
            
            return marked > 0;
        }
        void MergeSplitConditionerGPU::MakeSplits() {
            thrust::counting_iterator<int> zero(0);
            
            // Get list of all small aggregates to split 0-64
            IntVector_d smallSplits(splitsToMake.size());
            int smallSplitsCount = thrust::copy_if(zero, 
                                        zero + splitsToMake.size(),
                                        thrust::make_zip_iterator(
                                            thrust::make_tuple(
                                                splitsToMake.begin(), 
                                                partSizes.begin())),
                                        smallSplits.begin(),
                                        Functors::SplitFilter(0, 64))
                                    - smallSplits.begin();
            smallSplits.resize(smallSplitsCount);
            if (verbose)
                printf("Found %d small splits to do.\n", smallSplitsCount);
            
            // Get list of all big aggregates to split 65-256
            IntVector_d bigSplits(splitsToMake.size());
            int bigSplitsCount = thrust::copy_if(zero, 
                                        zero + splitsToMake.size(),
                                        thrust::make_zip_iterator(
                                            thrust::make_tuple(
                                                splitsToMake.begin(), 
                                                partSizes.begin())),
                                        bigSplits.begin(),
                                        Functors::SplitFilter(65, 256))
                                    - bigSplits.begin();
            bigSplits.resize(bigSplitsCount);
            splits += smallSplitsCount + bigSplitsCount;
//            if (verbose)
//                printf("Found %d big splits to do.\n", bigSplitsCount);
//            
//            int total = thrust::count(splitsToMake.begin(), splitsToMake.end(), 1);
//            
//            if (verbose)
//                printf("There should be %d total splits.\n", total);
//            
//            int d;
//            std::cin >> d;
            
            Graph_d* aggMap = Aggregation::GetAggregateMap(aggregation);
            
            // Making the splits is different for weighted/non-weighted
            if (nodeWeights.size() == 0) {
                // Do the small splits
                int offset = 0;
                while (smallSplitsCount - offset > 0) {
                    // Call either 64 blocks or all remaining
                    int toDo = smallSplitsCount - offset > 64 ? 
                                    64 : smallSplitsCount-offset;

                    if (verbose)
                        printf("Calling MakeSplits Kernel <<<%d, %d>>>\n",
                                toDo, 64);

                    Kernels::MakeSplits <<< toDo, 64 >>>
                            (aggMap->Size() + offset, 
                            StartOf(smallSplits) + offset, 
                            StartOf(aggregation), 
                            aggMap->indStart(), 
                            aggMap->adjStart(), 
                            graph->indStart(), 
                            graph->adjStart());
                    offset += toDo;
                }
                
                // Reset the offset and do the big splits
                offset = 0;
                while (bigSplitsCount - offset > 0) {
                    // Call either 64 blocks or all remaining
                    int toDo = bigSplitsCount - offset > 64 ? 
                                    64 : bigSplitsCount-offset;

                    if (verbose)
                        printf("Calling MakeSplits_Large Kernel <<<%d, %d>>>\n",
                                toDo, 256);
                    
                    Display::Print(bigSplits, "Big splits");
                    IntVector_h sizes(bigSplits.size());
                    for (int i = 0; i < bigSplits.size(); i++)
                        sizes[i] = partSizes[bigSplits[i]];
                    Display::Print(sizes, "Sizes of big splits");
                    Kernels::MakeSplits_Large <<< toDo, 256 >>>
                            (aggMap->Size() + smallSplitsCount + offset, 
                            StartOf(bigSplits) + offset, 
                            StartOf(aggregation), 
                            aggMap->indStart(), 
                            aggMap->adjStart(), 
                            graph->indStart(), 
                            graph->adjStart());
                    offset += toDo;
                }
                cudaDeviceSynchronize();

            }
            else {
                // Do the small splits
                int offset = 0;
                while (smallSplitsCount - offset > 0) {
                    // Call either 64 blocks or all remaining
                    int toDo = smallSplitsCount - offset > 64 ? 
                                    64 : smallSplitsCount-offset;

                    if (verbose)
                        printf("Calling MakeSplitsWeighted Kernel <<<%d, %d>>>\n",
                                toDo, 64);

                    Kernels::MakeSplitsWeighted <<< toDo, 64 >>>
                            (aggMap->Size() + offset, 
                            StartOf(smallSplits) + offset, 
                            StartOf(aggregation), 
                            aggMap->indStart(), 
                            aggMap->adjStart(), 
                            graph->indStart(), 
                            graph->adjStart(),
                            StartOf(nodeWeights));
                    offset += toDo;
                }
                
                // Reset the offset and do the big splits
                offset = 0;
                while (bigSplitsCount - offset > 0) {
                    // Call either 64 blocks or all remaining
                    int toDo = bigSplitsCount - offset > 64 ? 
                                    64 : bigSplitsCount-offset;

                    if (verbose)
                        printf("Calling MakeSplitsWeighted_Large Kernel <<<%d, %d>>>\n",
                                toDo, 256);

                    Kernels::MakeSplitsWeighted_Large <<< toDo, 256 >>>
                            (aggMap->Size() + smallSplitsCount + offset, 
                            StartOf(bigSplits) + offset, 
                            StartOf(aggregation), 
                            aggMap->indStart(), 
                            aggMap->adjStart(), 
                            graph->indStart(), 
                            graph->adjStart(),
                            StartOf(nodeWeights));
                    offset += toDo;
                }
                cudaDeviceSynchronize();
            }

            // Reset induced graph after changes
            delete inducedGraph;
            inducedGraph = GraphHelpers::GetInducedGraph(*graph, aggregation);
            
            // Getting the new part sizes:
            GraphHelpers::getPartSizes(aggregation, partSizes);
            if (nodeWeights.size() > 0)
                GraphHelpers::getPartSizes(aggregation, weightedSizes, nodeWeights);
            
//            IntVector_d afterSizes(bigSplits.size());
//            for (int i = 0; i < bigSplits.size(); i++)
//                    afterSizes[i] = partSizes[bigSplits[i]];
//            Display::Print(afterSizes, "Sizes of big splits after splitting");
//            
//            IntVector_d afterSizes2(bigSplits.size());
//            for (int i = 0; i < bigSplits.size(); i++)
//                    afterSizes2[i] = partSizes[i + aggMap->Size() + smallSplitsCount];
//            Display::Print(afterSizes, "Sizes of big splits after splitting 2");
            
            // Clean up
            smallSplits.clear();
            bigSplits.clear();
            delete aggMap;
        }
        void MergeSplitConditionerGPU::MakeMerges(bool markSplits) {
            CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
            if (verbose)
                printf("Starting MakeMerges. MarkSplits: %s InducedGraph Size: %d\n", markSplits ? "True" : "False", (inducedGraph->Size()));
            
            // Perform a prefix sum using a transform iterator which 
            // marks aggregates which are merging to mark the index 
            // offset for each aggregate:
            mergeOffsets.resize(mergesToMake.size());
            thrust::inclusive_scan(
                thrust::make_transform_iterator(
                    mergesToMake.begin(), 
                    Functors::NotNegOne()),
                thrust::make_transform_iterator(
                    mergesToMake.end(), 
                    Functors::NotNegOne()),
                mergeOffsets.begin());

            // If this is part of the merge split routine prepare to 
            // mark splits
            if (markSplits)
            {
                int lastOffset = mergeOffsets[mergeOffsets.size()-1];
                int newLength = mergeOffsets.size() - lastOffset;
                thrust::constant_iterator<int> zero(0);
                splitsToMake.assign(zero, zero + newLength);
            }

            // Figuring out block sizes for kernel call:
            int size = aggregation.size();
            int blockSize = 256;
            int nBlocks = size/blockSize + (size%blockSize == 0?0:1);

            // Calling the kernel 
            if (markSplits) {
                if (verbose) 
                    printf("Calling MakeMerges_MarkSplits Kernel <<<%d, %d>>>\n",
                            nBlocks, blockSize);
                Kernels::MakeMerges_MarkSplits <<<nBlocks, blockSize>>>
                        (size, 
                        StartOf(mergesToMake), 
                        StartOf(mergeOffsets), 
                        StartOf(aggregation), 
                        StartOf(splitsToMake));
                CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
            }
            else {
                if (verbose) 
                    printf("Calling MakeMerges Kernel <<<%d, %d>>>\n",
                            nBlocks, blockSize);
                Kernels::MakeMerges <<<nBlocks, blockSize>>> 
                        (size, 
                        StartOf(mergesToMake), 
                        StartOf(mergeOffsets), 
                        StartOf(aggregation));
                if (CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__)) {
                    
                }
            }

            // Re-figuring the aggregate adjacency graph (only if not mergeSplitting)
            if(!markSplits)
            {
                merges += mergeOffsets[mergeOffsets.size() - 1];
                delete inducedGraph;
                inducedGraph = GraphHelpers::GetInducedGraph(*graph, aggregation);
                // Getting the new part sizes:
                GraphHelpers::getPartSizes(aggregation, partSizes);
                if (nodeWeights.size() > 0)
                    GraphHelpers::getPartSizes(aggregation, weightedSizes, nodeWeights);
            }
            else
                mergeSplits += mergeOffsets[mergeOffsets.size() - 1];
            if (verbose)
                printf("Finished MakeMerges. InducedGraph Size: %d\n", (inducedGraph->Size()));
            CheckCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
        }
        bool MergeSplitConditionerGPU::MakeMergeSplits(int desiredSize) {
            if (MarkMergeSplits(desiredSize)) {
                MakeMerges(false);
                if (MarkSplits(false))
                    MakeSplits();
                return true;
            }
            return false;
        }
    }
}
