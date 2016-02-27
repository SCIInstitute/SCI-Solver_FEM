/* 
 * File:   AggMIS_GraphHelpers.cu
 * Author: T. James Lewis
 *
 * Created on April 16, 2013, 2:58 PM
 */
 #include <algorithm>
#include "AggMIS_GraphHelpers.h"
namespace AggMIS {
    namespace GraphHelpers {
        namespace Kernels {
            __global__ void mapAdjacencyToBlockKernel(int size, 
                                            int *adjIndexes, 
                                            int *adjacency, 
                                            int *adjacencyBlockLabel, 
                                            int *blockMappedAdjacency, 
                                            int *fineAggregate) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size)
                {
                    int begin = adjIndexes[idx];
                    int end = adjIndexes[idx + 1];
                    int thisBlock = fineAggregate[idx];

                    // Fill block labeled adjacency and block mapped adjacency vectors
                    for (int i = begin; i < end; i++)
                    {
                        int neighbor = fineAggregate[adjacency[i]];

                        if (thisBlock == neighbor)
                        {
                            adjacencyBlockLabel[i] = - 1;
                            blockMappedAdjacency[i] = - 1;                
                        }
                        else
                        {
                            adjacencyBlockLabel[i] = thisBlock;
                            blockMappedAdjacency[i] = neighbor;
                        }
                    }
                }
            }
            __global__ void findPartIndicesNegStartKernel(int size, 
                                            int *array, 
                                            int *partIndices) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
                if (idx < size)
                {
                    int value = array[idx];
                    int nextValue = array[idx + 1];
                    if (value != nextValue)
                        partIndices[value + 1] = idx;
                }    
            }
            __global__ void findPartIndicesKernel(int size, 
                                            int *array, 
                                            int *partIndices) {
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
            __global__ void findAdjacencySizesKernel(int size, 
                                            int *adjIndexes, 
                                            int *output) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size)
                {
                    output[idx] = adjIndexes[idx + 1] - adjIndexes[idx];
                }
            }
            __global__ void accumulatedPartSizesKernel(int size, 
                                            int *part, 
                                            int *weights, 
                                            int *accumulatedSize) {
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

            __global__ void unaccumulatedPartSizesKernel(int size, int *accumulatedSize, int *sizes) {
              int idx = blockIdx.x * blockDim.x + threadIdx.x;
              if(idx == 0)
                sizes[idx] = accumulatedSize[0];

              else if(idx < size)
              {
                sizes[idx] = accumulatedSize[idx] - accumulatedSize[idx - 1];
              }
            }
            
        }
        Graph_d* GetInducedGraph(Graph_d &graph,
                                IntVector_d &aggregation) {
            // Get references to parts of input graph and output graph
            IntVector_d &adjIndexesIn = *(graph.indices);
            IntVector_d &adjacencyIn = *(graph.adjacency);
            Graph_d *result = new Graph_d();
            IntVector_d &adjIndexesOut = *(result->indices);
            IntVector_d &adjacencyOut = *(result->adjacency);
            
            // Declaring temporary vectors:
            IntVector_d adjacencyBlockLabel, blockMappedAdjacency;
            adjacencyBlockLabel.resize(adjacencyIn.size(),0);
            blockMappedAdjacency.resize(adjacencyIn.size(),0);

            // Get the blocklabeled adjacency:
            mapAdjacencyToBlock(adjIndexesIn, adjacencyIn, adjacencyBlockLabel, blockMappedAdjacency, aggregation);

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
            getPartIndicesNegStart(adjacencyBlockLabel, adjIndexesOut);
            adjacencyOut.resize(blockMappedAdjacency.size() - 1);
            thrust::copy(blockMappedAdjacency.begin() + 1, blockMappedAdjacency.end(), adjacencyOut.begin());
            return result;
        }
        Graph_h* GetInducedGraph(Graph_h& graph, 
                                IntVector_h& aggregation) {
            // Get references to graph indices and adjacency
            IntVector_h &ind = *(graph.indices);
            IntVector_h &adj = *(graph.adjacency);
            
            // A temporary container for the induced graph
            vector<vector<int> > tempGraph;
            
            // Filling in the temporary container
            for (int node = 0; node < graph.Size(); node++) {
                int startAgg = aggregation[node];
                for (int nIt = ind[node]; nIt < ind[node + 1]; nIt++) {
                    int endAgg = aggregation[adj[nIt]];
                    
                    // If this is an edge between two aggregates add to
                    // the induced graph.
                    if (startAgg != endAgg && startAgg < endAgg) {
                        // Making sure that there are entries in temp
                        if (endAgg >= tempGraph.size())
                            tempGraph.resize(endAgg + 1);
                        
                        // Adding edge entries
                        if (tempGraph[startAgg].size() == 0 || 
                                !(std::binary_search(tempGraph[startAgg].begin(), 
                                                    tempGraph[startAgg].end(), 
                                                    endAgg))) {
                            tempGraph[startAgg].push_back(endAgg);
                            std::sort(tempGraph[startAgg].begin(), tempGraph[startAgg].end());
                        }
                        if (tempGraph[endAgg].size() == 0 || 
                                !(std::binary_search(tempGraph[endAgg].begin(), 
                                                    tempGraph[endAgg].end(), 
                                                    startAgg))) {
                            tempGraph[endAgg].push_back(startAgg);
                            std::sort(tempGraph[endAgg].begin(), tempGraph[endAgg].end());
                        }
                    } 
                }
            }
            
            // Copying out to graph format
            Graph_h *result = new Graph_h();
            IntVector_h &indOut = *(result->indices);
            IntVector_h &adjOut = *(result->adjacency);
            
            // Getting new indices
            indOut.resize(tempGraph.size() + 1);
            indOut[0] = 0;
            for (int i = 0; i < tempGraph.size(); i++) 
                indOut[i + 1] = indOut[i] + tempGraph[i].size();
            
            // Writing out adjacency
            adjOut.resize(indOut.back());
            int insertAt = 0;
            for (int i = 0; i < tempGraph.size(); i++) {
                for (int j = 0; j < tempGraph[i].size(); j++) {
                    adjOut[insertAt++] = tempGraph[i][j];
                }
            }
            
            return result;
        }
        void mapAdjacencyToBlock(IntVector_d &adjIndexes, 
                                IntVector_d &adjacency, 
                                IntVector_d &adjacencyBlockLabel, 
                                IntVector_d &blockMappedAdjacency, 
                                IntVector_d &fineAggregate) {
            int size = adjIndexes.size() - 1;

            // Get pointers to device memory of arrays
            int *adjIndexes_d = thrust::raw_pointer_cast(&adjIndexes[0]);
            int *adjacency_d = thrust::raw_pointer_cast(&adjacency[0]);
            int *adjacencyBlockLabel_d = thrust::raw_pointer_cast(&adjacencyBlockLabel[0]);
            int *blockMappedAdjacency_d = thrust::raw_pointer_cast(&blockMappedAdjacency[0]);
            int *fineAggregate_d = thrust::raw_pointer_cast(&fineAggregate[0]);

            // Figuring out block sizes for kernel call:
            int blockSize = 256;
            int nBlocks = size / blockSize + (size % blockSize == 0?0:1);

            // Calling kernel:
            Kernels::mapAdjacencyToBlockKernel <<< nBlocks, blockSize >>> (size, adjIndexes_d, adjacency_d, adjacencyBlockLabel_d, blockMappedAdjacency_d, fineAggregate_d);   
        }
        void getPartIndicesNegStart(IntVector_d& sortedPartition, 
                                IntVector_d& partIndices) {
            // Sizing the array:
            int maxPart = max(0,sortedPartition[sortedPartition.size() - 1]);
            partIndices.resize(maxPart + 2, 0);

            // Figuring out block sizes for kernel call:
            int size = sortedPartition.size();
            int blockSize = 256;
            int nBlocks = size/blockSize + (size%blockSize == 0?0:1);

            // Getting pointers
            int *sortedPartition_d = thrust::raw_pointer_cast(&sortedPartition[0]);
            int *partIndices_d = thrust::raw_pointer_cast(&partIndices[0]);

            // Calling kernel to find indices for each part:
            Kernels::findPartIndicesNegStartKernel <<< nBlocks, blockSize >>> (size, sortedPartition_d, partIndices_d);
            partIndices[partIndices.size() - 1] = size - 1;
        }
        IntVector_d* GetIndicesVector(int size) {
            thrust::counting_iterator<int> start(0);
            thrust::counting_iterator<int> end = start + size;
            return new IntVector_d(start, end);
        }
        void SetToIndicesVector(int size, 
                            IntVector_d& toSet) {
            thrust::counting_iterator<int> start(0);
            thrust::counting_iterator<int> end = start + size;
            toSet.assign(start, end);
        }
        void getPartSizes(IntVector_d &aggregation, 
                            IntVector_d &sizes) {
            // Make a copy of the partition vector to mess with:
            IntVector_d temp(aggregation);

            // Sort the copy and find largest element
            thrust::sort(temp.begin(), temp.end());
            int maxPart = temp[temp.size() - 1];

            // Creating a new array size
            IntVector_d partIndices(maxPart + 2, 0);

            // Figuring out block sizes for kernel call:
            int size = aggregation.size();
            int blockSize = 256;
            int nBlocks = size/blockSize + (size%blockSize == 0?0:1);

            // Getting pointers
            int *temp_d = thrust::raw_pointer_cast(&temp[0]);
            int *partIndices_d = thrust::raw_pointer_cast(&partIndices[0]);  

            // Calling kernel to find indices for each part:
            Kernels::findPartIndicesKernel <<< nBlocks, blockSize >>> (size, temp_d, partIndices_d);    

            // Preparing sizes vector
            size = partIndices.size() - 1;
            sizes.resize(size);
            int *sizes_d = thrust::raw_pointer_cast(&sizes[0]);

            // Calling kernel to find sizes:
            Kernels::findAdjacencySizesKernel <<< nBlocks, blockSize >>> (size, partIndices_d, sizes_d);

            // Cleaning up
            temp.resize(0);
            partIndices.resize(0);
        }
        void getPartSizes(IntVector_d& aggregation, 
                            IntVector_d& sizes, 
                            IntVector_d& weights) {
              // Make copies to mess with
              IntVector_d tempAgg(aggregation.begin(), aggregation.end());
              IntVector_d tempWeight(weights.begin(), weights.end());

              // Sorting temp vectors together
              thrust::sort_by_key(tempAgg.begin(), tempAgg.end(), tempWeight.begin());

              // Getting prefix sum of values
              thrust::inclusive_scan(tempWeight.begin(), tempWeight.end(), tempWeight.begin());

              // Another temp vector for accumulated size at last nodes
              IntVector_d accumulatedSize(tempAgg[tempAgg.size() - 1] + 1);

              // Preparing to call kernel to fill accumulated size vector
              int size = tempAgg.size();

              // Figuring out block sizes for kernel call:
              int blockSize = 256;
              int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

              Kernels::accumulatedPartSizesKernel <<< nBlocks, blockSize >>> 
                      (size, 
                      StartOf(tempAgg), 
                      StartOf(tempWeight), 
                      StartOf(accumulatedSize));

              // Calling kernel to get the un-accumulated part sizes:
              size = accumulatedSize.size();
              nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);
              sizes.resize(size);
              Kernels::unaccumulatedPartSizesKernel <<< nBlocks, blockSize >>> 
                      (size, 
                      StartOf(accumulatedSize), 
                      StartOf(sizes));
        }
        IntVector_d* GetValences(Graph_d &graph) {
            IntVector_d *result = new IntVector_d(graph.indices->size()-1);
            thrust::adjacent_difference(graph.indices->begin() + 1, 
                                        graph.indices->end(), 
                                        result->begin());
            return result;
        }
        IntVector_h* GetValences(Graph_h &graph) {
            IntVector_h *result = new IntVector_h(graph.indices->size()-1);
            thrust::adjacent_difference(graph.indices->begin() + 1, 
                                        graph.indices->end(), 
                                        result->begin());
            return result;
        }
        bool IsGraphValid(Graph_d& graph) {
            // Call the override with the indices and adjacency of the graph
            return IsGraphValid(*graph.indices, *graph.adjacency);
        }
        bool IsGraphValid(Graph_h& graph) {
            // Call the override with the indices and adjacency of the graph
            return IsGraphValid(*graph.indices, *graph.adjacency);
        }
        bool IsGraphValid(IntVector_d& indices, IntVector_d& adjacency) {
            // Get temporary host vectors to call with
            IntVector_h tempInd(indices);
            IntVector_h tempAdj(adjacency);
            
            // Call host vector override
            bool result = IsGraphValid(tempInd, tempAdj);
            
            // Clean up temp arrays
            tempInd.clear();
            tempAdj.clear();
            
            return result;
        }
        bool IsGraphValid(IntVector_h& indices, IntVector_h& adjacency) {
            // Get size of graph
            int graphSize = indices.size() - 1;
            
            // If there are no nodes return with error
            if (graphSize <= 0) {
                printf("Graph is empty, no nodes specified!\n");
                return false;
            }
            
            // Check that the indices are all in order 
            if (indices[0] != 0) {
                int first = indices[0];
                printf("Indices are not proper, start with %d not 0\n", first);
                return false;
            }
            for (int i = 1; i < indices.size(); i++) {
                if (indices[i] <= indices[i - 1]) {
                    int a = indices[i - 1];
                    int b = indices[i];
                    printf("Non-sequential indices: indices[%d]=%d but indices[%d]=%d\n", 
                            i-1, a, i, b);
                    return false;
                }
            }
            if (indices[indices.size()-1] > adjacency.size()) {
                printf("Largest index points outside of adjacency array!\n");
                return false;
            }
            
            // Check that adjacency contains only valid node Id's
            for (int i = 0; i < adjacency.size(); i++) {
                int nodeId = adjacency[i];
                if (nodeId < 0 || nodeId >= graphSize) {
                    printf("adjacency[%d]=%d but graphSize=%d\n", 
                            i, nodeId, graphSize);
                    return false;
                }
            }
            
            // Check that all neighbor lists are mutually consistent
            int errorCount = 0;
            for (int i = 0; i < graphSize; i++)
            {
                int rootIdx = i;
                for (int j = indices[i]; j < indices[i + 1]; j++)
                {
                    int neighborIdx = adjacency[j];
                    bool found = false;
                    for (int jj = indices[neighborIdx]; jj < indices[neighborIdx + 1]; jj++)
                    {
                        if (adjacency[jj] == rootIdx)
                            found = true;
                    }
                    if (!found) {
                        printf("Node %d has neighbor %d but not reverse!\n", rootIdx, neighborIdx);
                        errorCount++;
                    }
                }
            }
            if (errorCount > 0) {
                printf("Found %d inconsistencies in adjacency.\n", errorCount);
                return false;
            }
            
            // If we haven't returned yet things are good.
            return true;
        }
    }
}