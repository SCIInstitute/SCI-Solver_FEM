/* 
 * File:   AggMIS_GraphHelpers.h
 * Author: T. James Lewis
 *
 * Created on April 16, 2013, 2:58 PM
 */
#ifndef AGGMIS_GRAPHHELPERS_H
#define	AGGMIS_GRAPHHELPERS_H
#include <AggMIS_Types.h>
#include <thrust/scan.h>
#include <thrust/adjacent_difference.h>
namespace AggMIS {
    namespace GraphHelpers {
        
        // GPU Kernels
        namespace Kernels {
            __global__ void mapAdjacencyToBlockKernel(int size, 
                                            int *adjIndexes, 
                                            int *adjacency, 
                                            int *adjacencyBlockLabel, 
                                            int *blockMappedAdjacency, 
                                            int *fineAggregate);
            __global__ void findPartIndicesNegStartKernel(int size, 
                                            int *array, 
                                            int *partIndices);
            __global__ void findPartIndicesKernel(int size, 
                                            int *array, 
                                            int *partIndices);
            __global__ void findAdjacencySizesKernel(int size, 
                                            int *adjIndexes, 
                                            int *output);
            __global__ void accumulatedPartSizesKernel(int size, 
                                            int *part, 
                                            int *weights, 
                                            int *accumulatedSize);
            __global__ void unaccumulatedPartSizesKernel(int size, 
                                            int *accumulatedSize, 
                                            int *sizes);            
        }
        Types::Graph_d* GetInducedGraph(Types::Graph_d &graph, 
                            Types::IntVector_d &aggregation);
        Types::Graph_h* GetInducedGraph(Types::Graph_h &graph,
                            Types::IntVector_h &aggregation);
        void mapAdjacencyToBlock(Types::IntVector_d &adjIndexes, 
                            Types::IntVector_d &adjacency, 
                            Types::IntVector_d &adjacencyBlockLabel, 
                            Types::IntVector_d &blockMappedAdjacency, 
                            Types::IntVector_d &fineAggregate);
        void getPartIndicesNegStart(Types::IntVector_d& sortedPartition, 
                            Types::IntVector_d& partIndices);
        /**
         * Gets a vector where the values are the indices of the elements
         * @param size Size of vector to create
         * @return A pointer to newly created vector
         */
        Types::IntVector_d* GetIndicesVector(int size);
        /**
         * Writes the index of each vector element as its value
         * @param size The size the vector should be
         * @param toSet The vector to set (Overwritten)
         */
        void SetToIndicesVector(int size, 
                            Types::IntVector_d& toSet);
        /**
         * Gets the size (count of nodes) of each aggregate. 
         * @param aggregation Labels each node with its aggregate ID
         * @param sizes Vector to output computed sized (Overwritten)
         */
        void getPartSizes(Types::IntVector_d &aggregation, 
                            Types::IntVector_d &sizes);
        /**
         * Gets the size of each aggregate, taking into account the weight of 
         * each node.
         * @param aggregation Labels each node with its aggregate ID
         * @param sizes Vector to put the computed sizes into (Overwritten)
         * @param weights The weights of each graph node
         */
        void getPartSizes(Types::IntVector_d &aggregation, 
                            Types::IntVector_d &sizes, 
                            Types::IntVector_d &weights);
        /**
         * Finds the valence of each node in the given graph.
         * @param graph Input graph
         * @return A vector containing the valence of each node
         */
        Types::IntVector_d* GetValences(Types::Graph_d &graph);
        Types::IntVector_h* GetValences(Types::Graph_h &graph);
        /**
         * Checks if a graph is a valid undirected graph. Valid being that each 
         * node listing a node as neighbor is a neighbor of the listed node, and
         * that all listed neighbors are valid graph nodes.
         * @param graph The graph to check
         * @return True if graph is valid, false otherwise
         */
        bool IsGraphValid(Types::Graph_d &graph);
        /**
         * Checks if a graph is a valid undirected graph. Valid being that each 
         * node listing a node as neighbor is a neighbor of the listed node, and
         * that all listed neighbors are valid graph nodes.
         * @param graph The graph to check
         * @return True if graph is valid, false otherwise
         */
        bool IsGraphValid(Types::Graph_h &graph);
        /**
         * Checks if a graph is a valid undirected graph. Valid being that each 
         * node listing a node as neighbor is a neighbor of the listed node, and
         * that all listed neighbors are valid graph nodes.
         * @param indices The vector of indices into adjacency list
         * @param adjacency The adjacency list
         * @return True if graph is valid, false otherwise
         */
        bool IsGraphValid(Types::IntVector_d &indices, Types::IntVector_d &adjacency);
        /**
         * Checks if a graph is a valid undirected graph. Valid being that each 
         * node listing a node as neighbor is a neighbor of the listed node, and
         * that all listed neighbors are valid graph nodes.
         * @param indices The vector of indices into adjacency list
         * @param adjacency The adjacency list
         * @return True if graph is valid, false otherwise
         */
        bool IsGraphValid(Types::IntVector_h &indices, Types::IntVector_h &adjacency);
    }
}
#endif	/* AGGMIS_GRAPHHELPERS_H */

