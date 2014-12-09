/* 
 * File:   AggMIS_Aggregation_CPU.h
 * Author: T. James Lewis
 *
 * Created on July 3, 2013, 4:21 PM
 */

#ifndef AGGMIS_AGGREGATION_CPU_H
#define	AGGMIS_AGGREGATION_CPU_H
#include <AggMIS_Types.h>
#include <Helper.h>
#include <queue>
#include <set>
#include <vector>

namespace AggMIS {
    namespace Aggregation {
        using namespace Types;
        using namespace std;
        /**
         * This method returns an aggregation where each node in the graph is 
         * assigned to the nearest root point.
         * @param graph The graph being aggregated
         * @param roots The root points for the aggregation
         * @return An array where return[i] is the ID of the aggregate to which
         * node i is assigned.
         */
        IntVector_h* AggregateToNearest(Graph_h &graph,
                                IntVector_h &roots);
        /**
         * This method checks if the given aggregation is valid.
         * @param graph The graph that was aggregated.
         * @param aggregation The aggregation array to check
         * @param verbose If true, more output will be printed.
         * @return True if aggregation is valid, False otherwise.
         */
        bool IsValidAggregation(Graph_h &graph,
                                IntVector_h &aggregation,
                                bool verbose);
        /**
         * This method returns an array where each element is the count of 
         * nodes in the corresponding aggregate.
         * @param aggregation The aggregation array.
         * @return An array giving the size of each aggregate.
         */
        IntVector_h* GetPartSizes(IntVector_h &aggregation);
        /**
         * This method returns an array where each element is the sum of the 
         * weights of all nodes in the corresponding aggregate.
         * @param aggregation The aggregation array.
         * @param nodeWeights The weights of each node.
         * @return An array giving the total weight of each aggregate.
         */
        IntVector_h* GetPartSizes(IntVector_h &aggregation, 
                                IntVector_h &nodeWeights);
        /**
         * This method returns a sub-graph of the input graph with only nodes
         * contained in nodeList, and only edges between nodes both contained
         * in nodeList.
         * @param graph The graph to get sub-graph of.
         * @param nodeList The list of nodes defining the subgraph
         * @return A graph where return[i][j] is the ID of the j'th neighbor
         * of the i'th node of the graph.
         */
        vector<vector<int> >* GetAggregateGraph(Graph_h &graph, 
                                vector<int> &nodeList);
        /**
         * Returns a node in the given graph with maximal path distance from the
         * specified start node.
         * @param graph The graph.
         * @param start The start node.
         * @return The ID of a node such that no other node has a higher 
         * distance to the start node.
         */
        int FindFarthestNode(vector<vector<int> > &graph, 
                                int start);
        /**
         * Marks the distance of all nodes in the given graph from the start
         * point. After completion distances[i] will contain the distance from
         * node i to the startPoint.
         * @param graph The input graph.
         * @param distances The array to mark distances in.
         * @param startPoint The starting point.
         */
        void MarkDistances(vector<vector<int> > &graph, 
                                vector<int> &distances,
                                int startPoint);
        /**
         * Marks the distance of all nodes in the given graph from the set of
         * nodes specified in startPoints. After completion distances[i] will 
         * contain the distance from node i to the nearest node in startPoints.
         * @param graph
         * @param distances
         * @param startPoints
         */
        void MarkDistances(vector<vector<int> > &graph, 
                                vector<int> &distances,
                                vector<int> startPoints);
        /**
         * Returns the set of nodes in the given graph for which the sum of all 
         * distances from them to all other nodes is minimal.
         * @param graph The input graph.
         * @return The set of nodes in the given graph for which the sum of all
         * distances from them to all other nodes is minimal.
         */
        int FindMassScore(vector<vector<int> > &graph,
                                int startPoint);
        vector<int>* GetCentroid(vector<vector<int> > &graph, 
                                int startPoint);
    }
}

#endif	/* AGGMIS_AGGREGATION_CPU_H */

