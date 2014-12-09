#include <AggMIS_Aggregation_CPU.h>
namespace AggMIS {
    namespace Aggregation {
        IntVector_h* AggregateToNearest(Graph_h &graph,
                                IntVector_h &roots) {
            // Allocating an array for distances:
            IntVector_h rootDistance(roots.size());
            
            // Allocating return array
            IntVector_h *aggregation = new IntVector_h(roots.size());
            
            // A queue of unallocated nodes
            queue<int> toAllocate;
            
            // Assigning initial distances, numbering aggregates, and adding
            // nodes to queue for allocation.
            int nextAggId = 0;
            for (int i = 0; i < rootDistance.size(); i++) {
                // If node is root assign Id and add neighbors to queue
                if (roots[i] == 1) {
                    rootDistance[i] = 0;
                    (*aggregation)[i] = nextAggId++;
                    
                    // Adding neighbors to queue to handle
                    int start = (*(graph.indices))[i];
                    int end = (*(graph.indices))[i + 1];
                    for (int nIt = start; nIt < end; nIt++)
                        toAllocate.push((*(graph.adjacency))[nIt]);
                }
                // If node is not root mark as unassigned
                else {
                    rootDistance[i] = -1;
                    (*aggregation)[i] = -1;
                }
            }

            // Handling unallocated nodes in the queue:
            while (!toAllocate.empty())
            {
                // Pull node off queue
                int node = toAllocate.front();
                toAllocate.pop();

                // Check if already handled
                if ((*aggregation)[node] != -1 && rootDistance[node] != -1)
                    continue;

                // Check it's neighbors to find where to allocate
                int newAgg = -1;
                int bestDistance = -1;
                int start = (*(graph.indices))[node];
                int end = (*(graph.indices))[node + 1];
                for (int nIt = start; nIt < end; nIt++)
                {
                    int neighbor = (*(graph.adjacency))[nIt];
                    int neighborDist = rootDistance[neighbor];

                    // We only care about non-negative distances
                    if (neighborDist >= 0)
                    {
                        int neighborAgg = (*aggregation)[neighbor];

                        // If this is the first real distance seen take it
                        if (bestDistance == -1)
                        {
                            bestDistance = neighborDist;
                            newAgg = (*aggregation)[neighbor];
                        }

                        // If this distance ties break tie with root id
//                        else if (neighborDist == bestDistance && rootPoints[neighborAgg] > rootPoints[newAgg])
//                            newAgg = aggregation[neighbor];

                        // If this distance is better take it
                        else if (neighborDist < bestDistance)
                        {
                            newAgg = (*aggregation)[neighbor];
                            bestDistance = neighborDist;
                        }
                    }
                    // If the neighbor is unallocated add to queue
                    else
                        toAllocate.push(neighbor);
                }

                // Set aggregate of current node:
                (*aggregation)[node] = newAgg;
                rootDistance[node] = bestDistance + 1;
            }
            
            // Clean up temp vector
            rootDistance.clear();
            
            return aggregation;
        }
        bool IsValidAggregation(Graph_h &graph,
                                IntVector_h &aggregation,
                                bool verbose) {
            int errorsFound = 0;
            IntVector_h* ps = GetPartSizes(aggregation);
            IntVector_h &partSizes = *ps;
            IntVector_h visitedNodes(graph.Size(), 0);
            IntVector_h exploredAggregates(partSizes.size(), 0);
            set<int> problemAggregates;
            queue<int> toExplore;
            for (int i = 0; i < graph.Size(); i++)
            {
                int thisAggregate = aggregation[i];
                if (exploredAggregates[thisAggregate] == 0)
                {
                    // Explore the aggregate starting from this node
                    toExplore.push(i);
                    while(!toExplore.empty()) {
                        int node = toExplore.front();
                        toExplore.pop();
                        if (visitedNodes[node] == 0) {
                            visitedNodes[node] = 1;
                            int start = (*(graph.indices))[node];
                            int end = (*(graph.indices))[node + 1];
                            for (int nIt = start; nIt < end; nIt++) {
                                int neighbor = (*(graph.adjacency))[nIt];
                                if (aggregation[neighbor] == thisAggregate)
                                    toExplore.push(neighbor);
                            }
                        }
                    }
                    exploredAggregates[thisAggregate] = 1;        
                }
                else if (visitedNodes[i] == 0)
                {
                    // This node is not connected to others in the same aggregate
                    if (verbose)
                        printf("Node %d in aggregate %d was not visited but aggregate %d was explored!\n", i, thisAggregate, thisAggregate);
                    problemAggregates.insert(thisAggregate);
                    errorsFound++;
                }
            }
            if (errorsFound > 0)
            {
                printf("Found %d errors while checking aggregation!\n", errorsFound);
                set<int>::iterator it;
                for (it = problemAggregates.begin(); it != problemAggregates.end(); it++)
                    printf("\t%d", *it);
                printf("\n");
                return false;
            }
            return true;            
        }
        IntVector_h* GetPartSizes(IntVector_h &aggregation) {
            // Allocate return array
            IntVector_h *partSizes = new IntVector_h();
            
            // Iterate over the aggregation and count nodes
            for (int i = 0; i < aggregation.size(); i++) {
                int part = aggregation[i];
                if (part >= partSizes->size())
                    partSizes->resize(part + 1, 0);
                (*partSizes)[part]++;
            }
            return partSizes;
        }
        IntVector_h* GetPartSizes(IntVector_h &aggregation, 
                                IntVector_h &nodeWeights) {
            // Allocate return array
            IntVector_h *partSizes = new IntVector_h();
            
            // Iterate over the aggregation and count nodes
            for (int i = 0; i < aggregation.size(); i++) {
                int part = aggregation[i];
                if (part >= partSizes->size())
                    partSizes->resize(part + 1, 0);
                (*partSizes)[part] += nodeWeights[i];
            }
            return partSizes;
        }
        vector<vector<int> >* GetAggregateGraph(Graph_h& graph, 
                                vector<int> &nodeList) {
            // Create the return structure.
            vector<vector<int> > *aggGraph = new vector<vector<int> >(nodeList.size());
            
            // Fill the adjacency by translating the adjacency to local indices
            for (int nIt = 0; nIt < nodeList.size(); nIt++) {
                for (int* n = graph.nStart(nodeList[nIt]); 
                        n != graph.nEnd(nodeList[nIt]);
                        n++) {
                    // Trying to find the neighbor in aggregate's nodeList
                    int localId = Helper::BinarySearch(*n, &nodeList[0], nodeList.size());
                    
                    // If found add ID to neighbors
                    if (localId != -1) 
                        (*aggGraph)[nIt].push_back(localId);
                }
            }
            return aggGraph;
        }
        int FindFarthestNode(vector<vector<int> > &graph, 
                                int start) {
            // Data structures for flood fill
            vector<int> distances(graph.size(), -1);
            queue<int> toExplore;
            toExplore.push(start);
            distances[start] = 0;
            
            int farthestNode = start;
            int maxDistance = 0;

            while (!toExplore.empty())
            {
                // Getting next node off of queue
                int explorer = toExplore.front();
                toExplore.pop();

                int distance = distances[explorer] + 1;

                // Checking the neighbors to see if they need to go on the queue       
                for (int nIt = 0; nIt < graph[explorer].size(); nIt++)
                {
                    int neighbor = graph[explorer][nIt];
                    if (distances[neighbor] == -1)
                    {
                        if (distance > maxDistance)
                        {
                            farthestNode = neighbor;
                            maxDistance = distance;
                        }
                        distances[neighbor] = distance;
                        toExplore.push(neighbor);
                    }
                }
            }
            return farthestNode;
        }
        void MarkDistances(vector<vector<int> >& graph, 
                                vector<int>& distances, 
                                int startPoint) {
            // Put single start point into vector and call vector version.
            vector<int> startPoints;
            startPoints.push_back(startPoint);
            MarkDistances(graph, distances, startPoints);
        }
        void MarkDistances(vector<vector<int> >& graph, 
                                vector<int>& distances, 
                                vector<int> startPoints) {
            // Initialize data structures for flood fill
            distances.assign(graph.size(), -1);
            queue<int> toExplore;
            
            // Handle start points
            for (int i = 0; i < startPoints.size(); i++) {
                toExplore.push(startPoints[i]);
                distances[startPoints[i]] = 0;
            }

            // Explore the rest of the graph
            while (!toExplore.empty())
            {
                // Getting next node off of queue
                int explorer = toExplore.front();
                toExplore.pop();

                // Checking the neighbors to see if they need to go on the queue       
                for (int nIt = 0; nIt < graph[explorer].size(); nIt++)
                {
                    int neighbor = graph[explorer][nIt];
                    if (distances[neighbor] == -1) {
                        distances[neighbor] = distances[explorer] + 1;
                        toExplore.push(neighbor);
                    }
                }
            }
        }
        int FindMassScore(vector<vector<int> >& graph, 
                                int startPoint) {
            // Initialize data structures for flood fill
            vector<int> distances(graph.size(), -1);
            queue<int> toExplore;
            
            // Put start point on queue
            toExplore.push(startPoint);
            distances[startPoint] = 0;

            // Explore the rest of the graph
            int score = 0;
            while (!toExplore.empty())
            {
                // Getting next node off of queue
                int explorer = toExplore.front();
                toExplore.pop();
                
                // Add score of current node to total
                score += distances[explorer];

                // Checking the neighbors to see if they need to go on the queue       
                for (int nIt = 0; nIt < graph[explorer].size(); nIt++)
                {
                    int neighbor = graph[explorer][nIt];
                    if (distances[neighbor] == -1) {
                        distances[neighbor] = distances[explorer] + 1;
                        toExplore.push(neighbor);
                    }
                }
            }
            return score;
        }
        vector<int>* GetCentroid(vector<vector<int> >& graph, 
                                int startPoint) {
            vector<int> scores(graph.size(), -1);
            int currentNode = startPoint;

            // Find score for first node
            int bestScore = FindMassScore(graph, currentNode);
            scores[currentNode] = bestScore;

            bool betterFound = true;
            while(betterFound)
            {       
                betterFound = false;
                for (int i = 0; i < graph[currentNode].size() && !betterFound; i++)
                {
                    int neighbor = graph[currentNode][i];
                    if (scores[neighbor] == -1)
                        scores[neighbor] = FindMassScore(graph, currentNode);

                    if (scores[neighbor] < bestScore) {
                        bestScore = scores[neighbor];
                        currentNode = neighbor;
                        betterFound = true;
                    }
                }
            }

            // Find any adjacent nodes with equivalent score
            vector<int> *result = new vector<int>();
            result->push_back(currentNode);
            for (int i = 0; i < graph[currentNode].size(); i++)
            {
                int neighbor = graph[currentNode][i];
                if (scores[neighbor] == -1)
                    scores[neighbor] = FindMassScore(graph, currentNode);

                if (scores[neighbor] == bestScore) {
                    result->push_back(neighbor);
                }
            }
            return result;            
        }
    }
}