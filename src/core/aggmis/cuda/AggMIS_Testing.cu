#include "AggMIS_Testing.h"
//#include "debugHelpers.h"
namespace AggMIS {
    namespace Testing {
        MetricsContext::MetricsContext(Graph_CSR& graph, IdxVector_h& aggregation) {
            this->graph = &graph;
            this->aggregation.resize(aggregation.size());
            thrust::copy(aggregation.begin(), aggregation.end(), this->aggregation.begin());
            Initialize();
        }
        MetricsContext::MetricsContext(int size, int* adjIndices, int* adjacency, int* aggregation) {
            graph = new Graph_CSR(size, adjIndices, adjacency);
            Initialize();
        }
        
        // Public Methods
        double MetricsContext::GetConvexityRatio(int aggId) {
            if (aggId < aggregates.size() && aggId >= 0)
            {
                distanceLookups = 0;
                makeConvexCalls = 0;
                EnsureConvex(aggId);
                printf("Found convexity ratio for aggregate %d of size %d with %d MakeConvex calls and %d FindDistances calls.\n", aggId, aggregates[aggId].size(), makeConvexCalls, distanceLookups);
                
                return (double)aggregates[aggId].size() / convexAggregates[aggId].size();
            }
            return -1.0;
        } 
        double MetricsContext::GetEccentricityRatio(int aggId) {
            if (aggId < aggregates.size() && aggId >= 0)
            {
                EnsureConvex(aggId);
                return GetEccentricityRatio(convexAggregates[aggId]);
            }
            return -1.0;
        }
        int MetricsContext::GetAggregateCount() {
            return aggregates.size();
        }

        // Internal Methods
        double MetricsContext::GetEccentricityRatio(vector<int>& aggregate) {
            // Find the centroidal nodes
            vector<int> *centroid = FindCentroid(aggregate);

            // Find the distances from the centroidal nodes
            vector<map<int,int>*> distances(centroid->size());
            for (int i = 0; i < distances.size(); i++)
                distances[i] = FindDistances(centroid->at(i), aggregate);

            // Find farthest internal node by finding highest average distance to centroidal nodes
            double farthestInternal = 0.0;
            for (int i = 0; i < aggregate.size(); i++)
            {
                double distance = 0.0;
                for (int j = 0; j < distances.size(); j++)
                    distance += distances[j]->at(aggregate[i]);
                distance /= distances.size();
                farthestInternal = std::max(distance, farthestInternal);
            }

            // Find external nodes seen by centroidal nodes
            set<int> externalCandidates;
            for (int i = 0; i < distances.size(); i++)
            {
                map<int,int>::iterator it;
                for (it = distances[i]->begin(); it != distances[i]->end(); it++)
                {
                    int node = (*it).first;
                    if(!std::binary_search(aggregate.begin(), aggregate.end(), (*it).first))
                        externalCandidates.insert(node);
                }
            }
            
            // Check each external to find the one with minimum average distance 
            double nearestExternal = 100000;
            set<int>::iterator sIt;
            for (sIt = externalCandidates.begin(); sIt != externalCandidates.end(); sIt++)
            {
                double currentScore = 0.0;
                bool unseen = false;
                for (int i = 0; i < distances.size(); i++)
                {
                    if (distances[i]->count(*sIt) > 0)
                        currentScore += distances[i]->at(*sIt);
                    else
                        unseen = true;
                }
                if (!unseen)
                {
                    currentScore /= distances.size();
                    if (currentScore < nearestExternal)
                        nearestExternal = currentScore;
                }
            }
            
            // Memory clean up:
            for (int i = 0; i < distances.size(); i++)
                delete distances[i];

            // Return value for metric
            return (nearestExternal - 1) / farthestInternal;
        }
        double MetricsContext::GetMinimumEnclosingBallRatio(vector<int>& aggregate) {
            /* Method: find distances out from first and second aggregate nodes then
            find the intersection of the distance envelopes, then find the next nodes
            envelope and intersect it with the first intersection. The center of the 
            minimal enclosing ball must lie within the intersection of all nodes distance
            envelopes*/

            // Ensure input it sorted
            std::sort(aggregate.begin(), aggregate.end());

            // If there is only one node in aggregate return
            if (aggregate.size() < 2)
                return 1.0;

            // Otherwise find the intersection of the first two nodes distance envelope
            map<int,int> *a = FindDistances(aggregate[0], aggregate);
            map<int,int> *b = FindDistances(aggregate[1], aggregate);

            map<int,int>::iterator it;
            set<int> intersection;
            for (it = a->begin(); it != a->end(); it++)
                if (b->count((*it).first))
                    intersection.insert((*it).first);

        //    // Delete stuff
        //    delete a;
        //    delete b;

            // Continuing on for every envelope in the aggregate:
            for (int i = 2; i < aggregate.size(); i++)
            {
                a = FindDistances(aggregate[i], aggregate);
                set<int> toRemove;
                set<int>::iterator sIt;
                for (sIt = intersection.begin(); sIt != intersection.end(); sIt++)
                    if(a->count(*sIt) == 0)
                        toRemove.insert(*sIt);
                for (sIt = toRemove.begin(); sIt != toRemove.end(); sIt++)
                    intersection.erase(*sIt);
        //        delete a;
            }

            // Now finding the enclosing sphere sizes for all nodes in intersection
            set<int>::iterator sIt;
        //    int bestSphereCenter = -1;
            int bestSphereSize = 1000000;
            for(sIt = intersection.begin(); sIt != intersection.end(); sIt++)
            {
                a = FindDistances(*sIt, aggregate);
                int farthestInternal = 0;
                for (it = a->begin(); it != a->end(); it++)
                    if (std::binary_search(aggregate.begin(), aggregate.end(), (*it).first))
                        farthestInternal = std::max((*it).second, farthestInternal);

                int nodeCount = 0;
                for (it = a->begin(); it != a->end(); it++)
                    if ((*it).second <= farthestInternal)
                        nodeCount++;

                if (bestSphereSize > nodeCount)
                {
                    bestSphereSize = nodeCount;
        //            bestSphereCenter = *sIt;
                }
        //        delete a;
            }

            // Returning the ratio score
        //    printf("Found center of minimum enclosing sphere at %d with size of %d\n", bestSphereCenter, bestSphereSize);

            return (double)aggregate.size()/bestSphereSize;
        }
        void MetricsContext::MakeConvex(vector<int>& aggregate) {
            makeConvexCalls++;
            // Set to keep track of 'must have' external nodes
            set<int> neededExternal;

            // Keep track of all options for other external nodes
            vector<vector<vector<int> > > externalPossibilities;

            // Find paths 
            for (int rootIdx = 0; rootIdx < aggregate.size(); rootIdx++)
            {
                int startNode = aggregate[rootIdx];
                map<int,int> *distances = FindDistances(startNode, aggregate);
//                printf("Found distance map from node %d \n", startNode);
//                int counter = 0;
//                printf("\t");
//                for (map<int,int>::iterator it = distances->begin(); it != distances->end(); it++)
//                {
//                    if (counter++ == 10)
//                    {
//                        counter = 0;
//                        printf("\n\t");
//                    }
//                    printf("(%d,%d) ", (*it).first, (*it).second);
//                }
//                char s;
//                cin >> s;
                
                // Get paths for each distinct pair of nodes in the aggregate
                for (int endIdx = rootIdx + 1; endIdx < aggregate.size(); endIdx++)
                {
                    int endNode = aggregate[endIdx];
                    
                    vector<vector<int> > *paths = GetShortestPaths(startNode, endNode, *distances);
                    vector<vector<int> > *externals = FindExternalsInPaths(aggregate, paths);

                    // If path not satisfied add to externalPossibilities
                    if (!IsPathSatisfied(neededExternal, *externals))
                        externalPossibilities.push_back(vector<vector<int> >(*externals));

                    // Memory cleanup
                    delete paths;
                    delete externals;
                }
            }

            // Simplify until no longer possible:
            bool allDone = false;
            while (!allDone)
            {
                allDone = true;
                int counter = 0;
                while (counter < externalPossibilities.size())
                {
                    if (IsPathSatisfied(neededExternal, externalPossibilities[counter]))
                    {
                        externalPossibilities.erase(externalPossibilities.begin() + counter);
                        allDone = false;
                    }
                    else
                    {
                        counter++;
                    }
                }
            }

            // If there were required externals found add them and recurse:
            if (neededExternal.size() > 0)
            {
        //        printf("Adding some needed nodes:\n");
        //        debugHelpers::printResult(&neededExternal, "Nodes added");

                aggregate.insert(aggregate.end(), neededExternal.begin(), neededExternal.end());
                std::sort(aggregate.begin(), aggregate.end());
                MakeConvex(aggregate);
            }

            // If there are external possibilities and no required we need brute force:
            else if (externalPossibilities.size() > 0)
            {
                set<int>* result = BruteForceMinimalNodes(externalPossibilities);

        //        printf("Had to brute force solve:\n");
        //        debugHelpers::printResult(result, "Nodes added");

                aggregate.insert(aggregate.end(), result->begin(), result->end());
                std::sort(aggregate.begin(), aggregate.end());
                delete result;
                MakeConvex(aggregate);
            }   

            // Method falls through when the aggregate is convex
        }
        void MetricsContext::EnsureConvex(int aggId) {
            if (aggId < aggregates.size() && aggId >= 0)
            {
                if (convexAggregates[aggId].size() < aggregates[aggId].size())
                {
                    convexAggregates[aggId].assign(aggregates[aggId].begin(), aggregates[aggId].end());
                    MakeConvex(convexAggregates[aggId]);
                }
            }
        }
        vector<int>* MetricsContext::FindCentroid(vector<int>& aggregate) {
            // To store the mass scores in:
            map<int,int> scores;
            int currentNode = aggregate[0];

            // Find score for first node
            int bestScore = FindMassScore(currentNode, aggregate);
            scores[currentNode] = bestScore;

            bool betterFound = true;
            while(betterFound)
            {       
        //        printf("Checking around node %d with score %d\n", currentNode, scores[currentNode]);
                betterFound = false;
                int start = graph->adjIndexes[currentNode];
                int end = graph->adjIndexes[currentNode + 1];
                for (int i = start; i < end && !betterFound; i++)
                {
                    int neighbor = graph->adjacency[i];
                    if (scores.count(neighbor) == 0)
                        scores[neighbor] = FindMassScore(neighbor, aggregate);

        //            printf("\tNeighbor %d with score %d\n", neighbor, scores[neighbor]);
                    if (scores[neighbor] < bestScore)
                    {
                        bestScore = scores[neighbor];
                        currentNode = neighbor;
                        betterFound = true;
                    }
                }
            }

        //    printf("Best value of %d found at node %d\n", bestScore, currentNode);
            // Find any adjacent nodes with equivalent score
            vector<int> *result = new vector<int>();
            result->push_back(currentNode);
            int start = graph->adjIndexes[currentNode];
            int end = graph->adjIndexes[currentNode + 1];
            for (int i = start; i < end; i++)
            {
                int neighbor = graph->adjacency[i];
                if (scores.count(neighbor) == 0)
                    scores[neighbor] = FindMassScore(neighbor, aggregate);

        //        printf("\t Neighbor %d found with score %d\n", neighbor, scores[neighbor]);
                if (scores[neighbor] == bestScore)
                {
                    result->push_back(neighbor);
                }
            }
            return result;
        }
        int MetricsContext::FindMassScore(int node, vector<int>& aggregate) {
            map<int,int> *distances = FindDistances(node, aggregate);
            int score = 0;
            for (int i = 0; i < aggregate.size(); i++)
                score += (*distances)[aggregate[i]];
            delete distances;
            return score;
        }
        map<int,int>* MetricsContext::FindDistances(int rootNode, vector<int>& aggregate) {
            distanceLookups++;
            // Create queue and set distance of initial node
            map<int,int> *distances = new map<int,int>();
            (*distances)[rootNode] = 0;
            
            // Queue for exploring out from root node
            queue<int> exploring;
            exploring.push(rootNode);
            
            // Keeping track of whether the exploration should continue:
            int internalsSeen = 0;      // How many nodes in the aggregate have been marked 

            // Start to explore the queue:     
            while (!exploring.empty())
            {   
                // Pull node off of queue
                int node = exploring.front();
                exploring.pop();

                // Get distance of current node:
                int currentDistance = (*distances)[node];

                // Examine the neighbors
                for (int j = graph->adjIndexes[node]; j < graph->adjIndexes[node + 1]; j++)
                {
                    int neighbor = graph->adjacency[j];
                    if (distances->count(neighbor) == 0)
                    {
                        (*distances)[neighbor] = currentDistance + 1;
                        exploring.push(neighbor);
                    }
                }

                // Checking if the current node is an internal:
                if (std::binary_search(aggregate.begin(), aggregate.end(), node))
                {   
                    internalsSeen++;
                }

                // If we have seen all nodes in the aggregate stop the search
                if (internalsSeen == aggregate.size())
                {
                    while(!exploring.empty())
                        exploring.pop();
                    break;
                }
            }
            return distances;
        }
        vector<vector<int> >* MetricsContext::GetShortestPaths(int startId, int endId, map<int,int> &distances) {     
            vector<vector<int> >* result = new vector<vector<int> >(1);
            vector<vector<int> > &paths = *result;

            // Starting the trace back with current node:
            paths[0].push_back(endId);
            int activePath = 0;
            while (activePath < paths.size())
            {
                // Get the current end of the current path and its distance
                int endNode = paths[activePath].back();

                int distance = distances[endNode];
                bool branched = false;

                // Check the neighbors of the current endNode to continue
                for (int nIt = graph->adjIndexes[endNode]; nIt < graph->adjIndexes[endNode + 1]; nIt++)
                {
                    // Get neighbor and its distance
                    int neighbor = graph->adjacency[nIt];
                    int neighborDist = distances[neighbor];

                    // If the neighbor is one closer than current it is on 
                    // a shortest path
                    if (neighborDist == distance - 1)
                    {
                        // If there is a branch add a new path
                        if (branched)
                        {
                            // Add a new path starting with a copy of current
                            // and append current neighbor
                            paths.push_back(vector<int>(paths[activePath].begin(),paths[activePath].end() - 1));
                            paths.back().push_back(neighbor);    
                        }
                        // Add node to current path and mark as branched
                        else
                        {
                            paths[activePath].push_back(neighbor);
                            branched = true;
                        }
                    }
                }

                // Check if the active path is complete and move on
                if (paths[activePath].back() == startId)
                    activePath++;
            }
            return result;
        }
        vector<vector<int> >* MetricsContext::FindExternalsInPaths(vector<int>& aggregate, vector<vector<int> >* p) {
            // Create the result vector and get a local reference to the paths
            vector<vector<int> > *result = new vector<vector<int> >();
            vector<vector<int> > &paths = *p;

            // Go through each path and find external nodes to add
            for (int p = 0; p < paths.size(); p++)
            {
                vector<int> externals;
                for (int pp = 0; pp < paths[p].size(); pp++)
                {
                    int pathNode = paths[p][pp];

                    if (!std::binary_search(aggregate.begin(), aggregate.end(), pathNode))
                    {
                        externals.push_back(pathNode);
                    }
                }

                // If there were external nodes found add to the list
                if (externals.size() > 0)
                {
                    result->push_back(vector<int>(externals));
                }

                // If a clear path was found return an empty path list
                else
                {
                    result->clear();
                    return result;
                }
            }

            // Only keep meaningful elements
            vector<bool> keepers(result->size(), true);
            vector<vector<int> > &r = *result;
            for(int a = 0; a < r.size(); a++)
            {
                // Compare with each following node:
                for (int b = a + 1; b < r.size() && keepers[a]; b++)
                {
                    // Make sure the b node is also not marked for deletion
                    if (keepers[b])
                    {
                        // If the nodes are equal eliminate one
                        if (r[a].size() == r[b].size())
                        {
                            // Assume that they are matched:
                            bool matched = true;
                            for (int i = 0; i < r[a].size(); i++)
                            {
                                bool good = false;
                                for (int j = 0; j < r[b].size(); j++)
                                {
                                    if (r[a][i] == r[b][j])
                                    {
                                        good = true;
                                        break;
                                    }
                                }
                                if (!good)
                                {
                                    matched = false;
                                    break;
                                }
                            }
                            if (matched)
                            {
                                keepers[b] = false;
                            }
                        }

                        // Otherwise if the every element in the smaller is in the 
                        // larger as well, eliminate the larger.
                        else
                        {
                            int small = a;
                            int big = b;
                            if (a > b)
                            {
                                small = b;
                                big = a;
                            }

                            // Assume that they are matched:
                            bool matched = true;
                            for (int i = 0; i < r[small].size(); i++)
                            {
                                bool good = false;
                                for (int j = 0; j < r[big].size(); j++)
                                {
                                    if (r[small][i] == r[big][j])
                                    {
                                        good = true;
                                        break;
                                    }
                                }
                                if (!good)
                                {
                                    matched = false;
                                    break;
                                }
                            }
                            if (matched)
                            {
                                keepers[big] = false;
                            }
                        }
                    }
                }
            }
            int toRemove = std::count(keepers.begin(), keepers.end(), false);

            if (toRemove == 0)
                return result;

            // If there were some to remove do it:
            vector<vector<int> > *trimmed = new vector<vector<int> >();
            trimmed->resize(r.size() - toRemove);
            int insertNext = 0;
            for (int i = 0; i < r.size(); i++)
                if (keepers[i])
                    (*trimmed)[insertNext++] = r[i];

            delete result;
            return trimmed;
        }
        bool MetricsContext::IsPathSatisfied(set<int>& required, vector<vector<int> >& pathOptions) {
            // If there are no options it is satisfied.
            if (pathOptions.size() == 0)
                return true;

            // If there is only one option those nodes are required so add and be done:
            if (pathOptions.size() == 1)
            {
                required.insert(pathOptions[0].begin(), pathOptions[0].end());
                return true;
            }

            // If any option is satisfied by nodes already required we are done:
            for (int i = 0; i < pathOptions.size(); i++)
            {
                bool satisfied = true;
                for (int j = 0; j < pathOptions[i].size(); j++)
                    if (required.count(pathOptions[i][j]) == 0)
                        satisfied = false;
                if (satisfied)
                {
                    return true;
                }
            }

            return false;
        }
        set<int>* MetricsContext::BruteForceMinimalNodes(vector<vector<vector<int> > >& pathOptions) {
            set<int> attempt;
            for (int cIt = 0; cIt < pathOptions.size(); cIt++)
                attempt.insert(pathOptions[cIt][0].begin(),pathOptions[cIt][0].end());

            // Finding how many possibilites there are to check:
            int possibilities = pathOptions[0].size();
            for (int i = 1; i < pathOptions.size(); i++)
                possibilities *= pathOptions[i].size();
            if (possibilities > 100)
            {
                printf("%d options for brute forcing!\n", possibilities);
                debugHelpers::printResult(&pathOptions, "Options");
                
            }
            
            // Set best found to initial attempt size
            int bestFound = attempt.size();
            set<int> bestFoundSet = attempt;

            // Setting the guess
            vector<int> guess(pathOptions.size(), 0);

            // Trying all remaining combinations
            int guessCount = 1;
            while(IncrementGuessVector(guess, pathOptions))
            {
                guessCount++;

                // Clearing attempt to known nodes
                attempt.clear();

                // Making choices according to guess vector
                for (int cIt = 0; cIt < pathOptions.size(); cIt++)
                {
                    attempt.insert(pathOptions[cIt][0].begin(),pathOptions[cIt][0].end());
                }

                // If this attempt has better result save it
                if (attempt.size() < bestFound)
                {
                    bestFound = attempt.size();
                    bestFoundSet = attempt;
                }
            }

            set<int> *result = new set<int>(bestFoundSet);
            return result;
        }
        bool MetricsContext::IncrementGuessVector(vector<int>& guess, vector<vector<vector<int> > >& externalOptions) {
            bool incremented = false;
            if (guess.size() == 0)
                return false;

            // Increment the guess vector
            int position = guess.size() - 1;
            while (true)
            {
                // If the current position has a higher option increment and stop:
                if (guess[position] < externalOptions[position].size() - 1)
                {
                    guess[position]++;
                    incremented = true;
                    break;
                }
                // If the current position can go no higher:
                else
                {
                    // If there is no preceding position all options have been explored.
                    if (position == 0)
                        break;

                    // Otherwise reset this position to zeros and move up to next position.
                    else
                    {
                        guess[position] = 0;
                        position--;
                    }
                }
            }
            return incremented;
        }        
        
        // Setup Helpers
        void MetricsContext::Initialize() {
            GetAggregates();
            convexAggregates.resize(aggregates.size());
        }
        void MetricsContext::GetAggregates() {
            // Iterate through the aggregation and populate aggregates
            for (int i = 0; i < aggregation.size(); i++)
            {
                int aggId = aggregation[i];

                // Make sure an entry for this aggregate exists
                if (aggregates.size() <= aggId)
                {
                    aggregates.resize(aggId + 1);
                }

                // Push the current node onto the list for its aggregate.
                aggregates[aggId].push_back(i);     
            }
        }
    }
}
