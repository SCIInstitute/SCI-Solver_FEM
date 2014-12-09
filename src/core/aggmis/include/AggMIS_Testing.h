/* 
 * File:   AggMIS_Testing.h
 * Author: T. James Lewis
 *
 * Created on April 11, 2013, 1:42 PM
 */

#ifndef AGGMIS_TESTING_H
#define	AGGMIS_TESTING_H

#include "types.h"
#include <vector>
#include <map>
#include <queue>
#include <set>

namespace AggMIS {
    namespace Testing {
        using namespace std;
        class MetricsContext {
        public:
            MetricsContext(Graph_CSR &graph, IdxVector_h &aggregation);
            MetricsContext(int size, int *adjIndices, int *adjacency, int *aggregation);

            double GetConvexityRatio(int aggId);
            double GetEccentricityRatio(int aggId);
            double GetMinimumEnclosingBallRatio(int aggId);
            int GetAggregateCount();
            
        private:
            // Data structures
            Graph_CSR *graph;
            vector<int> aggregation;
            vector<vector<int> > aggregates;
            vector<vector<int> > convexAggregates;
            int currentAggregate;
            
            // Counters
            int distanceLookups, makeConvexCalls;
            
            // Internal Methods
            int Distance(int a, int b);
            double GetEccentricityRatio(vector<int> &aggregate);
            double GetMinimumEnclosingBallRatio(vector<int> &aggregate);
            void MakeConvex(vector<int> &aggregate);
            void EnsureConvex(int aggId);
            vector<int>* FindCentroid(vector<int>& aggregate);
            int FindMassScore(int node, vector<int>& aggregate);
            map<int,int>* FindDistances(int rootNode, vector<int>& aggregate);
            vector<vector<int> >* GetShortestPaths(int startId, int endId, map<int,int> &distances);
            vector<vector<int> >* FindExternalsInPaths(vector<int>& aggregate, vector<vector<int> >* p);
            bool IsPathSatisfied(set<int>& required, vector<vector<int> >& pathOptions);
            set<int>* BruteForceMinimalNodes(vector<vector<vector<int> > >& pathOptions);
            bool IncrementGuessVector(vector<int>& guess, vector<vector<vector<int> > >& externalOptions);
            
            // Setup helpers
            void Initialize();
            void GetAggregates();
        };
    }
}

#endif	/* AGGMIS_TESTING_H */

