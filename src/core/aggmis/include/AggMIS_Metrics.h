/* 
 * File:   AggMIS_Metrics.h
 * Author: T. James Lewis
 *
 * Created on May 1, 2013, 12:19 PM
 */

#ifndef AGGMIS_METRICS_H
#define	AGGMIS_METRICS_H
#include "AggMIS_Types.h"
#include <vector>
#include <queue>
#include <set>
#include <map>
namespace AggMIS {
    namespace Metrics {
        using namespace Types;
        using namespace std;
        class MetricsContext {
        public:
            MetricsContext(Graph_h &graph, IntVector_h &aggregation);
            double GetConvexityRatio(int aggId);
            double GetEccentricityRatio(int aggId);
            double GetMinimumEnclosingBallRatio(int aggId);
            int GetAggregateCount();
            
        private:
            // Data structures
            Graph_h *graph;
            IntVector_h aggregation;
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

#endif	/* AGGMIS_METRICS_H */

