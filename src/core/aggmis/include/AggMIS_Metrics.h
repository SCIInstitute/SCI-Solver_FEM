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
    class MetricsContext {
    public:
      MetricsContext(Types::Graph_h &graph, Types::IntVector_h &aggregation);
      double GetConvexityRatio(int aggId);
      double GetEccentricityRatio(int aggId);
      double GetMinimumEnclosingBallRatio(int aggId);
      int GetAggregateCount();

    private:
      // Data structures
      Types::Graph_h *graph;
      Types::IntVector_h aggregation;
      std::vector<std::vector<int> > aggregates;
      std::vector<std::vector<int> > convexAggregates;
      int currentAggregate;

      // Counters
      int distanceLookups, makeConvexCalls;

      // Internal Methods
      int Distance(int a, int b);
      double GetEccentricityRatio(std::vector<int> &aggregate);
      double GetMinimumEnclosingBallRatio(std::vector<int> &aggregate);
      void MakeConvex(std::vector<int> &aggregate);
      void EnsureConvex(int aggId);
      std::vector<int>* FindCentroid(std::vector<int>& aggregate);
      int FindMassScore(int node, std::vector<int>& aggregate);
      std::map<int, int>* FindDistances(int rootNode, std::vector<int>& aggregate);
      std::vector<std::vector<int> >* GetShortestPaths(int startId, int endId, 
        std::map<int, int> &distances);
      std::vector<std::vector<int> >* FindExternalsInPaths(std::vector<int>& aggregate, 
        std::vector<std::vector<int> >* p);
      bool IsPathSatisfied(std::set<int>& required,
        std::vector<std::vector<int> >& pathOptions);
      std::set<int>* BruteForceMinimalNodes(std::vector<
        std::vector<std::vector<int> > >& pathOptions);
      bool IncrementGuessVector(std::vector<int>& guess, std::vector<std::vector<
        std::vector<int> > >& externalOptions);

      // Setup helpers
      void Initialize();
      void GetAggregates();
    };
  }
}

#endif	/* AGGMIS_METRICS_H */

