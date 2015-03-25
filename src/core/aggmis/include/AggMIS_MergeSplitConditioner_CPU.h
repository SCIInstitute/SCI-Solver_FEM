/* 
 * File:   AggMIS_MergeSplitConditioner_CPU.h
 * Author: T. James Lewis
 *
 * Created on July 4, 2013, 1:14 PM
 */

#ifndef AGGMIS_MERGESPLITCONDITIONER_CPU_H
#define	AGGMIS_MERGESPLITCONDITIONER_CPU_H

#include "AggMIS_Types.h"
#include "AggMIS_Aggregation_CPU.h"
#include "AggMIS_GraphHelpers.h"
#include "thrust/count.h"
#include "AggMIS_IOHelpers.h"

namespace AggMIS {
    namespace MergeSplitCPU {
        using namespace Types;
        using namespace std;
        
        // Functors for Thrust calls
        namespace Functors {
            struct isOutSized:public thrust::unary_function<int,int> {
                const int a, b;
                isOutSized(int _a, int _b):a(_a),b(_b){}

                __host__ __device__ 
                bool operator()(const int &x) const
                {
                    return x < a || x > b;
                }
            };
            struct lessThan:public thrust::unary_function<int,int> {
                const int a;

                lessThan(int _a): a(_a){}

                __host__ __device__
                int operator()(const int &x) const
                {
                    return x < a;
                }
            };
            struct greaterThan:public thrust::unary_function<int,int> {
                const int a;

                greaterThan(int _a): a(_a){}

                __host__ __device__
                int operator()(const int &x) const
                {
                    return x > a;
                }
            };
            struct NotNegOne:public thrust::unary_function<int,int> {
                __host__ __device__
                int operator()(int a) const
                {
                    // If value is negative return 0 else return one
                    return a < 0 ? 0 : 1;
                }
            };
            struct EqualTo:public thrust::unary_function<int,int> {
                const int a;

                EqualTo(int _a): a(_a){}

                __host__ __device__
                int operator()(const int &x) const
                {
                    return x == a;
                }
            };
        }
        
        // Merge and Split conditioner main container
        class MergeSplitConditionerCPU {
        public:
            /**
            * The primary constructor it creates a conditioner given the 
            * specified graph and aggregation. It makes an internal copy of
            * the aggregation vector, but it uses a pointer back to the given 
            * Graph object
            * @param graph The graph that the aggregation being conditioned
            * is an aggregation of.
            * @param aggregation A vector which labels every node in the graph
            * with an aggregate ID.
            */            
            MergeSplitConditionerCPU(Graph_h &graph, 
                    IntVector_h &aggregation);
            void SetSizeBounds(int min, int max);
            void SetVerbose(bool v);
            /**
             * Sets the node weights by swapping the contents of the provided 
             * vector into the nodeWeights member. Then it re-computes the part
             * sizes with weighting.
             * @param nodeWeights A vector containing the weights of each node
             * the contents of this vector are swapped out by the method.
             */
            void SetNodeWeights(IntVector_h &nodeWeights);
            /**
             * Getter for the aggregation vector.
             * @return A pointer to the current aggregation vector
             */
            IntVector_h* GetAggregation();
            /**
             * Getter for the NodeWeights vector
             * @return A pointer to the current NodeWeights vector
             */
            IntVector_h* GetNodeWeights();
            void CycleMerges(bool force);
            void CycleSplits(bool force);
            void CycleMergeSplits(float minImprove, 
                    int desiredSize);
            bool Condition(int desiredSize,
                    bool respectUpper, 
                    float tolerance, 
                    float minImprove, 
                    int maxCycles);
            void PrintProgress(ostream* output, 
                    string note,
                    bool graphStat,
                    bool progressStat,
                    bool sizeStat);
            void PrintSizeStats(ostream* output,
                    bool makeHeader);
            void PrintProgressStats(ostream* output,
                    bool makeHeader);
            void PrintGraphStats(ostream* output,
                    bool makeHeader);
            void InteractiveConsole(string message);
        private:
            bool MarkMerges(bool force);
            bool MarkSplits(bool force);
            void MarkMergeSplits(int desiredSize);
            void MakeSplits();
            void MakeMerges(bool markSplits);
            void MakeMergesDirect(bool force);
            int MergeAggregates(int aggA, int aggB);
            int MergeAggregates(int aggA, int aggB, bool fillSpot);
            void MakeSplitsDirect(bool force);
            void SplitAggregate(int agg, int newAgg);
            void MakeMergeSplits(int desiredSize);
            void UnlinkAggregate(int aggId);
            void FixSizesFromAggMap(int aggId);
            void LinkAggregate(int aggId);
            void FillAggAdjacency();
            void FillAggMap();
            void ValidateAggAdjacency();
            void ValidateAggMap();
            void ValidatePartSizes();
            void ValidateArraySizes(string message);
            
            // Data members
            Graph_h *graph;
            Graph_h *inducedGraph;
            
            int minSize, 
                maxSize, 
                outsizedParts,
                merges,
                mergeSplits,
                splits;
            
            IntVector_h aggregation,
                nodeWeights,
                distances,
                partSizes,
                weightedSizes,
                splitsToMake, 
                mergesToMake,
                mergeOffsets;
            
            // Stores lists of nodes in each aggregate
            vector<vector<int> > aggMap;
            
            // Stores the neighbors of each aggregate
            vector<vector<int> > aggAdjacency;
            
            // Stores the root point sets for each aggregate
            vector<vector<int> > rootPoints;
            
            bool verbose;
        };
    }
}
#endif	/* AGGMIS_MERGESPLITCONDITIONER_CPU_H */

