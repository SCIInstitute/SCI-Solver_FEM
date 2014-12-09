/* 
 * File:   AggMIS_MergeSplitConditioner_CPU.cu
 * Author: T. James Lewis
 *
 * Created on July 4, 2013, 1:14 PM
 */
 #include <algorithm>
#include "AggMIS_MergeSplitConditioner_CPU.h"
namespace AggMIS {
    namespace MergeSplitCPU {
        MergeSplitConditionerCPU::MergeSplitConditionerCPU(Graph_h &graph, 
                                        IntVector_h &aggregation) {
            this->graph = &graph;
            this->aggregation.assign(aggregation.begin(), aggregation.end());
            inducedGraph = GraphHelpers::GetInducedGraph(graph, aggregation);
            
            // Getting the sizes of each aggregate:
            IntVector_h* ps = Aggregation::GetPartSizes(aggregation);
            partSizes.swap(*ps);
            delete ps;
            
            // Filling in
            FillAggAdjacency();
            FillAggMap();
            
            verbose = false;
            minSize = 20;
            maxSize = 30;
            outsizedParts = 0;
            merges = 0;
            mergeSplits = 0;
            splits = 0;
        }
        void MergeSplitConditionerCPU::SetSizeBounds(int min, 
                                        int max) {
            minSize = min;
            maxSize = max;            
        }
        void MergeSplitConditionerCPU::SetVerbose(bool v) {
            verbose = v;
        }
        void MergeSplitConditionerCPU::SetNodeWeights(IntVector_h &input) {
            nodeWeights.swap(input);
            IntVector_h *ws = Aggregation::GetPartSizes(aggregation, nodeWeights);
            weightedSizes.swap(*ws);
            ws->clear();
            delete ws;
        }
        IntVector_h* MergeSplitConditionerCPU::GetAggregation() {
            return &aggregation;
        }
        IntVector_h* MergeSplitConditionerCPU::GetNodeWeights() {
            return &nodeWeights;
        }
        void MergeSplitConditionerCPU::CycleMerges(bool force) {
//            int count = 0;
//            while (MarkMerges(force))
//            {
//                MakeMerges(false);
//                count++;
//            }
//            if (verbose)
//                printf("Finished cycling merges after %d cycles.\n", count);
            MakeMergesDirect(force);
        }
        void MergeSplitConditionerCPU::CycleSplits(bool force) {
            int count = 0;
//            while (MarkSplits(force))
//            {
//                MakeSplits();
//                count++;
//            }
            int splitsMade = 1;
            while (splitsMade > 0) {
                int startingSplits = splits;
                MakeSplitsDirect(force);
                splitsMade = splits - startingSplits;
                count++;
            }
            if (verbose)
                printf("Finished cycling splits after %d cycles.\n", count);
        }
        void MergeSplitConditionerCPU::CycleMergeSplits(float minImprove, 
                int desiredSize) {
            // Start with an initial cycle
            MakeMergeSplits(desiredSize);
            
            // Choosing which sizes to use:
            IntVector_h *sizes = &partSizes;
            if (nodeWeights.size() > 0)
                sizes = &weightedSizes;
            
            // Check to see how much improvement was made
            int after = thrust::count_if(sizes->begin(), sizes->end(), Functors::isOutSized(minSize, maxSize));
            float improvement = (float)(outsizedParts - after)/outsizedParts;
            outsizedParts = after;
            
            // While good progress is being made continue cycling
            while (improvement > minImprove)
            {
                // Perform Cycle and check improvement
                MakeMergeSplits(desiredSize);
                after = thrust::count_if(sizes->begin(), sizes->end(), Functors::isOutSized(minSize, maxSize));
                improvement = (float)(outsizedParts - after)/outsizedParts;
                outsizedParts = after;
            }
        }
        bool MergeSplitConditionerCPU::Condition(int desiredSize,
                bool respectUpper, 
                float tolerance, 
                float minImprove, 
                int maxCycles) {
            if (verbose)
                PrintProgress(&cout, "Starting conditioning.", true, true, true);
            
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
                printf("Initial outsized ratio is: %d / %d = %f\n", 
                        outsizedParts, partSizes.size(), currentRatio);
            
            // Starting main cycle phase
            int counter = 1;
            bool highCycle = false;
            while(currentRatio > tolerance && counter++ < maxCycles)
            {
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
                    ss << "After condition cycle: " << counter++;
                    PrintProgress(&cout, ss.str(), true, true, true);
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

            if (verbose)
                PrintProgress(&cout, "After conditioning completed.", true, true, true);
            
            // Checking if the size constraints are met for the return
            if (respectUpper)
                return (oversized == 0 && (float)outsizedParts / partSizes.size() < tolerance);
            else
                return (undersized == 0 && (float)outsizedParts / partSizes.size() < tolerance);
        }
        void MergeSplitConditionerCPU::PrintProgress(ostream* output, 
                string note,
                bool graphStat,
                bool progressStat,
                bool sizeStat) {
            *output << "\n------------------- Progress Check ------------------\n";
            *output << "Note: " << note.c_str() << "\n";

            if (graphStat)
                PrintGraphStats(output, false);
            
            if (progressStat)
                PrintProgressStats(output, false);
            
            if (sizeStat)
                PrintSizeStats(output, false);

            *output << "-----------------------------------------------------\n\n";
        }
        void MergeSplitConditionerCPU::PrintSizeStats(ostream* output,
                bool makeHeader) {
            if (makeHeader)
                *output << "\n--------------------- Size Check --------------------\n";
            
            // Choosing which sizes to use:
            IntVector_h *sizes = &partSizes;
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
        void MergeSplitConditionerCPU::PrintProgressStats(ostream* output,
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
        void MergeSplitConditionerCPU::PrintGraphStats(ostream* output,
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
            
            IntVector_h *valences = GraphHelpers::GetValences(*graph);
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
        void MergeSplitConditionerCPU::InteractiveConsole(string message) {
            // Start off by printing overall status info and message
            PrintProgress(&cout, message, true, true, true);
            
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
//                        if (cycling)
//                            CycleMerges(force);
//                        else {
//                            MarkMerges(force);
//                            MakeMerges(false);
//                        }
                        MakeMergesDirect(force);
                        string msg = force ? "After forced merges" : "After optimal merges";
                        PrintProgress(&cout, msg, false, true, true);
                    }
                    if (operation == 's')
                    {
//                        if (cycling)
//                            CycleSplits(force);
//                        else {
//                            MarkSplits(force);
//                            MakeSplits();
//                        }
                        MakeSplitsDirect(force);
                        string msg = force ? "After forced splits" : "After optimal splits";
                        PrintProgress(&cout, msg, false, true, true);                               
                    }
                    if (operation == 'g')
                    {
//                        if (cycling)
//                            CycleMergeSplits(minImprove, desiredSize);
//                        else 
//                            MakeMergeSplits(desiredSize);
                        MakeMergeSplits(desiredSize);
                        PrintProgress(&cout, "After merge-splits", false, true, true);
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
                    PrintProgress(&cout, "After conditioning", false, true, true);
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
        bool MergeSplitConditionerCPU::MarkMerges(bool force) {
            bool marked = false;
            
            // Initializing mergesToMake array
            mergesToMake.assign(inducedGraph->Size(), -1);
            
            // Get the appropriate sizes
            IntVector_h &sizes = nodeWeights.size() > 0 ? weightedSizes : partSizes;
            
            // Figure out how large aggregates should be
            int desiredSize = (minSize + maxSize) / 2;
            
            // For every aggregate see if it should merge
            for (int aggId = 0; aggId < inducedGraph->Size(); aggId++) {
                // Getting size of current aggregate
                int currentSize = sizes[aggId];
                
                // Tracking the best seen merge
                int bestMerge = -1;
                int smallestDifference = INT_MAX;
                
                // If aggregate too small check for merges:
                if (currentSize < minSize && mergesToMake[aggId] == -1) {
                    // Look at neighboring aggregates
                    for (int* nIt = inducedGraph->nStart(aggId); 
                            nIt != inducedGraph->nEnd(aggId); 
                            nIt++) {
                        int neighborAgg = *nIt;
                        
                        // Only handle neighbors not already merging
                        if (mergesToMake[neighborAgg] == -1) {
                            int neighborSize = sizes[neighborAgg];
                            int mergedSize = currentSize + neighborSize;
                            int difference = mergedSize > desiredSize ? 
                                                mergedSize - desiredSize : 
                                                desiredSize - mergedSize;
                            if (mergedSize <= maxSize || force) {
                                if (difference < smallestDifference) {
                                    smallestDifference = difference;
                                    bestMerge = neighborAgg;
                                }
                            }
                        }
                    }
                }
                
                if (bestMerge != -1) {
                    mergesToMake[aggId] = bestMerge;
                    mergesToMake[bestMerge] = aggId;
                    marked = true;
                }
            }
            return marked;
        }
        bool MergeSplitConditionerCPU::MarkSplits(bool force) {
            // Initialize
            
            
            // Get the appropriate sizes
            IntVector_h &sizes = nodeWeights.size() > 0 ? weightedSizes : partSizes;
            return false;
        }
        void MergeSplitConditionerCPU::MarkMergeSplits(int desiredSize) {
            
        }
        void MergeSplitConditionerCPU::MakeSplits() {
            
        }
        void MergeSplitConditionerCPU::MakeMerges(bool markSplits) {
            // Figuring the offsets to use
            int offset = 0;
            mergeOffsets.resize(mergesToMake.size());
            for (int i = 0; i < mergesToMake.size(); i++)
                mergeOffsets[i] = mergesToMake[i] != -1 && mergesToMake[i] < i ?
                    ++offset : offset;
            
            // Making the merges
            for (int i = 0; i < aggregation.size(); i++) {
                int aggId = aggregation[i];
                int mergeTo = mergesToMake[aggId];
                if (mergeTo != -1 && mergeTo < aggId) 
                    aggregation[i] = mergeTo - mergeOffsets[mergeTo];
                else
                    aggregation[i] = aggId - mergeOffsets[aggId];
            }
            
            // Refiguring stuff
            merges += mergeOffsets.back();
            delete inducedGraph;
            inducedGraph = GraphHelpers::GetInducedGraph(*graph, aggregation);
            IntVector_h *ps = Aggregation::GetPartSizes(aggregation);
            partSizes.swap(*ps);
            ps->clear();
            delete ps;
            if (nodeWeights.size() > 0) {
                IntVector_h *ws = Aggregation::GetPartSizes(aggregation, nodeWeights);
                weightedSizes.swap(*ws);
                ws->clear();
                delete ws;
            }
        }
        void MergeSplitConditionerCPU::MakeMergesDirect(bool force) {
            // Get the appropriate sizes
            IntVector_h &sizes = nodeWeights.size() > 0 ? weightedSizes : partSizes;
            
            // Figure out how large aggregates should be
            int desiredSize = (minSize + maxSize) / 2;
            
            // For every aggregate see if it should merge
            int aggId = 0;
            while (aggId < aggAdjacency.size()) {
                // Getting size of current aggregate
                int currentSize = sizes[aggId];
                
                // Tracking the best seen merge
                int bestMerge = -1;
                int smallestDifference = INT_MAX;
                
                // If aggregate too small check for merges:
                while (currentSize < minSize) {
                    // Look at neighboring aggregates
                    for (int nIt = 0; nIt < aggAdjacency[aggId].size(); nIt++) {
                        int neighborAgg = aggAdjacency[aggId][nIt];
                        int neighborSize = sizes[neighborAgg];
                        int mergedSize = currentSize + neighborSize;
                        int difference = mergedSize > desiredSize ? 
                                            mergedSize - desiredSize : 
                                            desiredSize - mergedSize;
                        if (mergedSize <= maxSize || force) {
                            if (difference < smallestDifference) {
                                smallestDifference = difference;
                                bestMerge = neighborAgg;
                            }
                        }
                    }
                    if (bestMerge != -1) {
                        if (verbose) {
                            printf("Aggregate %d of size %d found neighbor %d of size %d to merge with.\n",
                                    aggId, currentSize, bestMerge, sizes[bestMerge]);
                        }
                        aggId = MergeAggregates(aggId, bestMerge);
                        if (verbose) {
                            printf("After merge Aggregate %d has size %d\n",
                                    aggId, sizes[aggId]);
                        }
                        merges++;
                        
                        // Resetting to look for other merges
                        currentSize = sizes[aggId];
                        bestMerge = -1;
                        smallestDifference = INT_MAX;
                    }
                    else {
                        if (verbose) {
                            printf("No merges found for aggregate %d of size %d\n",
                                    aggId, currentSize);
                        }
                        break;
                    }
                }
                aggId++;
            }
        }
        int MergeSplitConditionerCPU::MergeAggregates(int aggA, int aggB) {
            return MergeAggregates(aggA, aggB, true);
        }
        int MergeSplitConditionerCPU::MergeAggregates(int aggA, int aggB, bool fillSpot) {
            // Make sure aggA has the lower index
            if (aggA > aggB) {
                int swapper = aggB;
                aggB = aggA;
                aggA = swapper;
            }
            
            // Mark nodes in aggB as in aggA
            for (int nIt = 0; nIt < aggMap[aggB].size(); nIt++)
                aggregation[aggMap[aggB][nIt]] = aggA;
            
            // Add nodes in aggB to aggA's node list
            aggMap[aggA].insert(aggMap[aggA].end(), 
                                aggMap[aggB].begin(), 
                                aggMap[aggB].end());
            sort(aggMap[aggA].begin(), aggMap[aggA].end());
            
            // Clearing out aggB's node list
            aggMap[aggB].clear();
            
            // Removing edges to aggB and replacing with edges to aggA
            for (int nIt = 0; nIt < aggAdjacency[aggB].size(); nIt++) {
                int neighborAgg = aggAdjacency[aggB][nIt];
                
                // If the neighbor of aggB is also a neighbor of aggA
                // or is aggA just remove reference to aggB. 
                if (binary_search(aggAdjacency[neighborAgg].begin(),
                                aggAdjacency[neighborAgg].end(),
                                aggA) || neighborAgg == aggA) {
                    remove(aggAdjacency[neighborAgg].begin(),
                            aggAdjacency[neighborAgg].end(),
                            aggB);
                    aggAdjacency[neighborAgg].pop_back();
                }
                // Otherwise remove the reference to aggB and add one
                // to aggA
                else {
                    remove(aggAdjacency[neighborAgg].begin(),
                            aggAdjacency[neighborAgg].end(),
                            aggB);
                    aggAdjacency[neighborAgg].back() = aggA;
                    sort(aggAdjacency[neighborAgg].begin(),
                            aggAdjacency[neighborAgg].end());
                }
            }
            
            // Setting new size of aggA
            partSizes[aggA] += partSizes[aggB];
            if (nodeWeights.size() > 0)
                weightedSizes[aggA] += weightedSizes[aggB];
            
            // Getting the union of adjacency for merged aggregate
            vector<int> temp(aggAdjacency[aggA].size() + 
                            aggAdjacency[aggB].size());
            remove(aggAdjacency[aggB].begin(),
                            aggAdjacency[aggB].end(),
                            aggA);
            vector<int>::iterator newEnd;
            newEnd = set_union(aggAdjacency[aggA].begin(),
                    aggAdjacency[aggA].end(),
                    aggAdjacency[aggB].begin(),
                    aggAdjacency[aggB].end() - 1,
                    temp.begin());
            temp.resize(newEnd - temp.begin());
            aggAdjacency[aggA].swap(temp);
            temp.clear();
            aggAdjacency[aggB].clear();
            
            if (fillSpot) {
                // Finding an aggregate to shift into the empty spot
                if (aggB == aggMap.size() - 1) {
                    aggMap.pop_back();
                    aggAdjacency.pop_back();
                }
                else {
                    // Move the last aggregate to fill
                    int aggToMove = aggMap.size() - 1;

                    // Swap out the node list
                    aggMap[aggB].swap(aggMap[aggToMove]);
                    aggMap.pop_back();

                    // Mark nodes in aggregation
                    for (int nIt = 0; nIt < aggMap[aggB].size(); nIt++)
                        aggregation[aggMap[aggB][nIt]] = aggB;

                    // Swap out the adjacency list
                    aggAdjacency[aggB].swap(aggAdjacency[aggToMove]);
                    aggAdjacency.pop_back();

                    // Fix neighbor's adjacency lists
                    for (int nIt = 0; nIt < aggAdjacency[aggB].size(); nIt++) {
                        // The old Id has to be last in the list
                        int neighborAgg = aggAdjacency[aggB][nIt];
                        aggAdjacency[neighborAgg].back() = aggB;
                        sort(aggAdjacency[neighborAgg].begin(),
                                aggAdjacency[neighborAgg].end());
                    }

                    partSizes[aggB] = partSizes[aggToMove];
                    if (nodeWeights.size() > 0)
                        weightedSizes[aggB] = weightedSizes[aggToMove];
                }

                // Resize the sizes arrays
                partSizes.pop_back();
                if (nodeWeights.size() > 0)
                    weightedSizes.pop_back();
            }
            ValidateArraySizes("At end of MergeAggregates");
            return aggA;
        }
        void MergeSplitConditionerCPU::MakeSplitsDirect(bool force) {
            if (verbose) {
                printf("Beginning MakeSplitsDirect\n");
            }
            
            // Get the appropriate sizes
            IntVector_h &sizes = nodeWeights.size() > 0 ? weightedSizes : partSizes;
            
            // For every aggregate see if it should split
            int aggId = 0;
            while (aggId < aggAdjacency.size()) {
                // Getting size of current aggregate
                int currentSize = sizes[aggId];
                
                if (verbose) {
                    printf("Checking if aggregate %d of size %d should split\n",
                            aggId, currentSize);
                    
                }
                
                // If aggregate too big split
                if (currentSize > maxSize && (currentSize > minSize * 2 || force)) {
                    if (verbose) {
                        printf("Aggregate %d of size %d is being split.\n", 
                                aggId, sizes[aggId]);
                    }
                    
                    // Creating empty entry for new aggregate
                    partSizes.resize(partSizes.size() + 1, 0);
                    if (nodeWeights.size() > 0)
                        weightedSizes.resize(weightedSizes.size() + 1, 0);
                    aggMap.resize(aggMap.size() + 1);
                    aggAdjacency.resize(aggAdjacency.size() + 1);
                    
                    SplitAggregate(aggId, aggMap.size() - 1);
                    splits++;
                    
                    if (verbose) {
                        printf("Split into aggregate %d of size %d and %d of size %d\n",
                                aggId, sizes[aggId], aggMap.size() -1, sizes.back());
                    }
                }
                aggId++;
            }
        }
        void MergeSplitConditionerCPU::SplitAggregate(int agg, int newAgg) {
            if (verbose) {
                printf("SplitAggregate called to split aggregate %d into %d and %d\n",
                        agg, agg, newAgg);
                stringstream s1;
                s1 << "Node list of aggregate " << agg;
                Display::Print(aggMap[agg], s1.str());
            }
            
            if (agg == newAgg) {
                printf("Problem! SplitAggregate called with agg=%d and newAgg=%d\n",
                        agg, newAgg);
                int t;
                cin >> t;
            }
            
            UnlinkAggregate(agg);
            
            // Getting the graph of the aggregate
            vector<vector<int> > *am = Aggregation::GetAggregateGraph(*graph, aggMap[agg]);
            vector<vector<int> > &aggGraph = *am;

            // Getting the node weights if needed
            IntVector_h weights;
            if (nodeWeights.size() > 0) {
                weights.resize(aggMap[agg].size());
                for (int i = 0; i < weights.size(); i++)
                    weights[i] = nodeWeights[aggMap[agg][i]];
            }
            
            // Finding the root points:
            int rootA = Aggregation::FindFarthestNode(aggGraph, 0);
            int rootB = Aggregation::FindFarthestNode(aggGraph, rootA);

            // Keep track of the allocated nodes
            vector<int> allocated(aggGraph.size(), -1);
            
            // Storing the Id's of the aggregates
            vector<int> aggIds;
            aggIds.push_back(agg);
            aggIds.push_back(newAgg);
            
            // Queues of possible candidates 
            vector<queue<int> > filler(2);
            filler[0].push(rootA);
            filler[1].push(rootB);
            
            // Nodelists for each aggregate
            vector<vector<int> > nodeLists(2);
            
            // Sizes of each aggregate
            vector<int> aggSizes(2,0);
            
            // Count of allocated nodes
            int done = 0;
            
            // 0 or 1 for which aggregate is looking to allocate
            int activeAgg = 0;
            int inactiveAgg = 1;
            
            while (done < aggGraph.size()) {
                // Check if there is any possibilities
                if (!filler[activeAgg].empty()) {
                    // Checking the next candidate on the queue
                    int node = filler[activeAgg].front();
                    filler[activeAgg].pop();
                    
                    // If node is not allocated take it
                    if (allocated[node] == -1) {
                        // Mark node as allocated
                        allocated[node] = 1;
                        // Add to activeAgg's nodelist
                        nodeLists[activeAgg].push_back(aggMap[agg][node]);
                        // Mark in aggregation
                        aggregation[aggMap[agg][node]] = aggIds[activeAgg];
                        // Increment count of done nodes
                        done++;
                        
                        if (verbose) {
                            printf("Allocated local node %d global node %d to %d. Now %d nodes are done\n",
                                    node, aggMap[agg][node], aggIds[activeAgg], done);
                        }
                        
                        // Increment size
                        if (weights.size() > 0)
                            aggSizes[activeAgg] += weights[node];
                        else
                            aggSizes[activeAgg]++;
                        
                        // Add unallocated neighbors to queue
                        for (int nIt = 0; nIt < aggGraph[node].size(); nIt++) {
                            int neighbor = aggGraph[node][nIt];
                            if (allocated[neighbor] == -1)
                                filler[activeAgg].push(neighbor);
                        }
                        // Check to see if activeAgg should change
                        if (aggSizes[activeAgg] > aggSizes[inactiveAgg]
                                && !filler[inactiveAgg].empty()) {
                            activeAgg = (activeAgg + 1) % 2;
                            inactiveAgg = (inactiveAgg + 1) % 2;
                        }
                    }
                }
                else {
                    activeAgg = (activeAgg + 1) % 2;
                    inactiveAgg = (inactiveAgg + 1) % 2;
                }
            }
            
            // Sort the generated nodelists
            sort(nodeLists[0].begin(), nodeLists[0].end());
            sort(nodeLists[1].begin(), nodeLists[1].end());
            
            // Swap the generated nodelists into the aggMap
            nodeLists[0].swap(aggMap[agg]);
            nodeLists[1].swap(aggMap[newAgg]);
            
            if (verbose) {
                stringstream s2;
                s2 << "AggMap for " << agg;
                Display::Print(aggMap[agg], s2.str());

                stringstream s3;
                s3 << "AggMap for " << newAgg;
                Display::Print(aggMap[newAgg], s3.str());
            }
            
            // Link in the split aggregates
            LinkAggregate(agg);
            LinkAggregate(newAgg);
            
            // Fix sizes
            FixSizesFromAggMap(agg);
            FixSizesFromAggMap(newAgg);
            
            // Clean up temp stuff
            delete am;
            ValidateArraySizes("At end of SplitAggregate");
        }
        void MergeSplitConditionerCPU::MakeMergeSplits(int desiredSize) {
            // Get the appropriate sizes
            IntVector_h &sizes = nodeWeights.size() > 0 ? weightedSizes : partSizes;
            
            // For every aggregate see if it should merge-split
            int aggId = 0;
            while (aggId < aggAdjacency.size()) {
                // Getting size of current aggregate
                int currentSize = sizes[aggId];
                
                // Tracking the best seen merge
                int bestMerge = -1;
                int smallestDifference = INT_MAX;
                
                // If aggregate too small or too big check for merge splits:
                if (currentSize < minSize || currentSize > maxSize) {
                    // Look at neighboring aggregates
                    for (int nIt = 0; nIt < aggAdjacency[aggId].size(); nIt++) {
                        int neighborAgg = aggAdjacency[aggId][nIt];
                        int neighborSize = sizes[neighborAgg];
                        int mergedSize = currentSize + neighborSize;
                        int difference = mergedSize > (desiredSize * 2) ? 
                                            mergedSize - (desiredSize * 2) : 
                                            (desiredSize * 2) - mergedSize;
                        if (mergedSize >= (minSize * 2) && mergedSize <= (maxSize * 2)) {
                            if (difference < smallestDifference) {
                                smallestDifference = difference;
                                bestMerge = neighborAgg;
                            }
                        }
                    }
                    if (bestMerge != -1) {
                        if (verbose) {
                            printf("Aggregate %d of size %d found neighbor %d of size %d to merge-split with.\n",
                                    aggId, currentSize, bestMerge, sizes[bestMerge]);
                        }
                        
                        // Merging the aggregates and then splitting back into
                        // the same two ID's
                        int lowId = MergeAggregates(aggId, bestMerge, false);
                        if (lowId == aggId)
                            SplitAggregate(aggId, bestMerge);
                        else
                            SplitAggregate(bestMerge, aggId);
                        
                        if (verbose) {
                            printf("After merge-split aggregate %d has size %d and aggregate %d has size %d\n",
                                    aggId, sizes[aggId], bestMerge, sizes[bestMerge]);
                        }
                        ValidateArraySizes("After doing a merge-split");
                        mergeSplits++;
                    }
                    else {
                        if (verbose) {
                            printf("No merge-split found for aggregate %d of size %d\n",
                                    aggId, currentSize);
                        }
                    }
                }
                aggId++;
            }
        }
        void MergeSplitConditionerCPU::UnlinkAggregate(int aggId) {
            // Remove aggId from neighbors adjacency lists
            for (int i = 0; i < aggAdjacency[aggId].size(); i++) {
                int neighborAgg = aggAdjacency[aggId][i];
                remove(aggAdjacency[neighborAgg].begin(), 
                        aggAdjacency[neighborAgg].end(), 
                        aggId);
                aggAdjacency[neighborAgg].pop_back();
            }
            
            // Clear adjacency for aggId
            aggAdjacency[aggId].clear();
            ValidateArraySizes("At end of UnlinkAggregate");
        }
        void MergeSplitConditionerCPU::FixSizesFromAggMap(int aggId) {
            partSizes[aggId] = aggMap[aggId].size();
            if (nodeWeights.size() > 0) {
                int weight = 0;
                for (int i = 0; i < aggMap[aggId].size(); i++)
                    weight += nodeWeights[aggMap[aggId][i]];
                weightedSizes[aggId] = weight;
            }
            ValidateArraySizes("At end of FixSizesFromAggMap");
        }
        void MergeSplitConditionerCPU::LinkAggregate(int aggId) {
            // Check aggregate of all neighbors of nodes in aggId
            for (int i = 0; i < aggMap[aggId].size(); i++) {
                int node = aggMap[aggId][i];
                for (int* nIt = graph->nStart(node); 
                        nIt != graph->nEnd(node); 
                        nIt++) {
                    int neighborAgg = aggregation[*nIt];
                    if (neighborAgg != aggId) {
                        aggAdjacency[aggId].push_back(neighborAgg);
                        
                        // Insert aggId into neighbor's adjacency if needed
                        if (!binary_search(aggAdjacency[neighborAgg].begin(),
                                        aggAdjacency[neighborAgg].end(),
                                        aggId)) {
                            aggAdjacency[neighborAgg].push_back(aggId);
                            sort(aggAdjacency[neighborAgg].begin(),
                                    aggAdjacency[neighborAgg].end());
                        }
                    }
                }
            }
            
            // Sort and remove duplicates
            sort(aggAdjacency[aggId].begin(), aggAdjacency[aggId].end());
            vector<int>::iterator newEnd = unique(aggAdjacency[aggId].begin(),
                                                aggAdjacency[aggId].end());
            aggAdjacency[aggId].resize(newEnd - aggAdjacency[aggId].begin());
            ValidateArraySizes("At end of LinkAggregate");
        }
        void MergeSplitConditionerCPU::FillAggAdjacency() {
            // Clearing anything there out first
            for (int i = 0; i < aggAdjacency.size(); i++)
                aggAdjacency[i].clear();
            aggAdjacency.clear();
            
            // Going through the aggregation vector to fill
            for (int node = 0; node < graph->Size(); node++) {
                int startAgg = aggregation[node];
                for (int *nIt = graph->nStart(node); nIt != graph->nEnd(node); nIt++) {
                    int endAgg = aggregation[*nIt];
                    
                    // If this is an edge between two aggregates add to
                    // the induced graph.
                    if (startAgg != endAgg && startAgg < endAgg) {
                        // Making sure that there are entries in temp
                        if (endAgg >= aggAdjacency.size())
                            aggAdjacency.resize(endAgg + 1);
                        
                        // Adding edge entries
                        if (aggAdjacency[startAgg].size() == 0 || 
                                !(std::binary_search(aggAdjacency[startAgg].begin(), 
                                                    aggAdjacency[startAgg].end(), 
                                                    endAgg))) {
                            aggAdjacency[startAgg].push_back(endAgg);
                            std::sort(aggAdjacency[startAgg].begin(), 
                                    aggAdjacency[startAgg].end());
                        }
                        if (aggAdjacency[endAgg].size() == 0 || 
                                !(std::binary_search(aggAdjacency[endAgg].begin(), 
                                                    aggAdjacency[endAgg].end(), 
                                                    startAgg))) {
                            aggAdjacency[endAgg].push_back(startAgg);
                            std::sort(aggAdjacency[endAgg].begin(), 
                                    aggAdjacency[endAgg].end());
                        }
                    } 
                }
            }            
        }
        void MergeSplitConditionerCPU::FillAggMap() {
            // Clearing anything there out first
            for (int i = 0; i < aggMap.size(); i++)
                aggMap[i].clear();
            aggMap.clear();
            
            // Going through the aggregation vector to fill
            for (int node = 0; node < graph->Size(); node++) {
                int aggId = aggregation[node];
                if (aggMap.size() <= aggId)
                    aggMap.resize(aggId + 1);
                aggMap[aggId].push_back(node);
            }

            // Make sure each map is sorted
            for (int i = 0; i < aggMap.size(); i++)
                sort(aggMap[i].begin(), aggMap[i].end());
        }
        void MergeSplitConditionerCPU::ValidateAggAdjacency() {
            // Move current into new spot
            vector<vector<int> > temp(aggAdjacency.size());
            for (int i = 0; i < temp.size(); i++)
                temp[i].swap(aggAdjacency[i]);
            
            // Regenerate member
            FillAggAdjacency();
            
            // Compare
            if (Compare::AreEqual(temp, aggAdjacency, true))
                printf("AggAdjacency validated.\n");
            else {
                printf("Failed to validate AggAdjacency.\n");
                int t;
                cin >> t;
            }
            for (int i = 0; i < temp.size(); i++)
                temp[i].clear();
        }
        void MergeSplitConditionerCPU::ValidateAggMap() {
            // Move current into new spot
            vector<vector<int> > temp(aggMap.size());
            for (int i = 0; i < temp.size(); i++)
                temp[i].swap(aggMap[i]);
            
            // Regenerate member
            FillAggMap();
            
            // Compare
            if (Compare::AreEqual(temp, aggMap, true))
                printf("AggMap validated.\n");
            else {
                printf("Failed to validate AggMap.\n");
                int t;
                cin >> t;
            }
            for (int i = 0; i < temp.size(); i++)
                temp[i].clear();
        }
        void MergeSplitConditionerCPU::ValidatePartSizes() {
            bool failed = false;
            IntVector_h *ps = Aggregation::GetPartSizes(aggregation);
            if (Compare::AreEqual(*ps, partSizes, true))
                printf("PartSizes validates.\n");
            else {
                printf("PartSizes does not validate.\n");
                failed = true;
            }
            if (nodeWeights.size() > 0) {
                IntVector_h* ws = Aggregation::GetPartSizes(aggregation, nodeWeights);
                if (Compare::AreEqual(*ws, weightedSizes, true))
                    printf("WeightedSizes validates.\n");
                else {
                    printf("WeightedSizes does not validate.\n");
                    failed = true;
                }    
            }
            if (failed) {
                int d;
                cin >> d;
            }
        }
        void MergeSplitConditionerCPU::ValidateArraySizes(string message) {
            bool error = nodeWeights.size() > 0 ?
                        aggAdjacency.size() != aggMap.size() ||
                            aggMap.size() != partSizes.size() ||
                            partSizes.size() != weightedSizes.size() :
                        aggAdjacency.size() != aggMap.size() ||
                            aggMap.size() != partSizes.size();
            if (error) {
                printf("Error with array sizes! %s\n", message.c_str());
                printf("\taggAdjacency.size()=%d aggMap.size()=%d partSizes.size()=%d weightedSizes.size()=%d\n",
                        aggAdjacency.size(), aggMap.size(), partSizes.size(), weightedSizes.size());
                int t;
                cin >> t;
            }
        }
    }
}
