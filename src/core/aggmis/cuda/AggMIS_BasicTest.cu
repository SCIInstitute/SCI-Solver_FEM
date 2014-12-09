#include <fstream>
#include <iostream>
#include <sstream>
#include "AggMIS_FileIO.h"
#include "AggMIS_MIS_GPU.h"
#include "AggMIS_Aggregation_GPU.h"
#include "AggMIS_MergeSplitConditioner.h"
#include "AggMIS_MergeSplitConditioner_CPU.h"
#include "AggMIS_Metrics.h"
#include "AggMIS_Aggregation_CPU.h"
#include "AggMIS_MIS_CPU.h"
#include "Helper.h"
#include "Logger.h"
//#include "DataRecorder.h"
using namespace std;
using namespace AggMIS;
using namespace Types;
int main(int argc, char** argv)
{
//    DataRecorder::SetFile("BasicTestData.rec");
    if (argc == 3)
    {
//        DataRecorder::Add("Mesh", argv[1]);
        // Read in the input file
        Graph_h *graph = AggMIS::FileIO::GetGraphFromFile_Auto(argv[1]);
        
        
        printf("Read in file successfully. There are %d nodes.\n", graph->Size());
        
        // Running device code
        if (argv[2][0] == 'd') {
            int deviceCount;
            cudaGetDeviceCount(&deviceCount);
            int device;
            for(device = 0; device < deviceCount; ++device)
            {
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, device);
                size_t totalMemory = deviceProp.totalGlobalMem;
                int totalMB = totalMemory / 1000000;
                printf("Device %d (%s) has compute capability %d.%d. and %dMb global memory.\n",
                       device, deviceProp.name, deviceProp.major, deviceProp.minor, totalMB);
            }

            cudaSetDevice(1);
            if(cudaDeviceReset()!=cudaSuccess)
                exit(0);
            else {
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, 1);
                size_t totalMemory = deviceProp.totalGlobalMem;
                int totalMB = totalMemory / 1000000;
//                DataRecorder::Add("CUDA Device", deviceProp.name);
//                DataRecorder::Add("Device Memory (MB)", totalMB);
            }
            Graph_d graph_d(*graph);
            
            printf("Getting a randomized MIS:\n");
            
            // Get an MIS aggregation of the input mesh:
            IntVector_d *mis = MIS::RandomizedMIS(2, graph_d);

            printf("Found GPU MIS of graph\n");

            // Aggregate using the mis as roots
            IntVector_d *aggregation = Aggregation::AggregateToNearest(graph_d, *mis);
            printf("Aggregated using GPU MIS.\n");

            // Checking aggregation is valid
            if (Aggregation::IsValidAggregation(graph_d, *aggregation, true))
                printf("Aggregation is valid.\n");
            else
                printf("Aggregation is not valid!\n");

            // Conditioning the aggregation
            MergeSplitGPU::MergeSplitConditionerGPU conditioner(graph_d, *aggregation);
            conditioner.SetSizeBounds(10,20);
            conditioner.SetVerbose(true);
            conditioner.InteractiveConsole("Starting GPU Conditioning");
//            DataRecorder::Add("Fine MinSize", 10);
//            DataRecorder::Add("Fine MaxSize", 20);

            // Checking conditioning is valid
            if (Aggregation::IsValidAggregation(graph_d, *(conditioner.GetAggregation()), true))
                printf("Aggregation is valid.\n");
            else
                printf("Aggregation is not valid!\n");
            IntVector_d partWeights;
            GraphHelpers::getPartSizes(*(conditioner.GetAggregation()),
                                        partWeights);
            Graph_d *inducedGraph = GraphHelpers::GetInducedGraph(graph_d, 
                                                        *(conditioner.GetAggregation()));
            IntVector_d *inducedMis = MIS::RandomizedMIS(2, *inducedGraph);
            IntVector_d *inducedAggregation = Aggregation::AggregateToNearest(*inducedGraph,
                                                                *inducedMis);
            
            // Checking conditioning is valid
            if (Aggregation::IsValidAggregation(*inducedGraph, *inducedAggregation, true))
                printf("Aggregation is valid.\n");
            else
                printf("Aggregation is not valid!\n");
            
            MergeSplitGPU::MergeSplitConditionerGPU inducedConditioner(*inducedGraph, 
                                                            *inducedAggregation);
            inducedConditioner.SetNodeWeights(partWeights);
            inducedConditioner.SetSizeBounds(300, 450);
            inducedConditioner.SetVerbose(true);
            inducedConditioner.InteractiveConsole("Weighted conditioning test.");
//            DataRecorder::Add("Coarse MinSize", 10);
//            DataRecorder::Add("Coarse MaxSize", 20);
            
            // Checking conditioning is valid
            if (Aggregation::IsValidAggregation(*inducedGraph, *(inducedConditioner.GetAggregation()), true))
                printf("Aggregation is valid.\n");
            else
                printf("Aggregation is not valid!\n");
        }
        
        // Running host code
        if (argv[2][0] == 'h') {
            IntVector_h *mis_cpu = MIS::FloodFillMIS(2, *graph);
            printf("Got CPU MIS\n");
            IntVector_h *agg_cpu = Aggregation::AggregateToNearest(*graph, *mis_cpu);
            printf("Got CPU aggregation\n");
            if (Aggregation::IsValidAggregation(*graph, *agg_cpu, true))
                printf("CPU aggregation is valid!\n");
            else
                printf("CPU aggregation is invalid!\n");

            MergeSplitCPU::MergeSplitConditionerCPU cond(*graph, *agg_cpu);
            cond.SetSizeBounds(10,20);
            cond.InteractiveConsole("Starting to condition CPU\n");

            printf("Finished with fine aggregation.\n");
            
            // Getting the aggregation out of the conditioner
            agg_cpu->swap(*(cond.GetAggregation()));
            IntVector_h *ps_cpu = Aggregation::GetPartSizes(*agg_cpu);
            Graph_h *inducedGraph_cpu = GraphHelpers::GetInducedGraph(*graph, *agg_cpu);
            IntVector_h *mis_cpu2 = MIS::FloodFillMIS(2, *inducedGraph_cpu);
            IntVector_h *agg_cpu2 = Aggregation::AggregateToNearest(*inducedGraph_cpu, *mis_cpu2);
            MergeSplitCPU::MergeSplitConditionerCPU condWeighted(*inducedGraph_cpu, *agg_cpu2);
            condWeighted.SetNodeWeights(*ps_cpu);
            condWeighted.SetSizeBounds(300, 450);
            condWeighted.InteractiveConsole("Starting weighted conditioning.");            
        }
        printf("Finished!\n");
    }
}



