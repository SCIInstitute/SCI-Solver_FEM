/* 
 * File:   AggMIS_MIS_GPU.cu
 * Author: T. James Lewis
 *
 * Created on April 17, 2013, 12:49 PM
 */
#include "AggMIS_MIS_GPU.h"
#include "AggMIS_Types.h"
namespace AggMIS {
    namespace MIS {
        using namespace AggMIS::Types;
        using namespace std;
        namespace Kernels {
            __global__ void GenerateRandoms(int size, 
                                            int iterations, 
                                            unsigned int *randoms, 
                                            unsigned int *seeds) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int z = seeds[idx];
                int offset = idx;
                int step = 32768;

                for (int i = 0; i < iterations; i++)
                {
                    if (offset < size)
                    {
                        unsigned int b = (((z << 13) ^ z) >> 19);
                        z = (((z & UINT_MAX) << 12) ^ b);
                        randoms[offset] = z;
                        offset += step;
                    }
                }
            }
            __global__ void PreInitialize(int size, 
                                            unsigned int *randoms, 
                                            int *bestSeen, 
                                            int *origin, 
                                            int *mis) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size)
                {   
                    // Set the origin to be self
                    origin[idx] = idx;

                    // Set MIS to be -1;
                    mis[idx] = -1;

                    // Set the bestSeen value to be random
                    bestSeen[idx] = randoms[idx] % 1000000;
                } 
            }
            __global__ void Initialize(int size, 
                                            unsigned int *randoms, 
                                            int *bestSeen, 
                                            int *origin, 
                                            int *mis, 
                                            int *incomplete) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size)
                {
                    // Taustep is performed with S1=13, S2=19, S3=12, and M=UINT_MAX coded into kernel
                    unsigned int z = randoms[idx];
                    unsigned int b = (((z << 13) ^ z) >> 19);
                    z = (((z & UINT_MAX) << 12) ^ b);

                    // Set the origin to be self
                    origin[idx] = idx;

                    // Set the bestSeen value to be either random from 0-1000000 or 1000001 if in MIS
                    int status = mis[idx];
                    int value = 0;
                    if (status == 1)
                        value = 1000001;

                    bestSeen[idx] = (mis[idx] == -1) ? (z % 1000000) : value;

                    // Write out new random value for seeding
                    randoms[idx] = z;
                } 

                // Reset incomplete value
                if (idx == 0)
                    incomplete[0] = 0;
            }
            __global__ void Iterate(int size, 
                                            int *originIn, 
                                            int *originOut, 
                                            int *bestSeenIn, 
                                            int *bestSeenOut, 
                                            int *adjIndexes, 
                                            int *adjacency) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size)
                {
                    int bestSeen = bestSeenIn[idx];
                    int origin = originIn[idx];
                    if (bestSeen < 1000001)
                    {
                        int start = adjIndexes[idx];
                        int end = adjIndexes[idx + 1];

                        // Look at all the neighbors and take best values:
                        for (int i = start; i < end; i++)
                        {
                            int neighbor = adjacency[i];
                            int challenger = bestSeenIn[neighbor];
                            int challengerOrigin = originIn[neighbor];

                            if (challenger > 0 && challenger == bestSeen && challengerOrigin > origin)
                            {
                                origin = challengerOrigin;
                            }


                            if (challenger > bestSeen)
                            {
                                bestSeen = challenger;
                                origin = challengerOrigin;
                            }
                        }
                    }

                    // Write out the best values found
                    bestSeenOut[idx] = bestSeen;
                    originOut[idx] = origin;
                }  
            }
            __global__ void Finalize(int size, 
                                            int *originIn, 
                                            int *originOut, 
                                            int *bestSeenIn, 
                                            int *bestSeenOut, 
                                            int *adjIndexes, 
                                            int *adjacency, 
                                            int *mis, 
                                            int *incomplete) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size)
                {
                    int bestSeen = bestSeenIn[idx];
                    int origin = originIn[idx];
                    if (bestSeen < 1000001)
                    {
                        int start = adjIndexes[idx];
                        int end = adjIndexes[idx + 1];

                        // Look at all the neighbors and take best values:
                        for (int i = start; i < end; i++)
                        {
                            int neighbor = adjacency[i];
                            unsigned int challenger = bestSeenIn[neighbor];
                            int challengerOrigin = originIn[neighbor];

                            if (challenger > 0 && challenger == bestSeen && challengerOrigin > origin)
                            {
                                origin = challengerOrigin;
                            }

                            if (challenger > bestSeen)
                            {
                                bestSeen = challenger;
                                origin = challengerOrigin;
                            }
                        }
                    }

                    // Write new MIS status 
                    int misStatus = -1;
                    if (origin == idx)
                        misStatus = 1;
                    else if (bestSeen == 1000001)
                        misStatus = 0;

                    mis[idx] = misStatus;


                    // If this node is still unassigned mark
                    if (misStatus == -1)
                    {
                        incomplete[0] = 1;
                    }
                }    
            }
        }
        IntVector_d* RandomizedMIS(int k, Graph_d &graph) {
            // Setting to prefer the cache:
            cudaFuncSetCacheConfig(Kernels::Initialize, cudaFuncCachePreferL1);
            cudaFuncSetCacheConfig(Kernels::Iterate, cudaFuncCachePreferL1);
            cudaFuncSetCacheConfig(Kernels::Finalize, cudaFuncCachePreferL1);

            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);

            IntVector_d incomplete(1);                  // This is a single value that will be marked with 1 by initialize kernel if there are unallocated nodes
            IntVector_d *misIn = new IntVector_d(graph.Size());              // The current MIS assignments 1 = in MIS, 0 = not in MIS, -1 = undetermined
            UIntVector_d randoms(graph.Size());           // Set of random values generated by each threads random generator                      
            IntVector_d bestSeenIn(graph.Size());         // Holds the highest value seen so far propogated through neigbhors each iteration
            IntVector_d bestSeenOut(graph.Size());        // Holds the highest value seen so far propogated through neigbhors each iteration
            IntVector_d originIn(graph.Size());           // The index where the best seen value originated     
            IntVector_d originOut(graph.Size());          // The index where the best seen value originated  
            UIntVector_d seeds(32768);                  // Stores the first few seeds for the random generation process

            // Getting raw pointers:
            int *incomplete_d = thrust::raw_pointer_cast(&incomplete[0]);
            int *mis_d = thrust::raw_pointer_cast(misIn->data());
            unsigned *randoms_d = thrust::raw_pointer_cast(&randoms[0]);
            unsigned *seeds_d = thrust::raw_pointer_cast(&seeds[0]);
            int *bestSeenIn_d = thrust::raw_pointer_cast(&bestSeenIn[0]);
            int *bestSeenOut_d = thrust::raw_pointer_cast(&bestSeenOut[0]);
            int *originIn_d = thrust::raw_pointer_cast(&originIn[0]);
            int *originOut_d = thrust::raw_pointer_cast(&originOut[0]);
            int *adjIndexes_d = thrust::raw_pointer_cast(graph.indices->data());
            int *adjacency_d = thrust::raw_pointer_cast(graph.adjacency->data());

            // Setting up for kernel launches
            int blockSize = 512;
            int nBlocks = graph.Size() / blockSize + (graph.Size() % blockSize == 0 ? 0 : 1);

            // Seeding the randoms array:
            srand(time(NULL));
            unsigned *seeds_h = new unsigned[32768];
            for (int i = 0; i < 32768; i++)
                seeds_h[i] = (unsigned)rand();

            thrust::copy(seeds_h, seeds_h + 32768, seeds.begin());
            int iterations = (graph.Size() + 32767) / 32768;

            Kernels::GenerateRandoms <<<128, 256>>> (graph.Size(), iterations, randoms_d, seeds_d);         

            // Running the initialize kernel:
            Kernels::PreInitialize <<< nBlocks, blockSize >>> (graph.Size(), randoms_d, bestSeenIn_d, originIn_d, mis_d);

            // Running the iteration kernel k times swapping in and out for each iteration
            for (int i = 0; i < k; i++)
            {
                if (i < k - 1)
                    Kernels::Iterate <<< nBlocks, blockSize >>> (graph.Size(), originIn_d, originOut_d, bestSeenIn_d, bestSeenOut_d, adjIndexes_d, adjacency_d);
                else 
                    Kernels::Finalize <<< nBlocks, blockSize >>> (graph.Size(), originIn_d, originOut_d, bestSeenIn_d, bestSeenOut_d, adjIndexes_d, adjacency_d, mis_d, incomplete_d);


                // Swap the pointers for the next iteration:
                int *temp = originIn_d;
                originIn_d = originOut_d;
                originOut_d = temp;

                int *temp2 = bestSeenIn_d;
                bestSeenIn_d = bestSeenOut_d;
                bestSeenOut_d = temp2;
            }

            // If not complete get new randoms and repeat  
            cudaThreadSynchronize();
            int unallocated = incomplete[0];

            while (unallocated == 1)
            {
                // Initialize kernel
                Kernels::Initialize <<< nBlocks, blockSize >>> (graph.Size(), randoms_d, bestSeenIn_d, originIn_d, mis_d, incomplete_d);                

                // Running the iteration kernel k times swapping in and out for each iteration
                for (int i = 0; i < k; i++)
                {
                    if (i < k - 1)
                        Kernels::Iterate <<< nBlocks, blockSize >>> (graph.Size(), originIn_d, originOut_d, bestSeenIn_d, bestSeenOut_d, adjIndexes_d, adjacency_d);
                    else 
                        Kernels::Finalize <<< nBlocks, blockSize >>> (graph.Size(), originIn_d, originOut_d, bestSeenIn_d, bestSeenOut_d, adjIndexes_d, adjacency_d, mis_d, incomplete_d);

                    // Swap the pointers for the next iteration:
                    int *temp = originIn_d;
                    originIn_d = originOut_d;
                    originOut_d = temp;

                    int *temp2 = bestSeenIn_d;
                    bestSeenIn_d = bestSeenOut_d;
                    bestSeenOut_d = temp2;
                }        

                // Checking if done:
                cudaThreadSynchronize();
                unallocated = incomplete[0];
            }

            // Deallocating temporary arrays:
            incomplete.resize(0);
            randoms.resize(0);
            bestSeenIn.resize(0);
            bestSeenOut.resize(0);
            originIn.resize(0);
            originOut.resize(0);

            // Returning the mis
            //return misIn;
            return misIn;
        }
        bool IsValidKMIS(IntVector_d& misIn, Graph_d& graphIn, int k, bool verbose) {
            // Copy to host data
            Graph_h graph(graphIn);
            IntVector_h mis(misIn);
            
            if (verbose)
                printf("Attempting to verify %d-mis properties\n", k);

            // Checking if the mis vector is the right size
            int misSize = mis.size();
            int gSize = graph.indices->size() - 1;
            if (misSize != gSize || misSize != graph.Size())
            {
                if (verbose)
                {
                    printf("The given vector is not the correct size to describe and MIS of the input graph! \n");
                    printf("\tMIS size: %d Graph Size: %d\n", misSize, gSize);
                }
                return false;
            }

            // Checking that at least one node is marked as in the MIS
            int count = 0;
            for (int i = 0; i < mis.size(); i++)
                if (mis[i] == 1)
                    count++;
            if (count == 0)
            {
                if (verbose)
                {
                    printf("No nodes are designated as in the MIS!\n");
//                    debugHelpers::printVector(mis, std::string("The MIS"));
                }
                return false;
            }

            // Checking that every node not in the MIS has a path to a root node
            // of less than k and that every node in the MIS does not
            for (int i = 0; i < mis.size(); i++)
            {
                vector< vector<int> > rings(k + 1);
                int distance = INT_MAX;
                rings[0].push_back(i);

                // Filling in the rings with breadth first search
                for (int j = 1; j < rings.size(); j++)            
                {
                    for (int root = 0; root < rings[j - 1].size(); root++)
                    {
                        int rootPoint = rings[j - 1][root];
                        int start = graph.indices->data()[rootPoint];
                        int end = graph.indices->data()[rootPoint + 1];
                        for (int nIt = start; nIt < end; nIt++)
                        {
                            int neighbor = graph.adjacency->data()[nIt];
                            bool visited = false;
                            for (int vLevel = 0; vLevel < j + 1; vLevel++)
                            {
                                for (int vIt = 0; vIt < rings[vLevel].size(); vIt++)
                                {
                                    if (rings[vLevel][vIt] == neighbor)
                                        visited = true;
                                }
                            }

                            if (!visited)
                            {
                                rings[j].push_back(neighbor);
                                if (mis[neighbor] == 1 && distance > j)
                                    distance = j;
                                if (mis[i] == 1 && mis[neighbor] == 1)
                                    printf("Found a %d-path from root node %d to root node %d\n", distance, neighbor, i);
                            }
                        }
                    }
                }

                // If this node is not in the MIS distance should be less than k
                if (mis[i] == 1 && distance <= k)
                {
                    if (verbose)
                    {
                        printf("Node %d is in the MIS but has a %d-path to another root node!\n", i, distance);
//                        debugHelpers::printVector(mis, std::string("The MIS:"));
//                        debugHelpers::printGraph(graph);
                    }

                    return false;
                }
                if (mis[i] == 0 && distance > k)
                {
                    if (verbose)
                        printf("Node %d is not in the MIS but has no conflicts with MIS nodes!\n", i);
//                        debugHelpers::printVector(mis, std::string("The MIS:"));
//                        debugHelpers::printGraph(graph);
                    return false;
                }

            }
            
            // Clean up temp memory
            mis.resize(0);
            
            // If we got this far the MIS must be valid (both maximal and independent)
            return true;
        }
    }
}