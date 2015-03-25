#include <smoothedMG/aggregators/misHelpers.h>

__global__ void Generate_Randoms_Kernel(int size, int iterations, unsigned int *randoms, unsigned int *seeds)
{
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

__global__ void First_Initialize_Kernel(int size, unsigned int *randoms, int *bestSeen, int *origin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {   
        // Set the origin to be self
        origin[idx] = idx;
        
        // Set the bestSeen value to be random
        bestSeen[idx] = randoms[idx] % 1000000;
    } 
}

__global__ void Initialize_Kernel(int size, unsigned int *randoms, int *bestSeen, int *origin, int *mis, int *incomplete)
{
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

__global__ void Iterate_Kernel(int size, int *originIn, int *originOut, int *bestSeenIn, int *bestSeenOut, int *adjIndexes, int *adjacency)
{
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
    
__global__ void Final_Iterate_Kernel(int size, int *originIn, int *originOut, int *bestSeenIn, int *bestSeenOut, int *adjIndexes, int *adjacency, int *mis, int *incomplete)
{
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

void misHelpers::randomizedMIS(IdxVector_d adjIndexes, IdxVector_d adjacency, IdxVector_d &mis, int k)
{   
    // Setting to prefer the cache:
    cudaFuncSetCacheConfig(Initialize_Kernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(Iterate_Kernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(Final_Iterate_Kernel, cudaFuncCachePreferL1);
    
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    
    int size = adjIndexes.size() - 1;
    mis.resize(size);
    thrust::fill(mis.begin(),mis.end(), -1);
    
    IntVector_d incomplete(1);            // This is a single value that will be marked with 1 by initialize kernel if there are unallocated nodes    //IdxVector_d misIn(size, -1);          // The current MIS assignments 1 = in MIS, 0 = not in MIS, -1 = undetermined 
    cusp::array1d<unsigned int, cusp::device_memory> randoms(size);           // Set of random values generated by each threads random generator                      
    IntVector_d bestSeenIn(size);         // Holds the highest value seen so far propogated through neigbhors each iteration
    IntVector_d bestSeenOut(size);        // Holds the highest value seen so far propogated through neigbhors each iteration
    IntVector_d originIn(size);           // The index where the best seen value originated     
    IntVector_d originOut(size);          // The index where the best seen value originated  
    cusp::array1d<unsigned int, cusp::device_memory> seeds(32768);              // Host side vector of initial random values
    
    // Getting raw pointers:
    int *incomplete_d = thrust::raw_pointer_cast(&incomplete[0]);
    int *misIn_d = thrust::raw_pointer_cast(&mis[0]);
    unsigned int *randoms_d = thrust::raw_pointer_cast(&randoms[0]);
    unsigned int *seeds_d = thrust::raw_pointer_cast(&seeds[0]);
    int *bestSeenIn_d = thrust::raw_pointer_cast(&bestSeenIn[0]);
    int *bestSeenOut_d = thrust::raw_pointer_cast(&bestSeenOut[0]);
    int *originIn_d = thrust::raw_pointer_cast(&originIn[0]);
    int *originOut_d = thrust::raw_pointer_cast(&originOut[0]);
    int *adjIndexes_d = thrust::raw_pointer_cast(&(adjIndexes[0]));
    int *adjacency_d = thrust::raw_pointer_cast(&(adjacency[0]));
    
    // Setting up for kernel launches
    int blockSize = 256;
    int nBlocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);
    
    
    // Seeding the randoms array:    
    srand(time(NULL));
    unsigned *seeds_h = new unsigned[32768];
    for (int i = 0; i < 32768; i++)
        seeds_h[i] = (unsigned)rand();
    thrust::copy(seeds_h, seeds_h + 32768, seeds.begin());
    int iterations = (size + 32767) / 32768;
    Generate_Randoms_Kernel <<<128, 256>>> (size, iterations, randoms_d, seeds_d);
    
    // Running the initialize kernel:
    First_Initialize_Kernel <<< nBlocks, blockSize >>> (size, randoms_d, bestSeenIn_d, originIn_d);
    
    // Running the iteration kernel k times swapping in and out for each iteration
    for (int i = 0; i < k; i++)
    {
        if (i < k - 1)
        {
            Iterate_Kernel <<< nBlocks, blockSize >>> (size, originIn_d, originOut_d, bestSeenIn_d, bestSeenOut_d, adjIndexes_d, adjacency_d);
        }
        else 
        {       
            Final_Iterate_Kernel <<< nBlocks, blockSize >>> (size, originIn_d, originOut_d, bestSeenIn_d, bestSeenOut_d, adjIndexes_d, adjacency_d, misIn_d, incomplete_d);
        }  
        
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
        Initialize_Kernel <<< nBlocks, blockSize >>> (size, randoms_d, bestSeenIn_d, originIn_d, misIn_d, incomplete_d);
        
        // Running the iteration kernel k times swapping in and out for each iteration
        for (int i = 0; i < k; i++)
        {
            if (i < k - 1)
            {
                Iterate_Kernel <<< nBlocks, blockSize >>> (size, originIn_d, originOut_d, bestSeenIn_d, bestSeenOut_d, adjIndexes_d, adjacency_d);
            }
            else 
            {       
                Final_Iterate_Kernel <<< nBlocks, blockSize >>> (size, originIn_d, originOut_d, bestSeenIn_d, bestSeenOut_d, adjIndexes_d, adjacency_d, misIn_d, incomplete_d);
            }


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
}
