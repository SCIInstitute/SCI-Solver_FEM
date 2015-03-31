#include <smoothedMG/aggregators/mis.h>
#include <algorithm>
#include <queue>
using namespace std;


template <class Matrix, class Vector>
void MIS_Aggregator<Matrix, Vector>::computePermutation(TetMesh* meshPtr, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitionlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize)
{
	// Getting the neighbors for the mesh
    meshPtr->need_neighbors();
    
    // Vertex count:
    int nn = meshPtr->vertices.size();
    
    // Counting up edges for adjacency:
    int edgeCount = 0; 
    for (int vIt = 0; vIt < nn; vIt++)
    {
        edgeCount += meshPtr->neighbors[vIt].size();
    }
    
    //Allocating storage for array values of adjacency
    int* xadj = new int[nn+1];
    int* adjncy = new int[edgeCount];
    
    // filling the arrays:
    xadj[0] = 0;
    int idx = 0;

    // Populating the arrays:
    for(int i = 1; i < nn + 1; i++)
    {				
        xadj[i] = xadj[i-1] + meshPtr->neighbors[i-1].size();
        for(int j =0; j < meshPtr->neighbors[i-1].size(); j++)
        {
            adjncy[idx++] = meshPtr->neighbors[i-1][j];
        }
    }
    
    // Calling the other override to finish:
    computePermutation(nn, xadj, adjncy, permutation, ipermutation, aggregateIdx, partitionIdx, partitionlabel, nnout, xadjout, adjncyout, metissize);
    
    // Freeing up memories:
    delete [] xadj;
    delete [] adjncy;
	
}

template <class Matrix, class Vector>
void MIS_Aggregator<Matrix, Vector>::computePermutation(TriMesh* meshPtr,  IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx,int* partitionlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize)
{    
    // Getting the neighbors for the mesh
    meshPtr->need_neighbors();
    
    // Vertex count:
    int nn = meshPtr->vertices.size();
    
    // Counting up edges for adjacency:
    int edgeCount = 0; 
    for (int vIt = 0; vIt < nn; vIt++)
    {
        edgeCount += meshPtr->neighbors[vIt].size();
    }
    
    //Allocating storage for array values of adjacency
    int* xadj = new int[nn+1];
    int* adjncy = new int[edgeCount];

    
    
    // filling the arrays:
    xadj[0] = 0;
    int idx = 0;

    // Populating the arrays:
    for(int i = 1; i < nn + 1; i++)
    {				
        xadj[i] = xadj[i-1] + meshPtr->neighbors[i-1].size();
        for(int j =0; j < meshPtr->neighbors[i-1].size(); j++)
        {
            adjncy[idx++] = meshPtr->neighbors[i-1][j];
        }
    }
    
    // Calling the other override to finish:
    computePermutation(nn, xadj, adjncy, permutation, ipermutation, aggregateIdx, partitionIdx, partitionlabel, nnout, xadjout, adjncyout, metissize);
    
    // Freeing up memories:
    delete [] xadj;
    delete [] adjncy;
}


template <class Matrix, class Vector>
void MIS_Aggregator<Matrix, Vector>::computePermutation(int nn, int* xadj, int* adjncy, IdxVector_h &permutation, IdxVector_h &ipermutation, IdxVector_h &aggregateIdx, IdxVector_h &partitionIdx, int* partitionlabel, int* nnout, int* &xadjout, int* &adjncyout, int metissize)
{
    //Get block aggregation
    int nparts, edgecut;
    int *npart = (int*)malloc(nn * sizeof(int));
    nparts = (nn / metissize);
    if (nparts < 2)
        nparts = 2;
    int options[10], pnumflag=0, wgtflag=0;
    for(int i=0; i<10; i++)
        options[i] = 0;
    
    METIS_PartGraphKway(&nn, xadj, adjncy, NULL, NULL, &wgtflag, &pnumflag, &nparts, options, &edgecut, npart);
    
    // Finding partitions that have vertices assigned:
    vector<int> realParts;
    realParts.resize(nn);
    for (int i=0; i<nn; i++)
    {
        realParts[i] = npart[i];
    }
    sort(realParts.begin(), realParts.end());
    
    // Scanning for gaps in the sorted array
    vector<int> empties;
    if (realParts[0] > 0)
        for (int i = 0; i < realParts[0]; i++)
            empties.push_back(i);
    
    for (int i = 1; i < nn; i++)
    {
        if (realParts[i] != realParts[i-1])
        {
            if (realParts[i] > realParts[i-1] + 1)
            {
                for (int j = realParts[i] + 1; j < realParts[i]; j++)
                    empties.push_back(j);
            }
        }
    }
    
    // Re-numbering the npart array to close the gaps
    for (int i = 0; i < empties.size(); i++)
    {
        for (int j = 0; j < nn; j++)
        {
            if(npart[j] > empties[i])
                npart[j]--;
        }
        for (int j = i; j < empties.size(); j++)
        {
            empties[j]--;
        }
    }
    
    // Getting the actual partition count:
    int partCount = *(realParts.end() - 1) - empties.size() + 1;
    
    //Building a structure of sub-graphs to aggregate:
    vector< vector<int> > blocks;
    blocks.resize(partCount);
    for (int i = 0; i < nn; i++)
        blocks[npart[i]].push_back(i);
    
    // Creating the sub graphs for each block
    // subgraphs[n][0] = pointer to xadj, [1] = pointer to adjncy [2]= pointer to npart [3]= number of aggregates
    int aggregateCount = 0;
    vector< vector<int *> > subGraphs(partCount);
    for (int bIt = 0; bIt < blocks.size(); bIt++)
    {
        // Resizing to hold all the pointers
        subGraphs[bIt].resize(4);
        
        // Storing counts for array sizing
        int adjacencySize = 0;
        
        // Temporary vector to hold adjacency;
        vector< vector<int> > adjacency(blocks[bIt].size());
        
        // For every vertex add it's in-block neighbors to the adjacency list:
        for (int vIt = 0; vIt < blocks[bIt].size(); vIt++)
        {
            int start = xadj[blocks[bIt][vIt]];
            int end = xadj[blocks[bIt][vIt] + 1];
            for (int nIt = start; nIt < end; nIt++)
            {
                // Checking if the neighbor is within block:
                int neighbor = adjncy[nIt];
                if (npart[neighbor] == bIt)
                {
                    int localNeighbor = -1;
                    // Find the local index of the neighbor:
                    for (int i = 0; i < blocks[bIt].size(); i++)
                    {
                        if (blocks[bIt][i] == neighbor)
                            localNeighbor = i;
                    }
                    adjacency[vIt].push_back(localNeighbor);
                    adjacencySize++;
                }
            }
        }
        
        // Now allocate the arrays:
        // The xadj array
        subGraphs[bIt][0] = (int *)malloc((blocks[bIt].size() + 1) * sizeof(int));
        
        // The adjncy array
        subGraphs[bIt][1] = (int *)malloc((adjacencySize) * sizeof(int));
        
        // The npart array
        subGraphs[bIt][2] = (int *)malloc((blocks[bIt].size()) * sizeof(int));
        
        // The number of aggregates
        subGraphs[bIt][3] = (int *)malloc(sizeof(int));
        
        // Populating the arrays from the adjacency vector:
        subGraphs[bIt][0][0] = 0;
        int idx = 0;
	
	// Populating the matrices:
	for(int i = 1; i < blocks[bIt].size() + 1; i++)
	{	
            subGraphs[bIt][0][i] = subGraphs[bIt][0][i-1] + adjacency[i-1].size();
            for(int j =0; j < adjacency[i-1].size(); j++)
            {
                subGraphs[bIt][1][idx++] = adjacency[i-1][j];
            }
	}
        
        // Checking if the block's subgraph is connected:
        queue<int> toCheck;
        vector<int> visited(blocks[bIt].size());
        for (int i=0; i < blocks[bIt].size(); i++)
            visited[i] = -1;
        int nextRoot = 0;
        int componentID = 0;
        bool connected = true;
        bool completed = false;
        
        while (!completed)
        {
            toCheck.push(nextRoot);
            visited[nextRoot] = componentID;
            while (!toCheck.empty()){
                int currentV = toCheck.front();
                toCheck.pop();
                for (int nIt = subGraphs[bIt][0][currentV]; nIt < subGraphs[bIt][0][currentV + 1]; nIt++)
                {
                    if(visited[subGraphs[bIt][1][nIt]] == -1)
                    {
                        visited[subGraphs[bIt][1][nIt]] = componentID;
                        toCheck.push(subGraphs[bIt][1][nIt]);
                    }
                }
            }

            completed = true;
            for (int i = 0; i < blocks[bIt].size(); i++)
                if(visited[i] < 0)
                {
                    connected = false;
                    componentID++;
                    nextRoot = i;
                    completed = false;
                    break;
                }
        }
        if (!connected)
        {
            cout << "Block: " << bIt << " is an unconnected graph:\n";
            for (int i = 0; i < blocks[bIt].size(); i++)
                cout << visited[i] << ", ";
            cout << "\n";
        }
        
        // Calling the mis_subroutine to partition
        aggregateGraphMIS(blocks[bIt].size(), subGraphs[bIt][0], subGraphs[bIt][1], subGraphs[bIt][2], subGraphs[bIt][3]);
        aggregateCount += subGraphs[bIt][3][0];
    }
    
    // Running a sanity check on the partitionings:
    for (int bIt=0; bIt < blocks.size(); bIt++)
    {
        for (int vIt=0; vIt < blocks[bIt].size(); vIt++)
        {
            if (subGraphs[bIt][2][vIt] < 0)
                cout << "There is a problem with block: " << bIt << " of " << blocks.size() << " vertex: " << vIt << " in partition: " << subGraphs[bIt][2][vIt] << "?\n";
        }
    }
    
    
    // Now that every block has been aggregated generate the permutation matrices
    aggregateIdx.resize(aggregateCount + 1); 
    partitionIdx.resize(blocks.size() + 1);
    *nnout = aggregateCount;
    
    //int aggregatelabel[nn];
    int* aggregatelabel = new int[nn];
    int currentPosition = 0;
    int aggregatePosition = 0;
    for (int bIt = 0; bIt < blocks.size(); bIt++)
    {
        partitionIdx[bIt] = aggregatePosition;
        for (int aIt = 0; aIt < subGraphs[bIt][3][0]; aIt++)
        {
            aggregateIdx[aggregatePosition] = currentPosition;
            // Find every vertex in the aggregate
            for (int i = 0; i < blocks[bIt].size(); i++)
            {
                if (subGraphs[bIt][2][i] == aIt)
                {
                    int globalVertex = blocks[bIt][i];
                    permutation[globalVertex] = currentPosition;
                    ipermutation[currentPosition] = globalVertex;
                    partitionlabel[globalVertex] = bIt;
                    aggregatelabel[globalVertex] = aggregatePosition;
                    currentPosition++;
                }
            }
            aggregatePosition++;
        }
    }
    aggregateIdx[aggregateCount] = nn;
    partitionIdx[blocks.size()] = aggregateCount;
    
    // Finding the adjacency for the graph of aggregates:
    vector< vector <int> > aggregateAdjacency(aggregateCount);
    int edgeCount = 0;
    for (int aIt = 0; aIt < aggregateCount; aIt++)
    {
        set<int> partEdges;
        int begin = aggregateIdx[aIt];
        int end = aggregateIdx[aIt + 1];
        for (int vIt = begin; vIt < end; vIt++)
        {
            // Getting the original id of the vertex
            int originalID = ipermutation[vIt];
            
            // Examining all neighbors of the vertex
            for (int nIt = xadj[originalID]; nIt < xadj[originalID + 1]; nIt++)
            {
                if (aggregatelabel[ adjncy[nIt] ] != aggregatelabel[originalID])
                    partEdges.insert(aggregatelabel[ adjncy[nIt] ]);
            }
        }
        for (set<int>::iterator i=partEdges.begin(); i != partEdges.end(); i++)
        {
            aggregateAdjacency[aIt].push_back(*i);
            edgeCount++;
        }
    }
    
    // Allocate storage for the xadjout and adjncyout arrays:
    xadjout = (int*)malloc((aggregateCount + 1) * sizeof(int));
    adjncyout = (int*)malloc((edgeCount) * sizeof(int));
    
    // Populate the outgoing arrays
    xadjout[0] = 0;
    int idx = 0;

    // Populating the matrices:
    for(int aIt = 1; aIt < aggregateCount + 1; aIt++)
    {				
        xadjout[aIt] = xadjout[aIt-1] + aggregateAdjacency[aIt-1].size();
        for(int nIt = 0; nIt  < aggregateAdjacency[aIt-1].size(); nIt ++)
        {
            adjncyout[idx++] = aggregateAdjacency[aIt-1][nIt];
        }
    }
    
    // Freeing up memory:
    for (int i = 0; i < subGraphs.size(); i++)
        for (int j = 0; j < 4; j++)
            free(subGraphs[i][j]);
    
    delete [] aggregatelabel;
    // And Done.
    return;
}

template <class Matrix, class Vector>
void MIS_Aggregator<Matrix, Vector>::aggregateGraphMIS(int n, int *adjIndexes, int *adjacency, int *partition, int *partCount)
{
    // Creating a graph with edges for every 2-path in original:
    vector< vector<int> > inducedAdj(n);

    // Every Vertex
    for (int i=0; i<n; i++) {
        // All neighbors
        for (int j=adjIndexes[i]; j<adjIndexes[i+1]; j++) {
            // All neighbors of neighbors
            int neighbor = adjacency[j];
            for (int jj = adjIndexes[neighbor]; jj < adjIndexes[neighbor +1]; jj++) {
                // Checking if this vertex is the original or a distance one
                int vertex = adjacency[jj];
                bool tooClose = false;
                if (vertex != i) {
                        // Checking against distance one
                        for (int ii = adjIndexes[i]; ii < adjIndexes[i+1]; ii++) {
                                if (adjacency[ii] == vertex) {
                                        tooClose = true;
                                        break;
                                }
                        }
                }
                else {
                        tooClose = true;
                }

                // If vertex is two away and not 1 or 0 then add to adjacency:
                if (!tooClose) {
                        inducedAdj[i].push_back(vertex);
                }
            }
        }
    }

    // Clearing partitions:
    for (int i = 0; i < n; i++) {
            partition[i] = -1;
    }

    // Picking a better maximal independent set:
    vector<int> mis(n, -1);
    vector<int> rootDistance(n, -1);
    bool incomplete = true;
    int nextVertex = 0;
    int curPart = 0;
    do {
        while (incomplete) {
            incomplete = false;
            mis[nextVertex] = 1;
            rootDistance[nextVertex] = 0;
            partition[nextVertex] = curPart;

            // Marking adjacent(squared) nodes as not in the mis:
            for (int i = 0; i < inducedAdj[nextVertex].size(); i++) {
                mis[inducedAdj[nextVertex][i]] = 0;
            }

            // Marking adjacent nodes as in the same partition:
            for (int i = adjIndexes[nextVertex]; i < adjIndexes[nextVertex + 1]; i++) {
                partition[ adjacency[i] ] = curPart;
                rootDistance[ adjacency[i] ] = 1;
            }

            curPart++;

            // Getting a list of potential next nodes:
            vector<int> potentialNodes;
            for (int i = 0; i < n; i++) {
                // For every node known to be outside MIS:
                if (mis[i] == 0) {
                    for (int j = adjIndexes[i]; j < adjIndexes[i+1]; j++) {
                        // If a neighbor of an outsider has not been treated add it:
                        if (mis[ adjacency[j] ] == -1) {
                                potentialNodes.push_back(adjacency[j]);
                        }
                    }
                }
            }

            // If there are potential nodes find the best one:
            if (potentialNodes.size() > 0) 
            {
                incomplete = true;
                sort(potentialNodes.begin(), potentialNodes.end());
                int occurs = 0;
                int maxOccur = 0;
                int curNode = potentialNodes[0];
                nextVertex = curNode;
                for (int i = 0; i < potentialNodes.size(); i++) 
                {
                    if (potentialNodes[i] == curNode) 
                    {
                            occurs++;
                    }
                    else 
                    {
                        // If this node has the most occurences seen, set it as next
                        if (maxOccur < occurs) 
                        {
                            nextVertex = curNode;
                            maxOccur = occurs;
                        }
                        // Reset the counters
                        occurs = 1;
                        curNode = potentialNodes[i];
                    }
                }
                if (maxOccur < occurs) 
                {
                    nextVertex = curNode;
                }
            }
        }

        // Setting the partCount:
        *partCount = curPart;

        // Adding unpartitioned nodes to best partition for them:
        for (int i = 0; i < n; i++) {
            if (partition[i] == -1) {
                int adjSize = adjIndexes[i + 1] - adjIndexes[i];
                int * adjParts = new int[ adjSize ];
                int * adjRootDist = new int[ adjSize ];

                // Getting adjacent partitions:
                for (int j = 0; j < adjSize; j++) {
                    int adjacentNodePart = partition[ adjacency[adjIndexes[i] + j] ];
                    adjParts[j] = adjacentNodePart;

                    // Getting the distance of the adjacent node to the root of its partition:
                    if (adjacentNodePart == -1) {
                        adjRootDist[j] = 1000;
                    }
                    else {
                        adjRootDist[j] = rootDistance[ adjacency[adjIndexes[i] + j] ];
                    }
                }

                // Finding the smallest partition distance:
                int smallestDistance = adjRootDist[0];
                for (int j = 0; j<adjSize; j++) {
                    if (smallestDistance > adjRootDist[j]) {
                        smallestDistance = adjRootDist[j];
                    }
                }

                // Finding most adjacent partition:
                int addToPart = -1;
                int adjCount = 0;
                for (int j = 0; j < adjSize; j++) {
                    if (adjParts[j] > -1 && adjRootDist[j] == smallestDistance) {
                        int curCount = 1;
                        int curPart = adjParts[j];
                        for (int jj = j + 1; jj < adjSize; jj++) {
                            if (adjParts[jj] == adjParts[j]) {
                                curCount++;
                            }		
                        }
                        if (curCount > adjCount) {
                            adjCount = curCount;
                            addToPart = curPart;
                        }
                    }
                }

                // Adding the node to best part found:
                partition[i] = addToPart;
                rootDistance[i] = smallestDistance + 1;
				delete adjParts;
				delete adjRootDist;
            }
        }
        // If there are unassigned nodes set the first one as a new root
        // (This should only happen if there were non-connected graphs supplied)
        nextVertex = -1;
        for (int vIt = 0; vIt < n; vIt++)
        {
            if (partition[vIt] == -1){
                nextVertex = vIt;
                incomplete = true;
                break;
            }
        }
        
    }while (nextVertex != -1);
}

template <class Matrix, class Vector>
void MIS_Aggregator<Matrix, Vector>::computePermutation_d(IdxVector_d &adjIndexesIn, IdxVector_d &adjacencyIn, IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionLabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, int parameters, int part_max_size)
{
}

template <class Matrix, class Vector>
void MIS_Aggregator<Matrix, Vector>::computePermutation_d(TriMesh *meshPtr, IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionLabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, int parameters, int part_max_size)
{
}

template <class Matrix, class Vector>
void MIS_Aggregator<Matrix, Vector>::computePermutation_d(TetMesh *meshPtr, IdxVector_d &permutation, IdxVector_d &ipermutation, IdxVector_d &aggregateIdx, IdxVector_d &partitionIdx, IdxVector_d &partitionLabel, IdxVector_d &adjIndexesOut, IdxVector_d &adjacencyOut, int parameters, int part_max_size)
{
}



/****************************************
 * Explict instantiations
 ***************************************/
template class MIS_Aggregator<Matrix_h, Vector_h>;
template class MIS_Aggregator<Matrix_d, Vector_d>;
