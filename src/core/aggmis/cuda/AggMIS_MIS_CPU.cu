#include <AggMIS_MIS_CPU.h>
namespace AggMIS {
    namespace MIS {
        struct fringeNode
        {
            int nodeIdx;
            int visits;
            fringeNode(int n, int v)
            {
                nodeIdx = n;
                visits = v;
            }
        };

        class fringeNodeComparer
        {
        public:
            bool operator()(fringeNode &f1, fringeNode &f2)
            {
                if (f1.visits < f2.visits)
                    return true;
                return false;
            }
        };
        IntVector_h* FloodFillMIS(int k, Graph_h &graph) {
            IntVector_h *m = new IntVector_h(graph.Size(), -1);
            IntVector_h &mis = *m; 
            IntVector_h visited(graph.Size(), 0);
            IntVector_h distances(graph.Size(), 1000);
            queue<int> frontier;
            priority_queue<fringeNode, vector<fringeNode>, fringeNodeComparer> fringe;


            // Picking a random starting point:
            srand(time(NULL));
            int starter = rand() % graph.Size();

            bool incomplete = true;
            while (incomplete)
            {   
                if (mis[starter] == -1)
                {
                    fringeNode toAdd(starter, 1);
                    fringe.push(toAdd);
                }

                while (!fringe.empty())
                {
                    // finding best fringe node
                    int nodeToAdd = -1;
                    while (!fringe.empty())
                    {
                        fringeNode candidate = fringe.top();
                        fringe.pop();
                        if (distances[candidate.nodeIdx] > k)
                        {
                            nodeToAdd = candidate.nodeIdx;
                            break;
                        }
                    }
                    if (nodeToAdd == -1)
                    {
                        break;
                    }

                    mis[nodeToAdd] = 1;
                    distances[nodeToAdd] = 0;

                    // Pushing neighbors of mis node onto frontier to start out
                    int start = (*(graph.indices))[nodeToAdd];
                    int end = (*(graph.indices))[nodeToAdd + 1];
                    for (int nIt = start; nIt < end; nIt++)
                    {

                        int neighbor = (*(graph.adjacency))[nIt];
                        if (distances[neighbor] > 1)
                        {
                            distances[neighbor] = 1;
                            frontier.push(neighbor);
                        }
                    }

                    // Exploring to the end of the frontier:
                    while (!frontier.empty())
                    {
                        int exploring = frontier.front();
                        frontier.pop();

                        int distance = distances[exploring];

                        // Mark out the node from the MIS
                        mis[exploring] = 0;

                        // Add the neighbors
                        if (distance < k)
                        {
                            int start = (*(graph.indices))[exploring];
                            int end = (*(graph.indices))[exploring + 1];
                            for (int nIt = start; nIt < end; nIt++)
                            {
                                int neighbor = (*(graph.adjacency))[nIt];
                                if (distances[neighbor] > distance + 1)
                                {
                                    distances[neighbor] = distance + 1;
                                    frontier.push(neighbor);
                                }
                            }
                        }
                        if (distance == k)
                        {
                            int start = (*(graph.indices))[exploring];
                            int end = (*(graph.indices))[exploring + 1];
                            for (int nIt = start; nIt < end; nIt++)
                            {
                                int neighbor = (*(graph.adjacency))[nIt];
                                if (distances[neighbor] >= distance + 1)
                                {
                                    distances[neighbor] = distance + 1;
                                    fringeNode toAdd(neighbor, ++visited[neighbor]);
                                    fringe.push(toAdd);
                                }
                            }
                        }
                    }
                }

                incomplete = false;
                for (int i = 0; i < graph.Size(); i++)
                {
                    if (mis[i] == -1)
                    {
                        incomplete = true;
                        starter = i;
                        break;
                    }
                }
            }
            visited.clear();
            distances.clear();
            return m;
        }
        IntVector_h* NaiveMIS(int k, Graph_h graph) {
            IntVector_h *m = new IntVector_h(graph.Size(), -1);
            IntVector_h &mis = *m;  
            IntVector_h distances(graph.Size(), 1000);
            for (int i = 0; i < graph.Size(); i++)
            {
                if (mis[i] == -1)
                {
                    mis[i] = 1;
                    distances[i] = 0;

                    queue<int> frontier;

                    // Pushing neighbors of mis node onto frontier to start out
                    int start = (*(graph.indices))[i];
                    int end = (*(graph.indices))[i + 1];
                    for (int nIt = start; nIt < end; nIt++)
                    {
                        int neighbor = (*(graph.adjacency))[nIt];
                        if (distances[neighbor] > 1)
                        {
                            frontier.push(neighbor);
                            distances[neighbor] = 1;
                        }
                    }

                    // Exploring to the end of the frontier:
                    while (!frontier.empty())
                    {
                        int exploring = frontier.front();
                        int distance = distances[exploring];
                        frontier.pop();

                        // Set node out of mis
                        mis[exploring] = 0;

                        // Add the neighbors
                        if (distance < k)
                        {
                            int start = (*(graph.indices))[exploring];
                            int end = (*(graph.indices))[exploring + 1];
                            for (int nIt = start; nIt < end; nIt++)
                            {
                                int neighbor = (*(graph.adjacency))[nIt];
                                if (distances[neighbor] > distance + 1)
                                {
                                    distances[neighbor] = distance + 1;
                                    frontier.push(neighbor);
                                }
                            }
                        }
                    }
                }
            }
            distances.clear();
            return m;
        }
    }
}