/* 
 * File:   AggMIS_Types.cu
 * Author: T. James Lewis
 *
 * Created on April 15, 2013, 2:18 PM
 */
#include "AggMIS_FileIO.h"
#include "AggMIS_Types.h"
namespace AggMIS {
    namespace FileIO {
        // DataCollector class
        DataCollector::DataCollector(string title) {
            this->title = title;
            dirty = false;
        }
        void DataCollector::set(string name, double value, bool keep) {
            stringstream ss;
            ss << value;
            set(name, ss.str(), keep);
        }
        void DataCollector::set(string name, double value) {
            set(name, value, false);
        }
        void DataCollector::set(string name, int value, bool keep) {
            stringstream ss;
            ss << value;
            set(name, ss.str(), keep);
        }
        void DataCollector::set(string name, int value) {
            set(name, value, false);
        }
        void DataCollector::set(string name, string value) {
            set(name, value, false);
        }
        void DataCollector::set(string name, string value, bool keep) {
            // Ensure that the name exists
            int idx = -1;
            int size = 2;
            for (int i = 0; i < data.size(); i++)
            {
                if (data[i][0] == name)
                    idx = i;
                size = data[i].size();
            }
            if (idx == -1)
            {
                idx = data.size();
                data.push_back(vector<string>());
                data.back().push_back(name);
                for (int i = 0; i < size - 1; i++)
                    data.back().push_back("-");
            }
            if (keep)
                keeping.push_back(idx);

            data[idx].back() = value;
        }
        void DataCollector::closeRow() {
            for (int i = 0; i < data.size(); i++)
                data[i].push_back("-");
            for (int i = 0; i < keeping.size(); i++)
            {
                int idx = keeping[i];
                int copyFrom = data[idx].size() - 2;
                string copy = data[idx][copyFrom];
                data[idx][copyFrom + 1] = copy;
            }
        }
        void DataCollector::blankRow() {
            keeping.resize(0);
            for (int i = 0; i < data.size(); i++)
                data[i].push_back("");
            closeRow();
        }
        void DataCollector::writeOutCSV(ostream* outputStream) {
            if (data.size() == 0)
                return;

            int rowCount = data[0].size();
            if (!dirty)
                rowCount--;

            *outputStream << title << "\n";
            for (int row = 0; row < rowCount; row++)
            {
                for (int column = 0; column < data.size(); column++)
                    *outputStream << data[column][row] << ",";
                *outputStream << "\n";
            }
            *outputStream << "End of " << title << "\n\n";
        }
        
        // Functions
        Graph_h* GetGraphFromFile_Auto(string filename) {
            // Finding the file extension
            size_t indexFound = filename.find_last_of('.');
            if (indexFound == string::npos)
            {
                return NULL;
            }
            string extension = (filename.substr(indexFound + 1));

            if(extension.compare("ply") == 0)
                return GetGraphFromFile_TriMesh(filename);

            if(extension.compare("csr") == 0)
            {
                ifstream *theInput = new ifstream();
                theInput->open(filename.c_str(), ifstream::in);
                return GetGraphFromFile_CSR(theInput);
            }

            if(extension.compare("msh") == 0)
            {
                ifstream *theInput = new ifstream();
                theInput->open(filename.c_str(), ifstream::in);
                return GetGraphFromFile_MSH(theInput);
            }

            if(extension.compare("ele") == 0)
                return GetGraphFromFile_TetMesh(filename.substr(0, indexFound));

            return NULL;
        }
        Graph_h* GetGraphFromFile_CSR(istream* theInput) {
            int adjIndexesSize, adjacencySize;
            string input;

            // Read the first line of input:
            getline(*theInput, input);
            istringstream theLine(input, istringstream::in);
            theLine >> adjIndexesSize >> adjacencySize;

            // Allocate the memory for the arrays
            Graph_h *result = new Graph_h();
            result->indices->resize(adjIndexesSize);
            result->adjacency->resize(adjacencySize);
            
            int *adjIndexes = result->indices->data();
            int *adjacency = result->adjacency->data();

            for (int i = 0; i < adjIndexesSize; i++)
            {
                getline(*theInput, input);
                istringstream theLine(input, istringstream::in);
                theLine >> adjIndexes[i];        
            }

            for (int i = 0; i < adjacencySize; i++)
            {
                getline(*theInput, input);
                istringstream theLine(input, istringstream::in);
                theLine >> adjacency[i];        
            }

            return result; 
        }
        Graph_h* GetGraphFromFile_MSH(istream* theInput) {
        //    char* readChars = new char[100];
        //    int vertCount, elementCount;
        //    string start, end;
        //    string input;
        //    
        //    // Reading in the first line
        //    getline(*theInput, input);
        //    while (!theInput->eof())
        //    {        
        //        // Checking if this is the beginning of a nodes section
        //        if(input.compare(string("$Nodes")) == 0)
        //        {
        //            int counter = 0;
        //            getline(*theInput, input);
        //            istringstream aLine(input, istringstream::in);
        //            aLine >> vertCount;
        //            
        //            // Reading the first node line
        //            getline(*theInput, input);
        //            while (!theInput->eof() && counter < vertCount + 2)
        //            {
        //                // Checking if the line closes the nodes section
        //                if(input.compare(string("$EndNodes")) == 0)
        //                    break;
        //                
        //                // Incrementing node count found
        //                counter++;
        //                
        //                // Adding the vertex id to the graph:
        //                istringstream aLine(input, istringstream::in);
        //                aLine >> readChars;
        //                addVertex(string(readChars));
        //                
        //                //Getting next line
        //                getline(*theInput, input);
        //            }
        //            
        //            // Checking that the appropriate number of nodes was read in
        //            if (counter != vertCount)
        //            {
        //                return false;
        //            }
        //        }
        //        
        //        // Checking if this is the beginning of the elements section
        //        if(input.compare(string("$Elements")) == 0)
        //        {
        //            int counter = 0;
        //            getline(*theInput, input);
        //            istringstream aLine(input, istringstream::in);
        //            aLine >> elementCount;
        //            
        //            // Reading the first node line
        //            getline(*theInput, input);
        //            while (!theInput->eof() && counter < elementCount + 2)
        //            {
        //                // Checking if the line closes the nodes section
        //                if(input.compare(string("$EndElements")) == 0)
        //                    break;
        //                
        //                // Incrementing element count found
        //                counter++;
        //                
        //                // Reading the line and adding the edges
        //                istringstream aLine(input, istringstream::in);
        //                int tagCount = 0;
        //                
        //                // Assume a triangle mesh
        //                int nodeCount = 3;
        //                
        //                // Reading element id and type
        //                aLine >> tagCount >> tagCount;
        //                
        //                // If reading in tets:
        //                if (tagCount == 4)
        //                    nodeCount = 4;
        //                
        //                // Reading to find tagCount
        //                aLine >> tagCount;
        //                
        //                // Reading in tags
        //                for (int i = 0; i < tagCount; i++)
        //                {
        //                    int dummy;
        //                    aLine >> dummy;
        //                }
        //                
        //                // Reading in the node ids
        //                vector<string> nodes;
        //                for (int i = 0; i < nodeCount; i++)
        //                {
        //                    aLine >> readChars;
        //                    nodes.push_back(string(readChars));
        //                }
        //                
        //                // Adding edges between each pair of nodes in element
        //                for (int i = 0; i < nodeCount -1; i++)
        //                    for (int j = i + 1; j < nodeCount; j++)
        //                    {
        //                        addEdge(nodes[i], nodes[j]);
        //                    }
        //                
        //                //Getting next line
        //                getline(*theInput, input);
        //            }
        //            
        //            // Checking that the appropriate number of elements was read in
        //            if (counter != elementCount)
        //            {
        //                printf("There was an error reading in the MSH file!\n");
        //                printf("Declared node count does not match found nodes.\n");
        //                return false;
        //            }            
        //        }
        //        getline(*theInput, input);
        //    }
        //    
        //    // Read the first line of input:
        //    getline(*theInput, input);
        //    while(input[0] == 'c')
        //        getline(*theInput, input);
        //    istringstream theLine(input, istringstream::in);
        //    theLine >> start >> end >> vertCount >> elementCount;
        //    while (!theInput->eof())
        //    {
        //        do {
        //            getline(*theInput, input);
        //        } while (!theInput->eof() && input[0] == 'c');
        //            
        //        istringstream aLine(input, istringstream::in);
        //        aLine >> end >> start;
        //        while (!aLine.eof())
        //        {
        //            aLine >> end;
        //            addEdge(start, end);
        //        }
        //    }
        //    
        //    delete[] readChars;
        //    return true;
            return NULL;
        }
        Graph_h* GetGraphFromFile_TetMesh(string filename) {
            Graph_h *result = new Graph_h();
            
            // Vectors to hold the adjacency:
            IntVector_h &adjIndexes_h = *(result->indices);
            IntVector_h &adjacency_h = *(result->adjacency);
            int vSize;

            printf("Trying to open: %s\n", filename.c_str());
            string nodeFilename = filename + string(".node");
            string eleFilename = filename + string(".ele");
            TetMesh *meshPtr = TetMesh::read(nodeFilename.c_str(), eleFilename.c_str());

            if(meshPtr == NULL)
            {
                printf("Error opening %s!\n", filename.c_str());
                exit(1);
            }

            vSize = meshPtr->vertices.size();
            meshPtr->need_neighbors();

            printf("Successfully opened file and obtained adjacency.\n");

            // Finding total size of adjacency list:
            int adjacencySize = 0;
            for (int i = 0; i < vSize; i++)
            {
                adjacencySize += meshPtr->neighbors[i].size();
            }

            adjIndexes_h.resize(vSize + 1);
            adjacency_h.resize(adjacencySize);

            // Populating adjacency
            adjIndexes_h[0] = 0;;

            int nextOffset = 0;
            for (int i = 0; i < vSize; i ++)
            {        
                for (int j = 0; j < meshPtr->neighbors[i].size(); j++)
                    adjacency_h[nextOffset + j] = meshPtr->neighbors[i][j];

                nextOffset += meshPtr->neighbors[i].size();
                adjIndexes_h[i + 1] = nextOffset;        
            }

            return result;
        }
        Graph_h* GetGraphFromFile_TriMesh(string filename) {
            Graph_h *result = new Graph_h();
            
            // Vectors to hold the adjacency:
            IntVector_h &adjIndexes_h = *(result->indices);
            IntVector_h &adjacency_h = *(result->adjacency);
            int vSize;

            printf("Trying to open: %s\n", filename.c_str());  
            TriMesh *meshPtr = TriMesh::read(filename.c_str());

            if(meshPtr == NULL)
            {
                printf("Error opening %s!\n", filename.c_str());
                exit(1);
            }        

            vSize = meshPtr->vertices.size();
            meshPtr->need_neighbors();

            // Finding total size of adjacency list:
            int adjacencySize = 0;
            for (int i = 0; i < vSize; i++)
            {
                adjacencySize += meshPtr->neighbors[i].size();
            }

            adjIndexes_h.resize(vSize + 1);
            adjacency_h.resize(adjacencySize);

            // Populating adjacency
            adjIndexes_h[0] = 0;;

            int nextOffset = 0;
            for (int i = 0; i < vSize; i ++)
            {        
                for (int j = 0; j < meshPtr->neighbors[i].size(); j++)
                    adjacency_h[nextOffset + j] = meshPtr->neighbors[i][j];

                nextOffset += meshPtr->neighbors[i].size();
                adjIndexes_h[i + 1] = nextOffset;        
            }

            return result;
        }
        IntVector_h* GetVectorFromFile_BIN(string filename) {
            ifstream *theInput = new ifstream();
            theInput->open(filename.c_str(), ios::binary);

            int size;
            theInput->read((char*)&size, sizeof(int));

            // Create the return vector
            IntVector_h *result = new IntVector_h(size);
            IntVector_h &output = *result;

            // Read in the contents from file
            theInput->read((char*)&output[0], sizeof(int) * size);

            // Close file stream
            theInput->close();
            delete theInput;
            return result;
        }
        void WriteGraphToFile_CSR(Graph_h &graphy, string filename) {
            // Writing out the graph:
            printf("Writing graph to file...");
            ofstream theOutput;
            theOutput.open(filename.c_str());
            theOutput << graphy.indices->size() << " " << graphy.adjacency->size() << "\n";
            for (int i = 0; i < graphy.indices->size(); i++)
            {
                theOutput << (*(graphy.indices))[i] << "\n";
            }
            for (int i = 0; i < graphy.adjacency->size(); i++)
            {
                theOutput << (*(graphy.adjacency))[i] << "\n";
            }        

            printf("Complete\n");
            theOutput.close();
        }
        void WriteVectorToFile_BIN(IntVector_h &toWrite, string filename) {
            int size = toWrite.size();
            ofstream theOutput;
            theOutput.open(filename.c_str(), ios::binary);

            // Write the size of the vector to the file
            theOutput.write((char*)&size, sizeof(int));

            // Write the contents of the vector to the file
            theOutput.write((char*)&toWrite[0], sizeof(int) * size);

            // Close the filestream
            theOutput.close();
        }
    }
}
