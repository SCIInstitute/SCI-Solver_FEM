/* 
 * File:   AggMIS_FileIO.h
 * Author: nachtluce
 *
 * Created on April 17, 2013, 4:23 PM
 */

#ifndef AGGMIS_FILEIO_H
#define	AGGMIS_FILEIO_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "TriMesh.h"
#include "tetmesh.h"
#include "AggMIS_Types.h"
namespace AggMIS {
    namespace FileIO
    {
        using namespace AggMIS::Types;
        using namespace std;

        class DataCollector {
        public:
            DataCollector(string title);
            void set(string name, double value);
            void set(string name, int value);
            void set(string name, string value);
            void set(string name, double value, bool keep);
            void set(string name, int value, bool keep);
            void set(string name, string value, bool keep);
            void closeRow();
            void blankRow();
            void writeOutCSV(ostream *outputStream);
        private:
            vector<vector<string> > data;
            vector<int> keeping;
            string title;
            bool dirty;
        };

        // Takes a filename and tries to load a graph from it
        // by automatically detecting the file type.
        Graph_h* GetGraphFromFile_Auto(string filename);

        // Takes an input stream and reads in a graph in text csr format
        Graph_h* GetGraphFromFile_CSR(istream *theInput);

        // Takes an input stream and reads in a graph in .MSH format
        Graph_h* GetGraphFromFile_MSH(istream *theInput);

        // Takes a filename and loads the graph from a triangular mesh
        // stored in .ply format using Trimesh library
        Graph_h* GetGraphFromFile_TriMesh(string filename);

        // Takes a filename and loads the graph from a tetrahedral mesh
        // stored in .node/.ele format using the tetmesh library
        Graph_h* GetGraphFromFile_TetMesh(string filename);

        // Takes a filename and loads a vector from it
        IntVector_h* GetVectorFromFile_BIN(string filename);

        // Writes out the graph to the specified file in CSR format
        void WriteGraphToFile_CSR(Graph_h graph, string filename);

        // Writes out vector to the specified file
        void WriteVectorToFile_BIN(IntVector_h toWrite, string filename);
    }
}
#endif	/* AGGMIS_FILEIO_H */


