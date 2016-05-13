/* 
 * File:   AggMIS_FileIO.h
 * Author: nachtluce
 *
 * Created on April 17, 2013, 4:23 PM
 */

#ifndef AGGMIS_FILEIO_H
#define	AGGMIS_FILEIO_H

#include <istd::ostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "TriMesh.h"
#include "tetmesh.h"
#include "AggMIS_Types.h"
namespace AggMIS {
    namespace FileIO
    {
        class DataCollector {
        public:
            DataCollector(std::string title);
            void set(std::string name, double value);
            void set(std::string name, int value);
            void set(std::string name, std::string value);
            void set(std::string name, double value, bool keep);
            void set(std::string name, int value, bool keep);
            void set(std::string name, std::string value, bool keep);
            void closeRow();
            void blankRow();
            void writeOutCSV(std::ostream *outputStream);
        private:
            vector<vector<std::string> > data;
            vector<int> keeping;
            std::string title;
            bool dirty;
        };

        // Takes a filename and tries to load a graph from it
        // by automatically detecting the file type.
        Types::Graph_h* GetGraphFromFile_Auto(std::string filename);

        // Takes an input stream and reads in a graph in text csr format
        Types::Graph_h* GetGraphFromFile_CSR(std::istream *theInput);

        // Takes an input stream and reads in a graph in .MSH format
        Types::Graph_h* GetGraphFromFile_MSH(std::istream *theInput);

        // Takes a filename and loads the graph from a triangular mesh
        // stored in .ply format using Trimesh library
        Types::Graph_h* GetGraphFromFile_TriMesh(std::string filename);

        // Takes a filename and loads the graph from a tetrahedral mesh
        // stored in .node/.ele format using the tetmesh library
        Types::Graph_h* GetGraphFromFile_TetMesh(std::string filename);

        // Takes a filename and loads a vector from it
        Types::IntVector_h* GetVectorFromFile_BIN(std::string filename);

        // Writes out the graph to the specified file in CSR format
        void WriteGraphToFile_CSR(Types::Graph_h graph, std::string filename);

        // Writes out vector to the specified file
        void WriteVectorToFile_BIN(Types::IntVector_h toWrite, std::string filename);
    }
}
#endif	/* AGGMIS_FILEIO_H */


