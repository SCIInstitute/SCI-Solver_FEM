/*
 * File:   AggMIS_Types.cu
 * Author: T. James Lewis
 *
 * Created on April 15, 2013, 2:18 PM
 */
#include "AggMIS_Types.h"

namespace AggMIS {
  bool CheckCudaError(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
      std::cout << "\n***************** CUDA Error detected ***************\n";
      std::cout << "Error: " << cudaGetErrorString(code) << "\n";
      std::cout << "In file " << file << " line " << line << "\n";
      std::cout << "\n*****************************************************\n";
    }
    code = cudaGetLastError();
    if (code != cudaSuccess) {
      std::cout << "\n*************** Past CUDA Error detected ************\n";
      std::cout << "Error: " << cudaGetErrorString(code) << "\n";
      std::cout << "In file " << file << " line " << line << "\n";
      std::cout << "\n*****************************************************\n";
    }
    return false;
  }
  namespace Types {
    // My timer implementation
    JTimer::JTimer() {
      cudaEventCreate(&startTimeCuda);
      cudaEventCreate(&endTimeCuda);
      started = false;
      stopped = false;
	  startTimeHost = endTimeHost = 0.;
    }
    JTimer::~JTimer() {}
    void JTimer::start() {
      cudaEventRecord(startTimeCuda, 0);
	  startTimeHost = CLOCK();
      started = true;
      stopped = false;
    }
    void JTimer::stop() {
      if (started && !stopped) {
        cudaEventRecord(endTimeCuda, 0);
        cudaEventSynchronize(endTimeCuda);
		endTimeHost = CLOCK();
        stopped = true;
      }
    }
    double JTimer::getElapsedTimeInSec(bool host) {
      if (!started || !stopped) {
        printf("Error: elapsed time requested when not valid.\n");
        return -1.0;
      }
      if (!host) {
        cudaEventElapsedTime(&elapsedCudaTime, startTimeCuda, endTimeCuda);
        return (double) elapsedCudaTime / 1000.0;
      }
      return 0.;
    }
    double JTimer::getElapsedTimeInMilliSec(bool host) {
      if (!host) {
        cudaEventElapsedTime(&elapsedCudaTime, startTimeCuda, endTimeCuda);
        return (double) elapsedCudaTime;
      }
      return 0.;
    }

    // Graph_d members
    Graph_d::Graph_d(IntVector_d& indices,
        IntVector_d& adjacency) {
      this->indices = new IntVector_d(indices);
      this->adjacency = new IntVector_d(adjacency);
      willClean = true;
    }
    Graph_d::Graph_d(IntVector_h& indices,
        IntVector_h& adjacency) {
      this->indices = new IntVector_d(indices);
      this->adjacency = new IntVector_d(adjacency);
      willClean = true;
    }
    Graph_d::Graph_d(IntVector_d* indices,
        IntVector_d* adjacency) {
      this->indices = indices;
      this->adjacency = adjacency;
      willClean = false;
    }
    Graph_d::Graph_d(Graph_h& graph) {
      indices = new IntVector_d(*(graph.indices));
      adjacency = new IntVector_d(*(graph.adjacency));
      willClean = true;
    }
    Graph_d::Graph_d() {
      indices = new IntVector_d();
      adjacency = new IntVector_d();
      willClean = true;
    }
    Graph_d::~Graph_d() {
      if (willClean)
      {
        indices->clear();
        adjacency->clear();
        delete indices;
        delete adjacency;
      }
    }
    int Graph_d::Size() {
      return indices->size() - 1;
    }
    DGraph Graph_d::GetD() {
      return DGraph(Size(), indStart(), adjStart());
    }
    int* Graph_d::indStart() {
      return thrust::raw_pointer_cast(indices->data());
    }
    int* Graph_d::adjStart() {
      return thrust::raw_pointer_cast(adjacency->data());
    }

    // Graph_h members
    Graph_h::Graph_h(IntVector_d& indices,
        IntVector_d& adjacency) {
      this->indices = new IntVector_h(indices);
      this->adjacency = new IntVector_h(adjacency);
      willClean = true;
    }
    Graph_h::Graph_h(IntVector_h& indices,
        IntVector_h& adjacency) {
      this->indices = new IntVector_h(indices);
      this->adjacency = new IntVector_h(adjacency);
      willClean = true;
    }
    Graph_h::Graph_h(IntVector_h* indices,
        IntVector_h* adjacency) {
      this->indices = indices;
      this->adjacency = adjacency;
      willClean = false;
    }
    Graph_h::Graph_h(Graph_d& graph) {
      indices = new IntVector_h(*(graph.indices));
      adjacency = new IntVector_h(*(graph.adjacency));
      willClean = true;
    }
    Graph_h::Graph_h() {
      indices = new IntVector_h();
      adjacency = new IntVector_h();
      willClean = true;
    }
    Graph_h::~Graph_h() {
      if (willClean)
      {
        indices->resize(0);
        adjacency->resize(0);
        delete indices;
        delete adjacency;
      }
    }
    int Graph_h::Size() {
      return indices->size() - 1;
    }
    int* Graph_h::nStart(int node) {
      return &((*adjacency)[(*indices)[node]]);
    }
    int* Graph_h::nEnd(int node) {
      return &((*adjacency)[(*indices)[node + 1]]);
    }

    // Functions
    int* StartOf(IntVector_d &target) {
      return thrust::raw_pointer_cast(target.data());
    }
    int* StartOf(IntVector_d *target) {
      return thrust::raw_pointer_cast(target->data());
    }

    namespace Compare {
      bool AreEqual(IntVector_h& a,
          IntVector_h& b,
          bool verbose) {
        bool good = true;
        if (a.size() != b.size())
        {
          if (verbose)
            printf("Vectors to compare differ in size: a.size()=%d b.size=%d\n", a.size(), b.size());
          return false;
        }

        for (int i = 0; i < a.size(); i++)
          if (a[i] != b[i])
          {
            if (verbose)
              printf("Difference found: a[%d]=%d b[%d]=%d\n", i, a[i], i, b[i]);
            good = false;
          }
        return good;
      }
      bool AreEqual(IntVector_d& a,
          IntVector_d& b,
          bool verbose) {
        IntVector_h tempA(a);
        IntVector_h tempB(b);
        bool result = AreEqual(tempA, tempB, verbose);
        tempA.clear();
        tempB.clear();
        return result;
      }
      bool AreEqual(IntVector_h& a,
          IntVector_d& b,
          bool verbose) {
        IntVector_h temp(b);
        bool result = AreEqual(a, temp, verbose);
        temp.clear();
        return result;
      }
      bool AreEqual(IntVector_d& a,
          IntVector_h& b,
          bool verbose) {
        return AreEqual(b, a, verbose);
      }
      bool AreEqual(vector<vector<int> > &a,
          vector<vector<int> > &b,
          bool verbose) {
        // Check that main containers have matching sizes
        if (a.size() != b.size()) {
          if (verbose)
            printf("Sizes of base vectors do not match! a=%d b=%d\n",
                a.size(), b.size());
          return false;
        }
        // Check that sizes of nested containers match
        for (int i = 0; i < a.size(); i++) {
          if (a[i].size() != b[i].size()) {
            if (verbose) {
              printf("Sizes of secondary vectors %d do not match!\n", i);
              printf("a[%d].size()=%d  b[%d].size()=%d\n",
                  i, a[i].size(), i, b[i].size());
              stringstream ss;
              ss << "Contents of A[" << i << "]";
              Display::Print(a[i], ss.str());
              ss.str("Contents of B[");
              ss << i << "]";
              Display::Print(b[i], ss.str());
            }
            return false;
          }
        }
        // Check that all entries are equal
        for (int i = 0; i < a.size(); i++) {
          for (int j = 0; j < a[i].size(); j++) {
            if (a[i][j] != b[i][j]) {
              if (verbose) {
                printf("Element[%d][%d] does not match!\n", i, j);
                stringstream ss;
                ss << "Contents of A[" << i << "]";
                Display::Print(a[i], ss.str());
                ss.str("Contents of B[");
                ss << i << "]";
                Display::Print(b[i], ss.str());
              }
              return false;
            }
          }
        }
        return true;
      }
      bool AreEqual(Graph_h& a,
          Graph_h& b,
          bool verbose) {
        bool indicesMatch = AreEqual(*(a.indices),
            *(b.indices),
            verbose);
        bool adjacencyMatch = AreEqual(*(a.adjacency),
            *(b.adjacency),
            verbose);
        if (!indicesMatch && verbose)
          printf("Indices of graphs differ!\n");
        if (!adjacencyMatch && verbose)
          printf("Adjacency lists of graphs differ!\n");
        return indicesMatch && adjacencyMatch;
      }
      bool AreEqual(Graph_d& a,
          Graph_d& b,
          bool verbose) {
        bool indicesMatch = AreEqual(*(a.indices),
            *(b.indices),
            verbose);
        bool adjacencyMatch = AreEqual(*(a.adjacency),
            *(b.adjacency),
            verbose);
        if (!indicesMatch && verbose)
          printf("Indices of graphs differ!\n");
        if (!adjacencyMatch && verbose)
          printf("Adjacency lists of graphs differ!\n");
        return indicesMatch && adjacencyMatch;
      }
      bool AreEqual(Graph_h& a,
          Graph_d& b,
          bool verbose) {
        bool indicesMatch = AreEqual(*(a.indices),
            *(b.indices),
            verbose);
        bool adjacencyMatch = AreEqual(*(a.adjacency),
            *(b.adjacency),
            verbose);
        if (!indicesMatch && verbose)
          printf("Indices of graphs differ!\n");
        if (!adjacencyMatch && verbose)
          printf("Adjacency lists of graphs differ!\n");
        return indicesMatch && adjacencyMatch;
      }
      bool AreEqual(Graph_d& a,
          Graph_h& b,
          bool verbose) {
        bool indicesMatch = AreEqual(*(a.indices),
            *(b.indices),
            verbose);
        bool adjacencyMatch = AreEqual(*(a.adjacency),
            *(b.adjacency),
            verbose);
        if (!indicesMatch && verbose)
          printf("Indices of graphs differ!\n");
        if (!adjacencyMatch && verbose)
          printf("Adjacency lists of graphs differ!\n");
        return indicesMatch && adjacencyMatch;
      }
    }
    namespace Display {
      void Print(IntVector_h& toPrint,
          int start,
          int end,
          string message) {
        printf("%s:\n", message.c_str());
        printf("\n %8d: ", 0);
        for (int i = start; i < end; i++)
        {
          if ((i-start) % 10 == 0 && (i-start) > 0)
            printf("\n %8d: ", i);

          int value = toPrint[i];
          printf(" %8d", value);
        }
        printf("\n");
      }
      void Print(IntVector_d& toPrint,
          int start,
          int end,
          string message) {
        IntVector_h temp(toPrint);
        Print(temp, start, end, message);
        temp.clear();
      }
      void Print(IntVector_d& toPrint,
          string message) {
        IntVector_h temp(toPrint);
        Print(temp, 0, temp.size(), message);
        temp.clear();
      }
      void Print(IntVector_h& toPrint,
          string message) {
        Print(toPrint, 0, toPrint.size(), message);
      }
      void Print(vector<vector<vector<int> > >& toPrint,
          string message) {
        // Print out general info:
        printf("Triple vector %s has %d entries:\n", message.c_str(), toPrint.size());

        for (int i = 0; i < toPrint.size(); i++)
        {
          cout << message << "[" << i << "]: ";
          for (int z = 0; z < toPrint[i].size(); z++)
          {
            cout << "(";
            for (int zz = 0; zz < toPrint[i][z].size(); zz++)
            {
              cout << toPrint[i][z][zz];
              if (zz < toPrint[i][z].size() -1)
                cout << " ";
            }
            cout << ") ";
          }
          cout << "\n";
        }
        cout << "\n";
      }
      void Print(vector<vector<int> >& toPrint,
          string message) {
        printf("%s:\n", message.c_str());
        for (int j = 0; j < toPrint.size(); j++) {
          printf("\n %4d: ", j);
          for (int i = 0; i < toPrint[j].size(); i++)
          {
            if (i % 10 == 0 && i > 0)
              printf("\n %4d: ", j);

            int value = toPrint[j][i];
            printf(" %4d", value);
          }
        }
        printf("\n");
      }
      void Print(vector<int>& toPrint,
          int start,
          int end,
          string message) {
        IntVector_h temp(toPrint.begin(), toPrint.end());
        Print(temp, start, end, message);
      }
      void Print(vector<int>& toPrint,
          string message) {
        Print(toPrint, 0, toPrint.size(), message);
      }
    }
  }
}
