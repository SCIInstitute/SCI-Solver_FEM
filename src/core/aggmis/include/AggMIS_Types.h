/*
 * File:   AggMIS_Types.h
 * Author: T. James Lewis
 *
 * Created on April 15, 2013, 2:18 PM
 */

#ifndef AGGMIS_TYPES_H
#define  AGGMIS_TYPES_H
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/sort.h"
#include "thrust/functional.h"
#include "thrust/unique.h"
#include "my_timer.h"
#include <time.h>
#include <string>
#include <sstream>
#include <vector>

namespace AggMIS {
  bool CheckCudaError(cudaError_t code, const char *file, int line);
  namespace Types {
    typedef thrust::device_vector<int> IntVector_d;
    typedef thrust::device_vector<unsigned int> UIntVector_d;
    typedef thrust::host_vector<int> IntVector_h;
    typedef thrust::host_vector<IntVector_h*> IntVector_h_ptr;
    typedef thrust::host_vector<IntVector_d*> IntVector_d_ptr;

    struct DGraph {
      __host__ __device__ DGraph(int _size,
          int *_ind,
          int *_adj)
        :size(_size),
        ind(_ind),
        adj(_adj){}
      __device__ int getNeighbor(int node, int neighbor) {
        return adj[ind[node] + neighbor];
      }
      int size;
      int *ind;
      int *adj;
    };

    class JTimer {
      public:
        JTimer();
        ~JTimer();
        void start();
        void stop();
        double getElapsedTimeInSec(bool host);
        double getElapsedTimeInMilliSec(bool host);
      private:

        double startTimeHost, endTimeHost;
        cudaEvent_t startTimeCuda, endTimeCuda;
        bool started, stopped;
        float elapsedCudaTime;
    };

    // Forward declarations of classes so the conversion constructors
    // will compile.
    class Graph_d;
    class Graph_h;

    class Graph_d {
      public:
        Graph_d(IntVector_d &indices, IntVector_d &adjacency);
        Graph_d(IntVector_h &indices, IntVector_h &adjacency);
        Graph_d(IntVector_d *indices, IntVector_d *adjacency);
        Graph_d(Graph_h &graph);
        Graph_d();
        ~Graph_d();
        int Size();
        int* indStart();
        int* adjStart();
        DGraph GetD();
        IntVector_d *indices;
        IntVector_d *adjacency;
      private:
        bool willClean;
    };
    class Graph_h {
      public:
        Graph_h(IntVector_d &indices, IntVector_d &adjacency);
        Graph_h(IntVector_h &indices, IntVector_h &adjacency);
        Graph_h(IntVector_h *indices, IntVector_h *adjacency);
        Graph_h(Graph_d &graph);
        Graph_h();
        ~Graph_h();
        int Size();
        int* nStart(int node);
        int* nEnd(int node);
        IntVector_h *indices;
        IntVector_h *adjacency;
      private:
        bool willClean;
    };

    int* StartOf(IntVector_d &target);
    int* StartOf(IntVector_d *target);

    namespace Compare {
      bool AreEqual(IntVector_h& a,
          IntVector_h& b,
          bool verbose);
      bool AreEqual(IntVector_d& a,
          IntVector_d& b,
          bool verbose);
      bool AreEqual(IntVector_h& a,
          IntVector_d& b,
          bool verbose);
      bool AreEqual(IntVector_d& a,
          IntVector_h& b,
          bool verbose);
      bool AreEqual(std::vector<std::vector<int> > &a,
        std::vector<std::vector<int> > &b,
          bool verbose);
      bool AreEqual(Graph_h& a,
          Graph_h& b,
          bool verbose);
      bool AreEqual(Graph_d& a,
          Graph_d& b,
          bool verbose);
      bool AreEqual(Graph_h& a,
          Graph_d& b,
          bool verbose);
      bool AreEqual(Graph_d& a,
          Graph_h& b,
          bool verbose);
    }
    namespace Display {
      void Print(IntVector_h& toPrint,
          int start,
          int end,
          std::string message);
      void Print(IntVector_d& toPrint,
          int start,
          int end,
          std::string message);
      void Print(IntVector_d& toPrint,
        std::string message);
      void Print(IntVector_h& toPrint,
        std::string message);
      void Print(std::vector<std::vector<std::vector<int> > >& toPrint, std::string message);
      void Print(std::vector<std::vector<int> >& toPrint,
        std::string message);
      void Print(std::vector<int> &toPrint,
          int start,
          int end,
          std::string message);
      void Print(std::vector<int> &toPrint,
        std::string message);
    }
  }
}
#endif  /* AGGMIS_TYPES_H */
