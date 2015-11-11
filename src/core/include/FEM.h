#ifndef __FEM_H__
#define __FEM_H__

#include <stdio.h>
#include <iostream>
#include <signal.h>
#include <exception>
#include <fstream>
#include <amg_config.h>
#include <types.h>
#include <TriMesh.h>
#include <tetmesh.h>
#include <cutil.h>
#include <FEM/FEM2D.h>
#include <FEM/FEM3D.h>
#include <my_timer.h>
#include <amg.h>
#include <typeinfo>

#include <time.h>

namespace FEM {
  /** The class that represents all of the available options for FEM */
  class FEM {
    public:
      FEM(std::string fname = "../src/test/test_data/sphere334",
          bool verbose = false) :
        verbose_(verbose),
        filename_(fname),
        seedPointList_(std::vector<int>(1, 0)),
        maxBlocks_(1000),
        maxVertsPerBlock_(64),
        isStructured_(false),
        squareLength_(16),
        squareWidth_(16),
        squareDepth_(16),
        squareBlockLength_(1),
        squareBlockWidth_(1),
        squareBlockDepth_(1),
        maxIterations_(1000)

    {}
      //3D data
      bool verbose_;
      std::string filename_;
      std::vector<int> seedPointList_;
      int maxBlocks_;
      int maxVertsPerBlock_;
      bool isStructured_;
      int squareLength_, squareWidth_, squareDepth_;
      int squareBlockLength_, squareBlockWidth_, squareBlockDepth_;
      int maxIterations_;
  };

  //The static pointer to the mesh
  static TetMesh * mesh_ = NULL;
  //the answer vector
  static std::vector < std::vector <float> > iteration_values_;
  //accessor functions to the results.
  std::vector < float >& getFinalResult() {
    return iteration_values_.at(iteration_values_.size() - 1);
  }
  std::vector < float >& getResultAtIteration(size_t i) {
    return iteration_values_.at(i);
  }
  size_t numIterations() { return iteration_values_.size(); }
  void writeVTK() {
    //TODO
  }

  /**
   * Creates the mesh, partitions the mesh, and runs the algorithm.
   *
   * @data The set of options for the Eikonal algorithm.
   *       The defaults are used if nothing is provided.
   */
  void solveFEM(FEM data = FEM()) {

    clock_t starttime, endtime;
    starttime = clock ();
    mesh_ = TetMesh::read(
      (data.filename_ + ".node").c_str(),
      (data.filename_ + ".ele").c_str(), true, data.verbose_);
    mesh_->rescale(4.0);
    mesh_->need_neighbors(data.verbose_);
    mesh_->need_meshquality(data.verbose_);
    //TODO
    endtime = clock();
    double duration = (double)(endtime - starttime) * 1000/ CLOCKS_PER_SEC;

    if (data.verbose_)
      printf("Computing time : %.10lf ms\n",duration);
  }

  /**
   * This function uses the provided analytical solutions to
   * visualize the algorithm's error after each iteration.
   *
   * @param solution The vector of expected solutions.
   */
  void printErrorGraph(std::vector<float> solution) {

    // now calculate the RMS error for each iteration
    std::vector<float> rmsError;
    rmsError.resize(numIterations());
    for (size_t i = 0; i < numIterations(); i++) {
      float sum = 0.f;
      std::vector<float> result = getResultAtIteration(i);
      for (size_t j = 0; j < solution.size(); j++) {
        float err = std::abs(solution[j] - result[j]);
        sum +=  err * err;
      }
      rmsError[i] = std::sqrt(sum / static_cast<float>(solution.size()));
    }
    //determine the log range
    float max_err = rmsError[0];
    float min_err = rmsError[rmsError.size() - 1];
    int max_log = -10, min_log = 10;
    while (std::pow(static_cast<float>(10),max_log) < max_err) max_log++;
    while (std::pow(static_cast<float>(10),min_log) > min_err) min_log--;
    // print the error graph
    printf("\n\nlog(Err)|\n");
    bool printTick = true;
    for(int i = max_log ; i >= min_log; i--) {
      if (printTick) {
        printf("   10^%2d|",i);
      } else {
        printf("        |");
      }
      for (size_t j = 0; j < numIterations(); j++) {
        if (rmsError[j] > std::pow(static_cast<float>(10),i) &&
            rmsError[j] < std::pow(static_cast<float>(10),i+1))
          printf("*");
        else
          printf(" ");
      }
      printf("\n");
      printTick = !printTick;
    }
    printf("--------|------------------------------------------");
    printf("  Converged to: %.4f\n",rmsError[rmsError.size() - 1]);
    printf("        |1   5    10   15   20   25   30   35\n");
    printf("                   Iteration\n");
  }
}

#endif
