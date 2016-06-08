#include "gtest/gtest.h"
#include "FEMSolver.h"
TEST(SanityTests, EggCarton3D) {
  //test the egg carton
  FEMSolver cfg(std::string(TEST_DATA_DIR) + "/tetVol", true, true);
  float lambda = 1.f;
  //read the A matrix
  cfg.readMatlabSparseMatrix(std::string(TEST_DATA_DIR) + "/tetVolA.mat");
  //read the b vector
  Vector_h_CG b_h(cfg.getMatrixRows(), 1.0), x_h(cfg.getMatrixRows(), 0.), x_answer;
  cfg.readMatlabArray(std::string(TEST_DATA_DIR) + "/tetVolb.mat", &b_h);
  //solve
  cfg.solveFEM(&x_h, &b_h);
  //read in known answer
  cfg.readMatlabArray(std::string(TEST_DATA_DIR) + "/tetVolAns.mat", &x_answer);
  //look for error
  double error = 0.f;
  std::vector<double> x_actual;
  for (int i = 0; i < cfg.getMatrixRows(); i++) {
    error += (x_h[i] - x_answer[i]) * (x_h[i] - x_answer[i]);
    x_actual.push_back(x_h[i]);
  }
  std::cout << "The error is : " << std::sqrt(error) << std::endl;
  ASSERT_TRUE(std::sqrt(error) < 25.);
}
