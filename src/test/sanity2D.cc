#include "gtest/gtest.h"
#include "FEMSolver.h"
TEST(SanityTests, EggCarton2D) {
  //test the egg carton
  FEMSolver cfg(std::string(TEST_DATA_DIR) + "/simple.ply", false, true);
  float lambda = 1.f;
  //read the A matrix
  cfg.readMatlabSparseMatrix(std::string(TEST_DATA_DIR) + "/simpleTri.mat");
  //read the b vector
  Vector_h_CG b_h(cfg.getMatrixRows(), 1.0), x_h(cfg.getMatrixRows(), 0.), x_answer;
  cfg.readMatlabArray(std::string(TEST_DATA_DIR) + "/simpleTrib.mat", &b_h);
  //solve
  cfg.solveFEM(&x_h, &b_h);
  //read in known answer
  cfg.readMatlabArray(std::string(TEST_DATA_DIR) + "/simpleTriAns.mat", &x_answer);
  //look for error
  double error = 0.f;
  std::vector<double> x_actual, err;
  for (int i = 0; i < x_answer.size(); i++) {
    error += (x_h[i] - x_answer[i]) * (x_h[i] - x_answer[i]);
    x_actual.push_back(x_h[i]);
    err.push_back(std::abs(x_h[i] - x_answer[i]));
  }
  std::cout << "The error is : " << std::sqrt(error) << std::endl;
  cfg.writeVTK(x_actual, "test_egg_carton1");
  ASSERT_TRUE(std::sqrt(error) < 100.);
}
