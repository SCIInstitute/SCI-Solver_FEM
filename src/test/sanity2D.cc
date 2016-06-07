#include "gtest/gtest.h"
#include "FEMSolver.h"
TEST(SanityTests, EggCarton2D) {
  //test the egg carton
  FEMSolver cfg(std::string(TEST_DATA_DIR) + "/simple.ply", false, true);
  size_t num_vert = cfg.getMatrixRows();
  float lambda = 1.f;
  //read the A matrix
  cfg.readMatlabSparseMatrix(std::string(TEST_DATA_DIR) + "/simpleTri.mat");
  //create the b vector
  Vector_h_CG b_h(num_vert, 1.0), x_h(num_vert, 0.0);
  for (int i = 0; i < num_vert; i++) {
    b_h[i] = 8. * M_PI * M_PI * std::sin(2. * M_PI * cfg.triMesh_->vertices[i][0]) *
      std::sin(2. * M_PI * cfg.triMesh_->vertices[i][1]) + lambda *
      std::sin(2. * M_PI * cfg.triMesh_->vertices[i][0]) *
      std::sin(2. * M_PI * cfg.triMesh_->vertices[i][1]);
  }
  //create the b vector
  std::vector<float> x_answer;
  for (int i = 0; i < num_vert; i++) {
    x_answer.push_back(
      std::sin(2. * M_PI * cfg.triMesh_->vertices[i][0]) *
      std::sin(2. * M_PI * cfg.triMesh_->vertices[i][1]));
  }
  cfg.solveFEM(&x_h, &b_h);
  //look for error
  float error = 0.f;
  std::vector<float> x_actual;
  for (int i = 0; i < num_vert; i++) {
    error += (x_h[i] - x_answer[i]) * (x_h[i] - x_answer[i]);
    x_actual.push_back(x_h[i]);
  }
  std::cout << "The error is : " << std::sqrt(error) << std::endl;
  cfg.writeVTK(x_actual, "test_egg_carton1");
  ASSERT_TRUE(std::sqrt(error) < 1.);
}
