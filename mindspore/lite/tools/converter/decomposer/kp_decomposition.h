/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_DECOMPOSER_KRONECKER_DECOMPOSITION_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_DECOMPOSER_KRONECKER_DECOMPOSITION_H_
#include <set>
#include <vector>
#include <utility>
#include "unsupported/Eigen/KroneckerProduct"

namespace mindspore::lite::decomposer {
// Kronecker Production Decomposition to matrix
class KPDecomposition {
 public:
  KPDecomposition(float *data, size_t data_size, const std::vector<int> &shape)
      : data_(data), data_size_(data_size), shape_(shape) {}
  ~KPDecomposition() = default;

  int Decomposition();

  std::vector<float> GetMatB() const { return mat_b_; }
  std::vector<int> GetMatShapeB() const { return mat_shape_b_; }
  std::vector<float> GetMatC() const { return mat_c_; }
  std::vector<int> GetMatShapeC() const { return mat_shape_c_; }

 private:
  // Integer factorization to multiply two constraints ==> A = B * C
  std::set<std::pair<int, int>> IntegerFactorization(int num);

  size_t FindBestFactorization(const std::set<std::pair<int, int>> &m_factors,
                               const std::set<std::pair<int, int>> &n_factors, size_t origin_element_num,
                               std::pair<int, int> *best_pair_m, std::pair<int, int> *best_pair_n);

  int PackMatrixA(const float *data, size_t num, const std::vector<int> &shape, const std::vector<int> &block_size,
                  float *new_data);

  double CalcFNorm(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix_b,
                   const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix_c);

 private:
  float *data_ = nullptr;
  size_t data_size_ = 0;
  std::vector<int> shape_;

  std::vector<float> mat_b_;
  std::vector<int> mat_shape_b_;
  std::vector<float> mat_c_;
  std::vector<int> mat_shape_c_;
};
}  // namespace mindspore::lite::decomposer
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_DECOMPOSER_KRONECKER_DECOMPOSITION_H_
