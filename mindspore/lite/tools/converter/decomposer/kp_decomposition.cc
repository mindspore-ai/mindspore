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

#include "tools/converter/decomposer/kp_decomposition.h"
#include <set>
#include <vector>
#include <cmath>
#include <utility>
#include "src/common/log_util.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "tools/converter/decomposer/svd_matrix.h"

namespace mindspore::lite::decomposer {
namespace {
constexpr int kRowIndex = 0;
constexpr int kColIndex = 1;
constexpr int kMinSize = 2;
}  // namespace
// Integer factorization to multiply two constraints ==> A = B * C
std::set<std::pair<int, int>> KPDecomposition::IntegerFactorization(int num) {
  std::set<std::pair<int, int>> result;
  for (size_t i = 2; i <= std::sqrt(num); ++i) {
    if (num % i == 0) {
      result.insert({i, num / i});
      result.insert({num / i, i});
    }
  }
  return result;
}

// Find factors with the maximum compression ratio.
size_t KPDecomposition::FindBestFactorization(const std::set<std::pair<int, int>> &m_factors,
                                              const std::set<std::pair<int, int>> &n_factors, size_t origin_element_num,
                                              std::pair<int, int> *best_pair_m, std::pair<int, int> *best_pair_n) {
  float best_compression = 1.0f;
  for (auto m : m_factors) {
    for (auto n : n_factors) {
      // B = M1 * N1
      auto element_m = m.first * n.first;
      // C = M2 * N2
      auto element_n = n.second * n.second;
      float compression = 1.0f * origin_element_num / element_m + element_n;
      if (compression > best_compression) {
        best_compression = compression;
        *best_pair_m = m;
        *best_pair_n = n;
      }
    }
  }
  return best_compression;
}

int KPDecomposition::PackMatrixA(const float *data, size_t num, const std::vector<int> &shape,
                                 const std::vector<int> &block_size, float *new_data) {
  auto rows = shape[kRowIndex];
  auto cols = shape[kColIndex];
  auto stride_row = rows / block_size.at(kRowIndex);
  auto stride_col = cols / block_size.at(kColIndex);

  size_t new_col = stride_col * stride_row;
  size_t new_row = block_size.at(kColIndex) * block_size.at(kRowIndex);
  if (new_col * new_row != num) {
    return RET_ERROR;
  }
  size_t total_index = 0;
  for (int col = 0; col < block_size.at(kColIndex); ++col) {
    for (int row = 0; row < block_size.at(kRowIndex); ++row) {
      auto start = row * (stride_row * cols) + stride_col * col;
      for (int c = 0; c < stride_col; ++c) {
        for (int r = 0; r < stride_row; ++r) {
          // rows => cols
          auto index = start + r * cols + c;
          // `index` and `total_index` are guaranteed to be less than `num`
          MS_ASSERT(index < num);
          MS_ASSERT(total_index < num);
          new_data[total_index++] = data[index];
        }
      }
    }
  }
  return RET_OK;
}

double KPDecomposition::CalcFNorm(
  const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix_b,
  const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix_c) {
  auto kp = kroneckerProduct(matrix_b, matrix_c).eval();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> origin_matrix =
    Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(data_, data_size_);
  origin_matrix.resize(shape_.at(kRowIndex), shape_.at(kColIndex));
  auto f_norm = (origin_matrix - kp).norm();
  return f_norm;
}

int KPDecomposition::Decomposition() {
  if (shape_.size() != kMinSize) {
    MS_LOG(ERROR) << "shape size only support 2, but get " << shape_.size();
    return RET_ERROR;
  }

  auto m_factors = IntegerFactorization(shape_.at(kRowIndex));
  auto n_factors = IntegerFactorization(shape_.at(kColIndex));

  std::pair<int, int> best_pair_m;
  std::pair<int, int> best_pair_n;
  auto ratio = FindBestFactorization(m_factors, n_factors, data_size_, &best_pair_m, &best_pair_n);
  MS_LOG(INFO) << "M is [" << best_pair_m.first << "," << best_pair_m.second << "] and N is [" << best_pair_n.first
               << "," << best_pair_n.second << "], kp compression ratio is " << ratio;
  if (ratio <= 1) {
    MS_LOG(WARNING) << "compression ratio " << ratio << " <= 1.";
    return RET_NO_CHANGE;
  }

  auto *pack_matrix_a = static_cast<float *>(malloc(data_size_ * sizeof(float)));
  CHECK_MALLOC_RES(pack_matrix_a, RET_ERROR);
  // Matrix A shape is (B_element, C_element)
  std::vector<int> pack_matrix_shapes{{best_pair_m.first * best_pair_n.first, best_pair_m.second * best_pair_n.second}};
  mat_shape_b_ = {best_pair_m.first, best_pair_n.first};
  mat_shape_c_ = {best_pair_m.second, best_pair_n.second};
  auto ret = PackMatrixA(data_, data_size_, shape_, mat_shape_b_, pack_matrix_a);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Pack MatrixA failed.";
    free(pack_matrix_a);
    return ret;
  }

  auto svd = SVDMatrix(pack_matrix_a, data_size_, pack_matrix_shapes);
  ret = svd.Decomposition();
  free(pack_matrix_a);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SVD decomposition failed.";
    return ret;
  }
  auto sqrt_sigma0 = std::sqrt(svd.GetSigma().data()[0]);
  // vec(B)
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_b =
    sqrt_sigma0 * svd.GetU().col(0).eval();
  // vec(C)
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_c =
    sqrt_sigma0 * svd.GetV().col(0).eval();
  // Matrix(B)
  matrix_b = matrix_b.reshaped(mat_shape_b_.at(kRowIndex), mat_shape_b_.at(kColIndex)).eval();
  // Matrix(C)
  matrix_c = matrix_c.reshaped(mat_shape_c_.at(kRowIndex), mat_shape_c_.at(kColIndex)).eval();

  mat_b_.assign(matrix_b.data(), matrix_b.data() + mat_shape_b_.at(kRowIndex) * mat_shape_b_.at(kColIndex));
  mat_c_.assign(matrix_c.data(), matrix_c.data() + mat_shape_c_.at(kRowIndex) * mat_shape_c_.at(kColIndex));

  auto f_norm = CalcFNorm(matrix_b, matrix_c);
  MS_LOG(INFO) << "Calc FNorm is " << f_norm;

  return RET_OK;
}
}  // namespace mindspore::lite::decomposer
