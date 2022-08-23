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
#include <map>
#include <algorithm>
#include <numeric>
#include "tools/converter/decomposer/svd_matrix.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore::lite::decomposer {
namespace {
constexpr int kRowIndex = 0;
constexpr int kColIndex = 1;
constexpr int kMinSize = 2;
}  // namespace
int SVDMatrix::CompressWithRank(int rank) {
  if (rank <= 0) {
    MS_LOG(ERROR) << " The rank = " << rank << ", it is too small";
    return RET_ERROR;
  }
  if (rank > std::min(shape_[kRowIndex], shape_[kColIndex])) {
    MS_LOG(ERROR) << " The rank = " << rank
                  << ", it is bigger than min(row, col) = " << std::min(shape_[kRowIndex], shape_[kColIndex]);
    return RET_ERROR;
  }
  rank_ = rank;

  // Clear mat
  mat_a_.clear();
  mat_b_.clear();

  Decomposition();
  TruncateSVD(svd_);

  return RET_OK;
}

int SVDMatrix::Decomposition() {
  if (shape_.size() != kMinSize) {
    MS_LOG(ERROR) << "shape size only support 2, but get " << shape_.size();
    return RET_ERROR;
  }
  int row = shape_[kRowIndex];
  int col = shape_[kColIndex];

  // Convert std::vector to Eigen::MatrixXf
  origin_matrix_ = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(data_.data(), data_.size()).eval();
  origin_matrix_.resize(row, col);

  // Singular Value Decomposition
  svd_ = Eigen::BDCSVD<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
    origin_matrix_, Eigen::ComputeThinU | Eigen::ComputeThinV);

  U_ = svd_.matrixU();
  V_ = svd_.matrixV();
  Sigma_ = svd_.singularValues().asDiagonal();
  return RET_OK;
}

int SVDMatrix::CompressWithFNorm(float f_norm) {
  // clear mat
  mat_a_.clear();
  mat_b_.clear();
  Decomposition();
  if (GetBestRank(origin_matrix_, svd_, f_norm) == RET_ERROR) {
    return RET_ERROR;
  }
  TruncateSVD(svd_);

  return RET_OK;
}

void SVDMatrix::TruncateSVD(
  const Eigen::BDCSVD<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &svd) {
  int row = shape_[kRowIndex];
  int col = shape_[kColIndex];
  // Compute U after rank-based truncation
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> layer_a = svd.matrixU().block(0, 0, row, rank_);
  std::vector<float> u(layer_a.data(), layer_a.data() + row * rank_);
  mat_a_.assign(u.begin(), u.end());
  mat_shape_a_ = {row, rank_};

  // Compute W * VT after rank-based truncation
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> layer_b =
    (svd.singularValues().asDiagonal() * svd.matrixV().transpose()).block(0, 0, rank_, col);
  std::vector<float> wvt(layer_b.data(), layer_b.data() + rank_ * col);
  mat_b_.assign(wvt.begin(), wvt.end());
  mat_shape_b_ = {rank_, col};
}

int SVDMatrix::GetBestRank(
  const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &orgin_matrix,
  const Eigen::BDCSVD<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &svd, float err) {
  int row = shape_[kRowIndex];
  int col = shape_[kColIndex];

  // Get the best rank from big to small
  int best_rank = -1;
  for (int rank = std::min(row, col); rank > 0; rank--) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> layer_a =
      svd.matrixU().block(0, 0, row, rank);

    // Compute W * VT after rank-based truncation
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> layer_b =
      (svd.singularValues().asDiagonal() * svd.matrixV().transpose()).block(0, 0, rank, col);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> reconstructMat = layer_a * layer_b;
    float recon_error = (reconstructMat - orgin_matrix).norm();
    if (recon_error >= err) {
      break;
    }
    best_rank = rank;
  }
  if (best_rank == -1) {
    MS_LOG(ERROR) << "Not Found the best rank";
    return RET_ERROR;
  }
  float reduce_memory_ratio = ComputeReduceMemoryRatio(best_rank);
  if (reduce_memory_ratio < 0) {
    MS_LOG(ERROR) << "ComputeReduceMemoryRatio =" << reduce_memory_ratio
                  << ", In this err case, SVD compression has no effect, the user needs to increase the err_ value";
    return RET_ERROR;
  }
  MS_LOG(INFO) << "The compute reduce memory ratio is " << reduce_memory_ratio;
  // Set best rank
  rank_ = best_rank;
  MS_LOG(INFO) << "The best rank is " << best_rank;

  return RET_OK;
}

float SVDMatrix::ComputeReduceMemoryRatio(const int rank) {
  int rows = shape_[kRowIndex];
  int cols = shape_[kColIndex];

  size_t original_size = rows * cols;
  size_t reduced_size = (rows + cols) * rank;
  float ratio = 1 - (static_cast<float>(reduced_size) / original_size);
  return ratio;
}
}  // namespace mindspore::lite::decomposer
