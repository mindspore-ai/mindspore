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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_DECOMPOSER_SVD_MATRIX_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_DECOMPOSER_SVD_MATRIX_H_

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace mindspore::lite::decomposer {
class SVDMatrix {
 public:
  SVDMatrix(const float *data_ptr, size_t data_size, const std::vector<int> shape)
      : data_(data_ptr, data_ptr + data_size), shape_(shape) {}

  ~SVDMatrix() = default;

  int DoSVDWithRank(int rank);
  // err means matrix Frobenius Norm
  int DoSVDWithErr(float err);

  std::vector<float> GetMatA() { return mat_a_; }
  std::vector<int> GetMatShapeA() { return mat_shape_a_; }
  std::vector<float> GetMatB() { return mat_b_; }
  std::vector<int> GetMatShapeB() { return mat_shape_b_; }

 private:
  void SVDCompress();
  int GetBestRank(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &orgin_matrix,
                  const Eigen::BDCSVD<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &svd,
                  float err);
  void TruncateSVD(const Eigen::BDCSVD<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &svd);
  float ComputeReduceMemoryRatio(const int rank);

  std::vector<float> data_;
  std::vector<int> shape_;
  std::vector<float> mat_a_;
  std::vector<int> mat_shape_a_;
  std::vector<float> mat_b_;
  std::vector<int> mat_shape_b_;
  int rank_ = -1;
};
}  // namespace mindspore::lite::decomposer
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_DECOMPOSER_SVD_MATRIX_H_
