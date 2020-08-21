/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_RANDOM_COLOR_OP_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_RANDOM_COLOR_OP_H

#include <memory>
#include <random>
#include <vector>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {

/// \class RandomColorOp random_color_op.h
/// \brief Blends an image with its grayscale version with random weights
///        t and 1 - t generated from a given range.
///        If the range is trivial then the weights are determinate and
///        t equals the bound of the interval
class RandomColorOp : public TensorOp {
 public:
  RandomColorOp() = default;

  ~RandomColorOp() = default;
  /// \brief Constructor
  /// \param[in] t_lb lower bound for the random weights
  /// \param[in] t_ub upper bound for the random weights
  RandomColorOp(float t_lb, float t_ub);
  /// \brief the main function performing computations
  /// \param[in] in 2- or 3- dimensional tensor representing an image
  /// \param[out] out 2- or 3- dimensional tensor representing an image
  /// with the same dimensions as in
  Status Compute(const std::shared_ptr<Tensor> &in, std::shared_ptr<Tensor> *out) override;
  /// \brief returns the name of the op
  std::string Name() const override { return kRandomColorOp; }

 private:
  std::mt19937 rnd_;
  std::uniform_real_distribution<float> dist_;
  float t_lb_;
  float t_ub_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_RANDOM_COLOR_OP_H
