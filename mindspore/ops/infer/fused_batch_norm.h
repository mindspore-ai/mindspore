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

#ifndef MINDSPORE_CORE_OPS_FUSED_BATCH_NORM_H_
#define MINDSPORE_CORE_OPS_FUSED_BATCH_NORM_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFusedBatchNorm = "FusedBatchNorm";
/// \brief FusedBatchNorm defined Enhanced BatchNorm operator prototype.
class OPS_API FusedBatchNorm : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FusedBatchNorm);
  /// \brief Constructor.
  FusedBatchNorm() : BaseOperator(kNameFusedBatchNorm) {
    InitIOName({"x", "scale", "b", "mean", "variance"},
               {"y", "running_mean", "running_variance", "save_mean", "save_inv_variance"});
  }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] mode Define the mode of batchnorm, which is useless.
  /// \param[in] epsilon Define a small value added for numerical stability, which is used to avoid that divisor is
  ///            zero.
  /// \param[in] momentum Define the hyper parameter to compute moving average for running_mean and running_var.
  void Init(const int64_t mode = 0, const float epsilon = 1e-5, const float momentum = 0.1);

  /// \brief Method to set mode attribute, which has been abandoned.
  ///
  /// \param[in] mode Define the mode of batchnorm, which is useless.
  void set_mode(const int64_t mode);

  /// \brief Method to set epsilon attribute.
  ///
  /// \param[in] epsilon Define a small value added for numerical stability, which is used to avoid that divisor is
  ///            zero.
  void set_epsilon(const float epsilon);

  /// \brief Method to set momentum attribute.
  ///
  /// \param[in] momentum Define the hyper parameter to compute moving average for running_mean and running_var.
  void set_momentum(const float momentum);

  /// \brief Method to get mode attribute, which has been abandoned.
  int64_t get_mode() const;

  /// \brief Method to get epsilon attribute.
  ///
  /// \return a small float value.
  float get_epsilon() const;

  /// \brief Method to get momentum attribute.
  ///
  /// \return the hyper parameter .
  float get_momentum() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FUSED_BATCH_NORM_H_
