/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_BATCH_NORMAL_H_
#define MINDSPORE_CORE_OPS_BATCH_NORMAL_H_
#include <map>
#include <vector>
#include <memory>
#include <string>
#include "ops/base_operator.h"
#include "mindapi/base/format.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBatchNorm = "BatchNorm";
constexpr auto kNameBatchNormWithActivation = "BatchNormWithActivation";
constexpr auto kNameBatchNormWithAddAndActivation = "BatchNormWithAddAndActivation";
/// \brief Batch Normalization for input data and updated parameters.
/// Refer to Python API @ref mindspore.ops.BatchNorm for more details.
class MIND_API BatchNorm : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(BatchNorm);
  /// \brief Constructor.
  BatchNorm() : BaseOperator(kNameBatchNorm) {
    InitIOName({"x", "scale", "offset", "mean", "variance"},
               {"y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"});
  }
  explicit BatchNorm(const std::string kernel_name) : BaseOperator(kernel_name) {
    InitIOName({"x", "scale", "offset", "mean", "variance"},
               {"y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.BatchNorm for the inputs.
  void Init(const bool is_training = false, const float epsilon = 1e-5, const float momentun = 0.1,
            const Format &format = NCHW);
  /// \brief Set is_training.
  void set_is_training(const bool is_training);
  /// \brief Set epsilon.
  void set_epsilon(const float epsilon);
  /// \brief Set format.
  void set_format(const Format &format);
  /// \brief Set momentum.
  void set_momentum(const float momentum);
  /// \brief Get is_training.
  ///
  /// \return is_training.
  bool get_is_training() const;
  /// \brief Get epsilon.
  ///
  /// \return epsilon.
  float get_epsilon() const;
  /// \brief Get format.
  ///
  /// \return format.
  Format get_format() const;
  /// \brief Get momentum.
  ///
  /// \return momentum.
  float get_momentum() const;
};

class MIND_API BatchNormWithActivation : public BatchNorm {
 public:
  MIND_API_BASE_MEMBER(BatchNormWithActivation);
  /// \brief Constructor.
  BatchNormWithActivation() : BatchNorm(kNameBatchNormWithActivation) {
    InitIOName({"x", "scale", "offset", "mean", "variance"},
               {"y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"});
  }
};

class MIND_API BatchNormWithAddAndActivation : public BatchNorm {
 public:
  MIND_API_BASE_MEMBER(BatchNormWithAddAndActivation);
  /// \brief Constructor.
  BatchNormWithAddAndActivation() : BatchNorm(kNameBatchNormWithAddAndActivation) {
    InitIOName({"x", "scale", "offset", "mean", "variance", "z"},
               {"y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"});
  }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BatchNorm_H_
