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

#ifndef MINDSPORE_CORE_OPS_SYNC_BATCH_NORM_H_
#define MINDSPORE_CORE_OPS_SYNC_BATCH_NORM_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSyncBatchNorm = "SyncBatchNorm";
/// \brief Calculates SyncBatchNorm.
/// Refer to Python API @ref mindspore.nn.SyncBatchNorm for more details.
class OPS_API SyncBatchNorm : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SyncBatchNorm);
  /// \brief Constructor.
  SyncBatchNorm() : BaseOperator(kNameSyncBatchNorm) {
    InitIOName({"input_x", "scale", "bias", "mean", "variance"},
               {"output_x", "updated_scale", "updated_bias", "updated_moving_mean", "updated_moving_variance"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.nn.SyncBatchNorm for the inputs.
  void Init(const float epsilon = 1e-5, const float momentum = 0.1, const std::string group = "sync_bn_group0",
            const int64_t device_num = 2);
  /// \brief Set epsilon.
  void set_epsilon(const float epsilon);
  /// \brief Set momentum.
  void set_momentum(const float momentum);
  /// \brief Set group.
  void set_group(const std::string group);
  /// \brief Set device_num.
  void set_device_num(const int64_t device_num);
};

OPS_API abstract::AbstractBasePtr SyncBatchNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SYNC_BATCH_NORM_H_
