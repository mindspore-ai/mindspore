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
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFusedBatchNorm = "FusedBatchNorm";
class FusedBatchNorm : public PrimitiveC {
 public:
  FusedBatchNorm() : PrimitiveC(kNameFusedBatchNorm) {
    InitIOName({"x", "scale", "b", "mean", "variance"},
               {"y", "running_mean", "running_variance", "save_mean", "save_inv_variance"});
  }
  ~FusedBatchNorm() = default;
  MS_DECLARE_PARENT(FusedBatchNorm, PrimitiveC);
  void Init(const int64_t mode = 0, const float epsilon = 1e-5, const float momentum = 0.1);
  void set_mode(const int64_t mode);
  void set_epsilon(const float epsilon);
  void set_momentum(const float momentum);
  int64_t get_mode() const;
  float get_epsilon() const;
  float get_momentum() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FUSED_BATCH_NORM_H_
