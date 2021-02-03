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

#ifndef MINDSPORE_CORE_OPS_BATCH_NORMAL_H_
#define MINDSPORE_CORE_OPS_BATCH_NORMAL_H_
#include <map>
#include <vector>
#include <memory>
#include <string>
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBatchNorm = "BatchNorm";
class BatchNorm : public PrimitiveC {
 public:
  BatchNorm() : PrimitiveC(kNameBatchNorm) {
    InitIOName({"x", "scale", "offset", "mean", "variance"},
               {"y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"});
  }
  ~BatchNorm() = default;
  MS_DECLARE_PARENT(BatchNorm, PrimitiveC);
  void Init(const bool is_training = false, const float epsilon = 1e-5, const float momentun = 0.1,
            const Format &format = NCHW);
  void set_is_training(const bool is_training);
  void set_epsilon(const float epsilon);
  void set_format(const Format &format);
  void set_momentum(const float momentum);
  bool get_is_training() const;
  float get_epsilon() const;
  Format get_format() const;
  float get_momentum() const;
};

AbstractBasePtr BatchNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args);
using PrimBatchNormPtr = std::shared_ptr<BatchNorm>;

}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BatchNorm_H_
