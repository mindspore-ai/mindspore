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

#ifndef MINDSPORE_CORE_C_OPS_BATCH_NORMAL_H_
#define MINDSPORE_CORE_C_OPS_BATCH_NORMAL_H_
#include <map>
#include <vector>
#include <memory>
#include <string>
#include "c_ops/op_utils.h"
#include "c_ops/primitive_c.h"
#include "abstract/abstract_value.h"

namespace mindspore {
constexpr auto kNameBatchNorm = "BatchNorm";
class BatchNorm : public PrimitiveC {
 public:
  BatchNorm() : PrimitiveC(kNameBatchNorm) {
    InitIOName({"x", "scale", "offset", "mean", "variance"},
               {"y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"});
  }
  ~BatchNorm() = default;
  MS_DECLARE_PARENT(BatchNorm, PrimitiveC);
  void Init(bool is_training = false, float epsilon = 1e-5, const Format &format = NCHW);
  void set_is_training(bool is_training);
  void set_epsilon(float epsilon);
  void set_format(const Format &format);
  bool get_is_trainging();
  float get_epsilon();
  Format get_format() const;
};

AbstractBasePtr BatchNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args);
using PrimBatchNormPtr = std::shared_ptr<BatchNorm>;

}  // namespace mindspore

#endif  // MINDSPORE_CORE_C_OPS_BatchNorm_H_
