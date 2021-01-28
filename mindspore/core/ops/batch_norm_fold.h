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
#ifndef MINDSPORE_CORE_OPS_BATCH_NORM_FOLD_H_
#define MINDSPORE_CORE_OPS_BATCH_NORM_FOLD_H_
#include <memory>
#include <vector>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBatchNormFold = "BatchNormFold";
class BatchNormFold : public PrimitiveC {
 public:
  BatchNormFold() : PrimitiveC(kNameBatchNormFold) {
    InitIOName({"x", "mean", "variance", "global_step"}, {"batch_mean", "batch_std", "running_mean", "running_std"});
  }
  ~BatchNormFold() = default;
  MS_DECLARE_PARENT(BatchNormFold, PrimitiveC);
  void Init(const float momentum = 0.9, const float epsilon = 1e-5, const bool is_training = true,
            const int64_t freeze_bn = 0);
  void set_momentum(const float momentum);
  void set_epsilon(const float epsilon);
  void set_is_training(const bool is_training);
  void set_freeze_bn(const int64_t freeze_bn);

  float get_momentum() const;
  float get_epsilon() const;
  bool get_is_training() const;
  int64_t get_freeze_bn() const;
};

AbstractBasePtr BatchNormFoldInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args);
using PrimBatchNormFoldPtr = std::shared_ptr<BatchNormFold>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BATCH_NORM_FOLD_H_
