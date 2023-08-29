/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_BATCH_NORM_GRAD_WITH_ADD_AND_ACTIVAATION_H_
#define MINDSPORE_CORE_OPS_BATCH_NORM_GRAD_WITH_ADD_AND_ACTIVAATION_H_

#include <vector>
#include <set>

#include "ops/ops_func_impl/batch_norm_grad.h"

namespace mindspore {
namespace ops {
class MIND_API BatchNormGradWithAddAndActivationFuncImpl : public BatchNormGradFuncImpl {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

  std::set<int64_t> GetValueDependArgIndices() const override { return {9, 10}; }

 protected:
  size_t GetAttrPosZero() const override { return 8; }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BATCH_NORM_GRAD_WITH_ADD_AND_ACTIVAATION_H_
