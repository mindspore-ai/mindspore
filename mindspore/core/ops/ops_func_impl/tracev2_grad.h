/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_OP_FUNC_IMPL_TRACEV2_GRAD_H
#define MINDSPORE_CORE_OPS_OP_FUNC_IMPL_TRACEV2_GRAD_H

#include <memory>
#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore::ops {
class TraceV2GradFuncImpl : public OpFuncImpl {
 public:
  TraceV2GradFuncImpl() = default;
  ~TraceV2GradFuncImpl() = default;

  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
};

using TraceV2GradFuncImplPtr = std::shared_ptr<TraceV2GradFuncImpl>;
}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_OP_FUNC_IMPL_TRACEV2_GRAD_H
