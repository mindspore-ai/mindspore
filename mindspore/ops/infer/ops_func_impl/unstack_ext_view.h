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

#ifndef MINDSPORE_CORE_OPS_OP_FUNC_IMPL_UNSTACK_EXT_VIEW_H
#define MINDSPORE_CORE_OPS_OP_FUNC_IMPL_UNSTACK_EXT_VIEW_H

#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"
#include "mindapi/base/macros.h"

namespace mindspore {
namespace ops {
class OPS_API UnstackExtViewFuncImpl : public OpFuncImpl {
 public:
  UnstackExtViewFuncImpl() = default;
  ~UnstackExtViewFuncImpl() = default;

  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OP_FUNC_IMPL_UNSTACK_EXT_VIEW_H
