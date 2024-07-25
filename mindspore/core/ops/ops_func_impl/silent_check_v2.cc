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

#include "ops/ops_func_impl/silent_check_v2.h"

#include <memory>
#include <utility>
#include <vector>
#include "abstract/dshape.h"
#include "ir/dtype/tensor_type.h"

namespace mindspore {
namespace ops {
BaseShapePtr SilentCheckV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_grad_shape = input_args[kIndex1]->GetShape();
  const auto &sfda_shape = input_args[kIndex2]->GetShape();
  const auto &step_shape = input_args[kIndex3]->GetShape();
  auto result_shape = std::make_shared<abstract::TensorShape>(std::vector<int64_t>{});
  std::vector<abstract::BaseShapePtr> output_list{input_grad_shape->Clone(), sfda_shape->Clone(), step_shape->Clone(),
                                                  std::move(result_shape)};
  return std::make_shared<abstract::TupleShape>(std::move(output_list));
}

TypePtr SilentCheckV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_grad_type = input_args[kIndex1]->GetType();
  const auto &sfda_type = input_args[kIndex2]->GetType();
  const auto &step_type = input_args[kIndex3]->GetType();
  auto result_type = std::make_shared<TensorType>(kInt32);
  std::vector<TypePtr> type_tuple{input_grad_type, sfda_type, step_type, std::move(result_type)};
  return std::make_shared<Tuple>(std::move(type_tuple));
}
}  // namespace ops
}  // namespace mindspore
