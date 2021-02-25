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

#include <set>
#include <memory>
#include "ops/gather.h"

namespace mindspore {
namespace ops {
AbstractBasePtr GatherInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInteger("gather_infer", input_args.size(), kEqual, 3, prim_name);

  // Infer type
  auto x_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  std::set<TypePtr> valid_x_type = {TypeIdToType(kObjectTypeTensorType)};
  CheckAndConvertUtils::CheckSubClass("x_type", input_args[0]->BuildType(), valid_x_type, prim_name);
  const std::set<TypeId> valid_index_types = {kNumberTypeInt32, kNumberTypeInt64};
  CheckAndConvertUtils::CheckTensorTypeValid("index_type", input_args[2]->BuildType(), valid_index_types, prim_name);
  std::set<TypePtr> valid_dim_type = {TypeIdToType(kNumberTypeInt32), TypeIdToType(kNumberTypeInt64)};
  CheckAndConvertUtils::CheckSubClass("dim_type", input_args[1]->BuildType(), valid_dim_type, prim_name);

  // Infer shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto index_shape = CheckAndConvertUtils::ConvertShapePtrToShape("dim_shape", input_args[2]->BuildShape(), prim_name);
  CheckAndConvertUtils::Check("x_rank", x_shape.size(), kEqual, "index_rank", index_shape.size(), prim_name);

  return std::make_shared<abstract::AbstractTensor>(x_type, index_shape);
}
REGISTER_PRIMITIVE_C(kNameGather, Gather);
}  // namespace ops
}  // namespace mindspore
