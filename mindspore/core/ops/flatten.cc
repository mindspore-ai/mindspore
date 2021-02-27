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
#include "ops/flatten.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto flatten_prim = primitive->cast<PrimFlattenPtr>();
  MS_EXCEPTION_IF_NULL(flatten_prim);
  auto prim_name = flatten_prim->name();
  CheckAndConvertUtils::CheckInteger("input args size", input_args.size(), kGreaterEqual, 1, prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto prod = 1;
  int64_t size = x_shape.size();
  for (int64_t i = 1; i < size; i++) {
    prod = prod * x_shape[i];
  }
  std::vector<int64_t> out_shape = {x_shape[0], prod};
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  const std::set<TypePtr> valid_types = {TypeIdToType(kObjectTypeTensorType)};
  CheckAndConvertUtils::CheckSubClass("infer type", input_args[0]->BuildType(), valid_types, prim->name());
  return infer_type;
}
}  // namespace

AbstractBasePtr FlattenInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameFlatten, Flatten);
}  // namespace ops
}  // namespace mindspore
