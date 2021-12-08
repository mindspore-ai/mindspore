/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input args size", SizeToLong(input_args.size()), kGreaterEqual, 1,
                                           prim_name);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  int64_t prod = 1;
  size_t size = x_shape.size();
  for (size_t i = 1; i < size; i++) {
    if (x_shape[i] == -1) {
      prod = -1;
      break;
    }
    prod = prod * x_shape[i];
  }
  ShapeVector out_shape = {x_shape[0], prod};
  if (min_shape.empty() || max_shape.empty()) {
    return std::make_shared<abstract::Shape>(out_shape);
  }
  int64_t min_prod = 1;
  size_t min_size = min_shape.size();
  for (size_t i = 1; i < min_size; i++) {
    min_prod = min_prod * min_shape[i];
  }
  ShapeVector out_min_shape = {min_shape[0], min_prod};
  int64_t max_prod = 1;
  size_t max_size = max_shape.size();
  for (size_t i = 1; i < max_size; i++) {
    max_prod = max_prod * max_shape[i];
  }
  ShapeVector out_max_shape = {max_shape[0], max_prod};
  return std::make_shared<abstract::Shape>(out_shape, out_min_shape, out_max_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  const std::set<TypePtr> valid_types = {kTensorType};
  (void)CheckAndConvertUtils::CheckSubClass("infer type", input_args[0]->BuildType(), valid_types, prim->name());
  return infer_type;
}
}  // namespace

AbstractBasePtr FlattenInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = InferShape(primitive, input_args);
  auto infer_shape = InferType(primitive, input_args);
  return abstract::MakeAbstract(infer_type, infer_shape);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Flatten, prim::kPrimFlatten, FlattenInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
