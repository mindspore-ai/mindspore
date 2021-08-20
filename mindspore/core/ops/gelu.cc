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
#include "ops/gelu.h"
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("gelu infer", SizeToLong(input_args.size()), kEqual, 1, op_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto in_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  if (min_shape.size() != 0 && max_shape.size() != 0) {
    return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
  }
  return std::make_shared<abstract::Shape>(in_shape);
}
TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto op_name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("gelu infer", SizeToLong(input_args.size()), kEqual, 1, op_name);
  std::map<std::string, TypePtr> types;
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  MS_EXCEPTION_IF_NULL(input_args[0]);
  (void)types.emplace("x", input_args[0]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace
AbstractBasePtr GeLUInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(InferShape(primitive, input_args), InferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(GeLU, prim::kPrimGeLU, GeLUInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
