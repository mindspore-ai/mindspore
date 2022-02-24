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

#include "ops/exp.h"
#include <map>
#include <string>
#include <vector>
#include <set>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ExpInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", int64_t(input_args.size()), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_ptr = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  return shape_ptr;
}

TypePtr ExpInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  std::set<TypePtr> valid_params_types = {kTensorType};
  (void)CheckAndConvertUtils::CheckSubClass("x_type", input_args[0]->BuildType(), valid_params_types, prim->name());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
}
}  // namespace
AbstractBasePtr ExpInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(ExpInferShape(primitive, input_args), ExpInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameExp, Exp);
}  // namespace ops
}  // namespace mindspore
