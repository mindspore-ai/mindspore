/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/tensor_copy_slices.h"
#include <functional>
#include <iostream>
#include <set>
#include <map>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kTensorCopySlicesInputnDim = 5;

abstract::ShapePtr TensorCopySlicesInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr TensorCopySlicesInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_type = input_args[0]->BuildType();
  auto value_type = input_args[1]->BuildType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", x_type);
  (void)types.emplace("value", value_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, all_types, prim_name);
  return x_type;
}
}  // namespace

AbstractBasePtr TensorCopySlicesInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (auto &input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kTensorCopySlicesInputnDim, prim_name);
  auto infer_type = TensorCopySlicesInferType(primitive, input_args);
  auto infer_shape = TensorCopySlicesInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(TensorCopySlices, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(TensorCopySlices, prim::kPrimTensorCopySlices, TensorCopySlicesInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
