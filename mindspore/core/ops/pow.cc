/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/pow.h"

#include <set>
#include <utility>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr PowInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x1_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto x2_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto x1_shape = x1_shape_map[kShape];
  auto x2_shape = x2_shape_map[kShape];
  auto x1_min_shape = x1_shape_map[kMinShape];
  auto x1_max_shape = x1_shape_map[kMaxShape];
  auto x2_min_shape = x2_shape_map[kMinShape];
  auto x2_max_shape = x2_shape_map[kMaxShape];
  if (x1_shape == x2_shape) {
    return std::make_shared<abstract::Shape>(x1_shape, x1_min_shape, x1_max_shape);
  }
  auto broadcast_shape = CalBroadCastShape(x1_shape, x2_shape, prim_name);
  auto min_broadcast_shape = CalBroadCastShape(x1_min_shape, x2_min_shape, prim_name);
  auto max_broadcast_shape = CalBroadCastShape(x1_max_shape, x2_max_shape, prim_name);
  return std::make_shared<abstract::Shape>(broadcast_shape, min_broadcast_shape, max_broadcast_shape);
}

TypePtr PowInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  TypePtr x1_type = input_args[kInputIndex0]->BuildType();
  TypePtr x2_type = input_args[kInputIndex1]->BuildType();
  std::set<TypePtr> complex_valid_types = {kComplex64, kComplex128};
  if (complex_valid_types.count(x1_type) || complex_valid_types.count(x2_type)) {
    std::map<std::pair<TypePtr, TypePtr>, TypePtr> type_infer_dict;
    (void)type_infer_dict.emplace(std::make_pair(kComplex64, kComplex64), kComplex64);
    (void)type_infer_dict.emplace(std::make_pair(kComplex128, kComplex128), kComplex128);
    (void)type_infer_dict.emplace(std::make_pair(kComplex128, kComplex64), kComplex128);
    (void)type_infer_dict.emplace(std::make_pair(kComplex64, kComplex128), kComplex128);
    if (!type_infer_dict.count(std::make_pair(x1_type, x2_type))) {
      MS_EXCEPTION(TypeError) << "For '" << prim->name()
                              << "', Complex math binary op expecting Tensor [complex64, complex64],"
                              << "[complex64, float32], [float32, complex64], [complex128, complex128],"
                              << "[complex128, float64], [float64, complex128],"
                              << "but got : " << x1_type->meta_type() << "," << x2_type->meta_type();
      return type_infer_dict[std::make_pair(x1_type, x2_type)];
    }
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x1", x1_type);
  (void)types.emplace("x2", x2_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
  return x1_type;
}
}  // namespace
AbstractBasePtr PowInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t kInputNum = 2;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, prim_name);
  auto infer_type = PowInferType(primitive, input_args);
  auto infer_shape = PowInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Pow, prim::kPrimPow, PowInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
