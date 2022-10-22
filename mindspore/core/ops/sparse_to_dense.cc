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

#include <memory>
#include "ops/sparse_to_dense.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kSparseToDenseInputMaxDim = 2;
constexpr int64_t kSparseToDenseInputMinDim = 1;
constexpr int64_t kSparseToDenseInputsNum = 3;
constexpr int64_t kNumZero = 0;

abstract::ShapePtr SparseToDenseInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto indice_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];

  std::vector<ShapeVector> all_shapes = {indice_shape, values_shape};
  auto is_dynamic = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamic);

  (void)CheckAndConvertUtils::CheckInteger("dimension of 'values'", values_shape.size(), kEqual,
                                           kSparseToDenseInputMinDim, op_name);
  if (!is_dynamic) {
    (void)CheckAndConvertUtils::CheckInteger("dimension of 'indices'", indice_shape.size(), kEqual,
                                             kSparseToDenseInputMaxDim, op_name);
    (void)CheckAndConvertUtils::CheckInteger("batch of 'indices'", indice_shape[kInputIndex0], kEqual,
                                             values_shape[kInputIndex0], op_name);
  }
  auto shape_arg = input_args[kInputIndex2];
  MS_EXCEPTION_IF_NULL(shape_arg);
  auto output_shape = GetShapeValue(primitive, shape_arg);
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr SparseToDenseInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto indice_type = input_args[kInputIndex0]->BuildType();
  auto values_type = input_args[kInputIndex1]->BuildType();

  const std::set<TypePtr> valid_types = {kInt64, kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeSame({{"indices", indice_type}}, valid_types, op_name);
  const std::set<TypePtr> valid_types_value = {kInt64,  kInt32, kInt16,   kInt8,    kUInt64,  kUInt32,
                                               kUInt16, kUInt8, kFloat16, kFloat32, kFloat64, kBool};
  std::map<std::string, TypePtr> types_value;
  (void)types_value.insert({"values", values_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types_value, valid_types_value, op_name);
  return values_type;
}
}  // namespace

abstract::AbstractBasePtr SparseToDenseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kSparseToDenseInputsNum,
                                     primitive->name());
  auto infer_type = SparseToDenseInferType(primitive, input_args);
  auto infer_shape = SparseToDenseInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(SparseToDense, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(SparseToDense, prim::kPrimSparseToDense, SparseToDenseInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
