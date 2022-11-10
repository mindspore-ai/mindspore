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
#include "ops/ragged_tensor_to_tensor.h"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>
#include <algorithm>
#include <set>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr RaggedTensorToTensorInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  CheckAndConvertUtils::CheckInteger("dimension of 'shape'", SizeToLong(shape_shape.size()), kEqual, 1, prim_name);
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto default_value_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  CheckAndConvertUtils::CheckInteger("dimension of 'default_value'", SizeToLong(default_value_shape.size()), kLessThan,
                                     SizeToLong(values_shape.size()), prim_name);

  auto shape_arg = input_args[kInputIndex0];
  MS_EXCEPTION_IF_NULL(shape_arg);
  auto output_shape = GetShapeValue(primitive, shape_arg);
  auto row_partition_tensors_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  primitive->AddAttr("num_row_partition_tensors", MakeValue(SizeToLong(row_partition_tensors_shape.size())));
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr RaggedTensorToTensorInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  std::map<std::string, TypePtr> types;
  std::set<TypePtr> valid_types = {kBool, kInt8, kUInt8, kInt16, kUInt16, kInt32, kInt64, kFloat64, kFloat, kFloat16};
  TypePtr shape_type = input_args[kInputIndex0]->BuildType();
  TypePtr values_type = input_args[kInputIndex1]->BuildType();
  TypePtr default_value_type = input_args[kInputIndex2]->BuildType();
  (void)types.emplace("values", values_type);
  (void)types.emplace("default_value", default_value_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("shape", shape_type, {kInt64, kInt32}, primitive->name());
  auto tensors_arg = input_args[kInputIndex3];
  if (!tensors_arg->isa<abstract::AbstractTuple>() && !tensors_arg->isa<abstract::AbstractList>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the row_partition_tensors must be list or tuple of tensors.";
  }
  auto tensors = tensors_arg->isa<abstract::AbstractTuple>()
                   ? tensors_arg->cast<abstract::AbstractTuplePtr>()->elements()
                   : tensors_arg->cast<abstract::AbstractListPtr>()->elements();
  const std::set<TypePtr> valid_tensor_types = {kInt32, kInt64};
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto input_dtype = tensors[i]->BuildType();
    (void)CheckAndConvertUtils::CheckTypeValid("row_partition_tensors", input_dtype, valid_tensor_types, prim_name);
  }
  return values_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(RaggedTensorToTensor, BaseOperator);
AbstractBasePtr RaggedTensorToTensorInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = RaggedTensorToTensorInferType(primitive, input_args);
  auto infer_shape = RaggedTensorToTensorInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(RaggedTensorToTensor, prim::kPrimRaggedTensorToTensor, RaggedTensorToTensorInfer, nullptr,
                             true);
REGISTER_HOST_DEPENDS(kNameRaggedTensorToTensor, {0});
}  // namespace ops
}  // namespace mindspore
