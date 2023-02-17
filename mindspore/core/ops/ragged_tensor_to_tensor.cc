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
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto default_value_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto shape_arg = input_args[kInputIndex0];
  MS_EXCEPTION_IF_NULL(shape_arg);
  auto output_shape = GetShapeValue(primitive, shape_arg);
  auto values_rank = values_shape.size();
  auto output_shape_rank = output_shape.size();
  auto tensors = input_args[kInputIndex3]->isa<abstract::AbstractTuple>()
                   ? input_args[kInputIndex3]->cast<abstract::AbstractTuplePtr>()->elements()
                   : input_args[kInputIndex3]->cast<abstract::AbstractListPtr>()->elements();
  auto tensors_size = tensors.size();
  auto tensor0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(tensors[0]->BuildShape())[kShape];
  auto tensor0_dim = tensor0_shape.size();
  const auto &row_partition_types_ptr = primitive->GetAttr("row_partition_types");
  MS_EXCEPTION_IF_NULL(row_partition_types_ptr);
  const auto &row_partition_types = GetValue<std::vector<std::string>>(row_partition_types_ptr);
  auto types_size = row_partition_types.size();

  if (IsDynamic(shape_shape) || IsDynamicRank(values_shape) || IsDynamicRank(default_value_shape) ||
      IsDynamicRank(tensor0_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }

  CheckAndConvertUtils::CheckInteger("dimension of 'shape'", SizeToLong(shape_shape.size()), kEqual, 1, prim_name);
  CheckAndConvertUtils::CheckInteger("dimension of 'default_value'", SizeToLong(default_value_shape.size()), kLessThan,
                                     SizeToLong(values_shape.size()), prim_name);

  if (tensors_size != types_size) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the number of row_partition_tensors must be equal to the "
                             << "number of row_partition_types: " << types_size << ", but got " << tensors_size << ".";
  }
  if (row_partition_types[0] == "FIRST_DIM_SIZE") {
    CheckAndConvertUtils::CheckInteger("dimension of row_partition_tensors[0](for 'FIRST_DIM_SIZE')",
                                       SizeToLong(tensor0_dim), kEqual, 0, prim_name);
    if (types_size - 1 + values_rank != output_shape_rank) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', row partition size plus 'values' rank should be equal to 'shape' rank: "
                               << output_shape.size() << ", but got row partition size: " << (types_size - 1)
                               << ", 'values' rank: " << values_rank << ".";
    }
  } else if (row_partition_types[0] == "ROW_SPLITS") {
    CheckAndConvertUtils::CheckInteger("dimension of row_partition_tensors[0](for 'ROW_SPLITS')",
                                       SizeToLong(tensor0_dim), kEqual, 1, prim_name);
    if (types_size + values_rank != output_shape_rank) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', row partition size plus 'values' rank should be equal to 'shape' rank: "
                               << output_shape.size() << ", but got row partition size: " << types_size
                               << ", 'values' rank: " << values_rank << ".";
    }
  } else if (row_partition_types[0] == "VALUE_ROWIDS") {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', cannot handle 'VALUE_ROWIDS' in row_partition_types[0].";
  } else {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', row_partition_types only support 'FIRST_DIM_SIZE', "
                             << "'VALUE_ROWIDS' and 'ROW_SPLITS', but got unknown string: " << row_partition_types[0]
                             << ".";
  }
  for (size_t i = 1; i < types_size; i++) {
    auto tensori_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(tensors[i]->BuildShape())[kShape];
    auto tensori_dim = tensori_shape.size();
    CheckAndConvertUtils::CheckInteger("dimension of row_partition_tensors[" + std::to_string(i) + "]",
                                       SizeToLong(tensori_dim), kEqual, 1, prim_name);
  }
  primitive->AddAttr("num_row_partition_tensors", MakeValue(SizeToLong(tensors_size)));
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

// AG means auto generated
class MIND_API AGRaggedTensorToTensorInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RaggedTensorToTensorInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return RaggedTensorToTensorInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return RaggedTensorToTensorInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(RaggedTensorToTensor, prim::kPrimRaggedTensorToTensor, AGRaggedTensorToTensorInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
