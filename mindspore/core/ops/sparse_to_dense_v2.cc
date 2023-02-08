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

#include "ops/sparse_to_dense_v2.h"
#include <set>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseToDenseV2InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const size_t Indiceselement = 2;
  const size_t OutShapeSize = 1;
  const size_t ValuesSize = 1;
  const size_t DefaultSize = 0;
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);
  auto indices_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto output_shape_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto values_shape_ptr = input_args[kInputIndex2]->BuildShape();
  auto default_value_shape_ptr = input_args[kInputIndex3]->BuildShape();
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];
  auto output_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(output_shape_shape_ptr)[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(values_shape_ptr)[kShape];
  auto default_value_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(default_value_shape_ptr)[kShape];
  (void)CheckAndConvertUtils::CheckInteger("indices dimension", SizeToLong(indices_shape.size()), kLessEqual,
                                           SizeToLong(Indiceselement), prim_name);
  (void)CheckAndConvertUtils::CheckInteger("outshape dimension", SizeToLong(output_shape_shape.size()), kEqual,
                                           SizeToLong(OutShapeSize), prim_name);
  (void)CheckAndConvertUtils::CheckInteger("values dimension", SizeToLong(values_shape.size()), kLessEqual,
                                           SizeToLong(ValuesSize), prim_name);
  (void)CheckAndConvertUtils::CheckInteger("default_value dimension", SizeToLong(default_value_shape.size()), kEqual,
                                           SizeToLong(DefaultSize), prim_name);
  size_t output_shape_numelement = LongToSize(output_shape_shape[0]);
  auto output_shape = input_args[1]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(output_shape);
  auto output_shape_value_ = output_shape->BuildValue();
  MS_EXCEPTION_IF_NULL(output_shape_value_);
  auto output_shape_tensor = output_shape_value_->cast<tensor::TensorPtr>();
  auto output_shape_type = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(output_shape_type);
  auto output_shape_type_id = output_shape_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(output_shape_type_id);
  auto output_shape_type_element = output_shape_type_id->element();
  MS_EXCEPTION_IF_NULL(output_shape_type_element);
  std::vector<int64_t> y_shape;
  if (!input_args[1]->BuildValue()->isa<AnyValue>() && !input_args[1]->BuildValue()->isa<None>()) {
    if (indices_shape.size() == 0) {
      if (values_shape.size() != 0 && values_shape[0] != 1) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the indices_shape[0] is 1"
                                 << " should match the the values element " << values_shape[0] << ".";
      }
    } else {
      if (values_shape.size() != 0) {
        if (indices_shape[0] != values_shape[0]) {
          MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the indices_shape[0] " << indices_shape[0]
                                   << " should match the the values element " << values_shape[0] << ".";
        }
      }
    }
    if (output_shape_type_element->type_id() == kNumberTypeInt32) {
      auto output_shape_data = static_cast<int32_t *>(output_shape_tensor->data_c());
      for (size_t i = 0; i < output_shape_numelement; ++i) {
        if (output_shape_data[i] > 0) {
          y_shape.push_back(output_shape_data[i]);
        } else {
          MS_EXCEPTION(ValueError) << "For '" << prim_name << "', each dimension must be greater than 0. But got the "
                                   << i << "th dimension of output " << output_shape_data[i] << ".";
        }
      }
    } else if (output_shape_type_element->type_id() == kNumberTypeInt64) {
      auto output_shape_data = static_cast<int64_t *>(output_shape_tensor->data_c());
      for (size_t i = 0; i < output_shape_numelement; ++i) {
        if (output_shape_data[i] > 0) {
          y_shape.push_back(output_shape_data[i]);
        } else {
          MS_EXCEPTION(ValueError) << "For '" << prim_name << "', each dimension must be greater than 0. But got the "
                                   << i << "th dimension of output " << output_shape_data[i] << ".";
        }
      }
    }
    return std::make_shared<abstract::Shape>(y_shape);
  } else {
    const uint32_t input_shapes = static_cast<uint32_t>(std::pow(max_length, 1.0 / SizeToInt(output_shape_numelement)));
    ShapeVector shape_min;
    ShapeVector shape_max;
    for (size_t i = 0; i < output_shape_numelement; i++) {
      y_shape.push_back(abstract::Shape::SHP_ANY);
      shape_min.push_back(0);
      shape_max.push_back(input_shapes);
    }
    return std::make_shared<abstract::Shape>(y_shape, shape_min, shape_max);
  }
}

TypePtr SparseToDenseV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto infer_type_indices = input_args[kInputIndex0]->BuildType();
  auto infer_type_output_shape = input_args[kInputIndex1]->BuildType();
  auto infer_type_values = input_args[kInputIndex2]->BuildType();
  auto infer_type_default_value = input_args[kInputIndex3]->BuildType();
  const std::set<TypePtr> valid_types = {kInt64, kInt32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("indices", infer_type_indices);
  (void)types.emplace("output_shape", infer_type_output_shape);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  const std::set<TypePtr> valid_types_value = {kInt64, kInt32,   kInt16,   kInt8,    kUInt16,
                                               kUInt8, kFloat16, kFloat32, kFloat64, kBool};
  std::map<std::string, TypePtr> types_value;
  (void)types_value.emplace("values", infer_type_values);
  (void)types_value.emplace("default_value", infer_type_default_value);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types_value, valid_types_value, prim_name);
  return infer_type_values;
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseToDenseV2, BaseOperator);

void SparseToDenseV2::set_validate_indices(const bool validate_indices) {
  (void)this->AddAttr(kValidateIndices, api::MakeValue(validate_indices));
}

bool SparseToDenseV2::get_validate_indices() const {
  auto value_ptr = GetAttr(kValidateIndices);
  return GetValue<bool>(value_ptr);
}

void SparseToDenseV2::Init(const bool validate_indices) { this->set_validate_indices(validate_indices); }

AbstractBasePtr SparseToDenseV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t input_num = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto infertype = SparseToDenseV2InferType(primitive, input_args);
  auto infershape = SparseToDenseV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SparseToDenseV2, prim::kPrimSparseToDenseV2, SparseToDenseV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
