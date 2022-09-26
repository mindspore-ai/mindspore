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

#include "ops/hamming_window.h"

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t MAX_WINDOW_LEN = 1024 * 1024;

#define WINDOW_LENGTH_CASE(DTYPE, TYPE, LENGTH_VALUE, LENGTH_TENSOR)                    \
  case (DTYPE): {                                                                       \
    LENGTH_VALUE = static_cast<int64_t>(*static_cast<TYPE *>(LENGTH_TENSOR->data_c())); \
    break;                                                                              \
  }

abstract::ShapePtr HammingWindowInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto length_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto length_size = length_shape.size();
  const int64_t length_dim = 1;
  (void)CheckAndConvertUtils::CheckInteger("length dim", SizeToLong(length_size), kEqual, length_dim,
                                           primitive->name());
  if (input_args[0]->isa<abstract::AbstractTensor>() && !input_args[0]->BuildValue()->isa<AnyValue>() &&
      !input_args[0]->BuildValue()->isa<None>()) {
    auto length = input_args[0]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(length);
    auto length_value_ptr = length->BuildValue();
    MS_EXCEPTION_IF_NULL(length_value_ptr);
    auto length_tensor = length_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(length_tensor);
    auto input_type = input_args[0]->BuildType();
    MS_EXCEPTION_IF_NULL(input_type);
    auto input_type_id = input_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(input_type_id);
    auto input_type_element = input_type_id->element();
    MS_EXCEPTION_IF_NULL(input_type_element);
    auto input_type_value = input_type_element->type_id();
    std::vector<int64_t> out_shape;
    int64_t length_value = 0;
    switch (input_type_value) {
      WINDOW_LENGTH_CASE(kNumberTypeInt8, int8_t, length_value, length_tensor)
      WINDOW_LENGTH_CASE(kNumberTypeInt16, int16_t, length_value, length_tensor)
      WINDOW_LENGTH_CASE(kNumberTypeInt32, int32_t, length_value, length_tensor)
      WINDOW_LENGTH_CASE(kNumberTypeInt64, int64_t, length_value, length_tensor)
      WINDOW_LENGTH_CASE(kNumberTypeUInt8, uint8_t, length_value, length_tensor)
      WINDOW_LENGTH_CASE(kNumberTypeUInt16, uint16_t, length_value, length_tensor)
      WINDOW_LENGTH_CASE(kNumberTypeUInt32, uint32_t, length_value, length_tensor)
      WINDOW_LENGTH_CASE(kNumberTypeUInt64, uint64_t, length_value, length_tensor)
      default: {
        MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                                << "', the dtype of 'length' should be integer data type, but got "
                                << TypeIdLabel(input_type_value);
      }
    }
    (void)CheckAndConvertUtils::CheckInteger("length value", length_value, kGreaterEqual, 0, primitive->name());
    out_shape.push_back(length_value);
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    std::vector<int64_t> out_shape = {abstract::Shape::kShapeDimAny};
    std::vector<int64_t> infer_shape_min = {0};
    std::vector<int64_t> infer_shape_max = {MAX_WINDOW_LEN};
    return std::make_shared<abstract::Shape>(out_shape, infer_shape_min, infer_shape_max);
  }
}

TypePtr HammingWindowInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  const std::set<TypePtr> valid_input_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("length", input_type, valid_input_types, primitive->name());
  auto dtype_attr = primitive->GetAttr("dtype");
  MS_EXCEPTION_IF_NULL(dtype_attr);
  int64_t dtype_value = GetValue<int64_t>(dtype_attr);
  const int64_t kFloat16Enum = 1;
  const int64_t kFloat32Enum = 0;
  const int64_t kFloat64Enum = 11;
  switch (dtype_value) {
    case kFloat16Enum: {
      return kFloat16;
    }
    case kFloat32Enum: {
      return kFloat32;
    }
    case kFloat64Enum: {
      return kFloat64;
    }
    default:
      MS_EXCEPTION(TypeError) << "For HammingWindow, the dtype of HammingWindow is invalid!";
  }
}
}  // namespace

MIND_API_OPERATOR_IMPL(HammingWindow, BaseOperator);
AbstractBasePtr HammingWindowInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = HammingWindowInferType(primitive, input_args);
  auto infer_shape = HammingWindowInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(HammingWindow, prim::kPrimHammingWindow, HammingWindowInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
