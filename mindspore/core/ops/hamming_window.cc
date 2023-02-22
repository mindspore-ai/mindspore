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

#include <memory>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
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
  CheckAndConvertUtils::CheckInteger("length dim", length_size, kEqual, length_dim, primitive->name());
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
    CheckAndConvertUtils::CheckInteger("length value", length_value, kGreaterEqual, 0, primitive->name());
    out_shape.push_back(length_value);
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    std::vector<int64_t> out_shape = {abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(out_shape);
  }
}

TypePtr HammingWindowInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  const std::set<TypePtr> valid_input_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
  CheckAndConvertUtils::CheckTensorTypeValid("length", input_type, valid_input_types, primitive->name());
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

void HammingWindow::set_periodic(const bool periodic) { (void)this->AddAttr(kPeriodic, api::MakeValue(periodic)); }
bool HammingWindow::get_periodic() const { return GetValue<bool>(GetAttr(kPeriodic)); }
void HammingWindow::set_alpha(const float alpha) { (void)this->AddAttr(kAlpha, api::MakeValue(alpha)); }
float HammingWindow::get_alpha() const { return GetValue<float>(GetAttr(kAlpha)); }
void HammingWindow::set_beta(const float beta) { (void)this->AddAttr(kBeta, api::MakeValue(beta)); }
float HammingWindow::get_beta() const { return GetValue<float>(GetAttr(kBeta)); }

void HammingWindow::Init(const bool periodic, const float alpha, const float beta) {
  set_periodic(periodic);
  set_alpha(alpha);
  set_beta(beta);
}

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

// AG means auto generated
class MIND_API AGHammingWindowInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return HammingWindowInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return HammingWindowInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return HammingWindowInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(HammingWindow, prim::kPrimHammingWindow, AGHammingWindowInfer, false);
}  // namespace ops
}  // namespace mindspore
