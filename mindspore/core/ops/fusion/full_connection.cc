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

#include "ops/fusion/full_connection.h"
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void FullConnection::set_has_bias(const bool has_bias) { (void)this->AddAttr(kHasBias, MakeValue(has_bias)); }

bool FullConnection::get_has_bias() const {
  auto value_ptr = GetAttr(kHasBias);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

void FullConnection::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, MakeValue(axis)); }
int64_t FullConnection::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

void FullConnection::set_use_axis(const bool use_axis) { (void)this->AddAttr(kUseAxis, MakeValue(use_axis)); }
bool FullConnection::get_use_axis() const {
  auto value_ptr = GetAttr(kUseAxis);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

void FullConnection::set_activation_type(const ActivationType &activation_type) {
  int64_t swi = activation_type;
  (void)this->AddAttr(kActivationType, MakeValue(swi));
}
ActivationType FullConnection::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ActivationType(GetValue<int64_t>(value_ptr));
}
void FullConnection::Init(const bool has_bias, const int64_t axis, const bool use_axis,
                          const ActivationType &activation_type) {
  this->set_has_bias(has_bias);
  this->set_axis(axis);
  this->set_use_axis(use_axis);
  this->set_activation_type(activation_type);
}
AbstractBasePtr FullConnectionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto input0 = input_args[0];
  auto input1 = input_args[1];
  auto input0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input0->BuildShape())[kShape];
  auto input1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input1->BuildShape())[kShape];
  auto prim_axis = GetValue<int64_t>(primitive->GetAttr(kAxis));
  auto has_bias = GetValue<bool>(primitive->GetAttr(kHasBias));
  const int64_t input_num_bias = 3;
  const int64_t input_num = 2;
  if (has_bias) {
    (void)CheckAndConvertUtils::CheckInteger("input_args.size()", SizeToLong(input_args.size()), kEqual, input_num_bias,
                                             prim_name);
  } else {
    (void)CheckAndConvertUtils::CheckInteger("input_args.size()", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
  }
  auto use_axis = GetValue<bool>(primitive->GetAttr(kUseAxis));
  if (use_axis && (prim_axis < 1 || prim_axis > (int64_t)input0_shape.size())) {
    MS_EXCEPTION(ValueError) << "Full Connection axis is invalid";
  }
  int64_t new_k = 1;
  if (use_axis) {
    for (size_t t = LongToSize(prim_axis); t < input0_shape.size(); t++) {
      new_k *= input0_shape[t];
    }
    if (new_k != input1_shape[1]) {
      MS_EXCEPTION(ValueError) << "Input1 size is invalid";
    }
  } else {
    new_k = input1_shape[1];
  }
  if (has_bias) {
    auto input2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
    if (input2_shape[0] != input1_shape[0]) {
      MS_EXCEPTION(ValueError) << "Bias size is invalid";
    }
  }
  std::vector<int64_t> out_shape = {(int64_t)input0_shape.size()};
  if (use_axis) {
    out_shape.resize(LongToSize(prim_axis) + 1);
    out_shape[LongToSize(prim_axis)] = input1_shape[0];
  } else {
    int64_t total = 1;
    for (size_t i = 0; i < input0_shape.size(); i++) {
      total *= input0_shape[i];
    }
    out_shape.resize(2);
    auto batch_size = total / new_k;
    out_shape[0] = batch_size;
    out_shape[1] = input1_shape[0];
  }
  auto input0_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  return std::make_shared<abstract::AbstractTensor>(input0_type, out_shape);
}
REGISTER_PRIMITIVE_C(kNameFullConnection, FullConnection);
}  // namespace ops
}  // namespace mindspore
