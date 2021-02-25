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

#include "ops/fusion/full_connection.h"
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void FullConnection::set_has_bias(const bool has_bias) { this->AddAttr(kHasBias, MakeValue(has_bias)); }
bool FullConnection::get_has_bias() const {
  auto value_ptr = GetAttr(kHasBias);
  return GetValue<bool>(value_ptr);
}

void FullConnection::set_axis(const int64_t axis) { this->AddAttr(kAxis, MakeValue(axis)); }
int64_t FullConnection::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

void FullConnection::set_use_axis(const bool use_axis) { this->AddAttr(kUseAxis, MakeValue(use_axis)); }
bool FullConnection::get_use_axis() const {
  auto value_ptr = GetAttr(kUseAxis);
  return GetValue<bool>(value_ptr);
}

void FullConnection::set_activation_type(const ActivationType &activation_type) {
  int64_t swi;
  swi = activation_type;
  this->AddAttr(kActivationType, MakeValue(swi));
}
ActivationType FullConnection::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
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
  auto full_prim = primitive->cast<PrimFullConnectionPtr>();
  MS_EXCEPTION_IF_NULL(full_prim);
  auto prim_name = full_prim->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[1]);
  auto input0 = input_args[0];
  auto input1 = input_args[1];
  auto input0_shape = CheckAndConvertUtils::ConvertShapePtrToShape("input0_shape", input0->BuildShape(), prim_name);
  auto input1_shape = CheckAndConvertUtils::ConvertShapePtrToShape("input1_shape", input1->BuildShape(), prim_name);
  auto prim_axis = full_prim->get_axis();
  if (full_prim->get_has_bias()) {
    CheckAndConvertUtils::CheckInteger("input_args.size()", input_args.size(), kEqual, 3, prim_name);
  } else {
    CheckAndConvertUtils::CheckInteger("input_args.size()", input_args.size(), kEqual, 2, prim_name);
  }
  if (full_prim->get_use_axis() && (prim_axis < 1 || prim_axis > (int64_t)input0_shape.size())) {
    MS_EXCEPTION(ValueError) << "Full Connection axis invalid";
  }
  int64_t new_k = 1;
  if (full_prim->get_use_axis()) {
    for (size_t t = prim_axis; t < input0_shape.size(); t++) {
      new_k *= input0_shape[t];
    }
    if (new_k != input1_shape[1]) {
      MS_EXCEPTION(ValueError) << "Input1 size invalid";
    }
  } else {
    new_k = input1_shape[1];
  }
  if (full_prim->get_has_bias()) {
    auto input2_shape =
      CheckAndConvertUtils::ConvertShapePtrToShape("input2_shape", input_args[2]->BuildShape(), prim_name);
    if (input2_shape[0] != input1_shape[0]) {
      MS_EXCEPTION(ValueError) << "Bias size invalid";
    }
  }
  std::vector<int64_t> out_shape = {(int64_t)input0_shape.size()};
  if (full_prim->get_use_axis()) {
    out_shape.resize(prim_axis + 1);
    out_shape[prim_axis] = input1_shape[0];
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
