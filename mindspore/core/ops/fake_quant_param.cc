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

#include "ops/fake_quant_param.h"
#include <vector>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindapi/ir/type.h"
#include "mindapi/base/type_id.h"
#include "ir/scalar.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(FakeQuantParam, BaseOperator);

void FakeQuantParam::Init(const QuantDataType &quant_dtype, const std::string &quant_algo_name,
                          const bool is_perchannel) {
  this->set_quant_dtype(quant_dtype);
  this->set_is_perchannel(is_perchannel);
  this->set_quant_algo_name(quant_algo_name);
}

void FakeQuantParam::set_quant_dtype(const QuantDataType &quant_dtype) {
  (void)AddAttr(kAttrKeyQuantDType, api::MakeValue<int>(static_cast<int>(quant_dtype)));
}

QuantDataType FakeQuantParam::get_quant_dtype() const {
  auto dtype_ptr = this->GetAttr(kAttrKeyQuantDType);
  MS_EXCEPTION_IF_NULL(dtype_ptr);
  auto type_id = QuantDataType(GetValue<int>(dtype_ptr));
  return type_id;
}

void FakeQuantParam::set_quant_algo_name(const std::string &quant_algo_name) {
  (void)AddAttr(kAttrKeyQuantAlgoName, api::MakeValue<std::string>(quant_algo_name));
}

std::string FakeQuantParam::get_quant_algo_name() const {
  auto value_ptr = this->GetAttr(kAttrKeyQuantAlgoName);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::string>(value_ptr);
}

void FakeQuantParam::set_is_perchannel(const bool is_perchannel) {
  (void)AddAttr(kAttrKeyQuantParamIsPerChannel, api::MakeValue<bool>(is_perchannel));
}

bool FakeQuantParam::get_is_perchannel() const {
  auto value_ptr = this->GetAttr(kAttrKeyQuantParamIsPerChannel);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

void FakeQuantParam::set_quant_param(const std::string &key, api::ValuePtr param, size_t channel_index) {
  if (this->get_is_perchannel()) {
    auto value_ptr = this->GetAttr(key);
    std::vector<api::ValuePtr> params;
    if (value_ptr != nullptr) {
      params = GetValue<std::vector<api::ValuePtr>>(value_ptr);
    }
    if (channel_index == params.size()) {
      params.emplace_back(param);
    } else if (channel_index < params.size()) {
      params[channel_index] = param;
    } else {
      MS_LOG(EXCEPTION) << "Please set quant parameter in ascending order of channels.";
    }
    (void)AddAttr(key, api::MakeValue(params));
  } else {
    if (channel_index != 0) {
      MS_LOG(EXCEPTION) << "'channel_index' should be equal to zero while set a per-layer quant parameter, but got "
                        << channel_index << ".";
    }
    std::vector<api::ValuePtr> params{param};
    (void)AddAttr(key, api::MakeValue(params));
  }
}

api::ValuePtr FakeQuantParam::get_quant_param(const std::string &key, size_t channel_index) const {
  auto value_ptr = this->GetAttr(key);
  if (value_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Quant parameter " << key << " not found!";
  }
  auto params = GetValue<std::vector<api::ValuePtr>>(value_ptr);
  if (channel_index >= params.size()) {
    MS_LOG(EXCEPTION) << "Channel index(" << channel_index << ") out of range of quant parameter size(" << params.size()
                      << ")!";
  }
  return params[channel_index];
}

void FakeQuantParam::set_scale(double scale, size_t channel_index) {
  this->set_quant_param(kAttrKeyLinearQuantParamScale, api::MakeValue(scale), channel_index);
}

double FakeQuantParam::get_scale(size_t channel_index) const {
  auto scale = this->get_quant_param(kAttrKeyLinearQuantParamScale, channel_index);
  MS_EXCEPTION_IF_NULL(scale);
  return GetValue<double>(scale);
}

void FakeQuantParam::set_zero_point(int64_t zero_point, size_t channel_index) {
  this->set_quant_param(kAttrKeyLinearQuantParamZeroPoint, api::MakeValue(zero_point), channel_index);
}

int64_t FakeQuantParam::get_zero_point(size_t channel_index) const {
  auto zp = this->get_quant_param(kAttrKeyLinearQuantParamZeroPoint, channel_index);
  MS_EXCEPTION_IF_NULL(zp);
  return GetValue<int64_t>(zp);
}

class FakeQuantParamInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_num = 1;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    auto x = input_args[kInputIndex0]->BuildShape();
    MS_EXCEPTION_IF_NULL(x);
    auto shape_element = x->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_element);
    return shape_element;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_num = 1;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    return input_args[kInputIndex0]->BuildType();
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FakeQuantParam, prim::kPrimFakeQuantParam, FakeQuantParamInfer, false);
}  // namespace ops
}  // namespace mindspore
