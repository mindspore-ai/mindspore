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
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/type_id.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

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

void FakeQuantParam::set_quant_param(const std::string &key, api::ValuePtr param) { (void)AddAttr(key, param); }

api::ValuePtr FakeQuantParam::get_quant_param(const std::string &key) const {
  auto value_ptr = this->GetAttr(key);
  if (value_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Quant parameter " << key << " not found!";
  }
  return value_ptr;
}

void FakeQuantParam::set_scales(std::vector<float> scales) {
  (void)this->AddAttr(kAttrKeyLinearQuantParamScale, api::MakeValue(scales));
}

std::vector<float> FakeQuantParam::get_scales() const {
  auto value_ptr = GetAttr(kAttrKeyLinearQuantParamScale);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<float>>(value_ptr);
}

void FakeQuantParam::set_zero_points(std::vector<int64_t> zero_points) {
  (void)this->AddAttr(kAttrKeyLinearQuantParamZeroPoint, api::MakeValue(zero_points));
}

std::vector<int64_t> FakeQuantParam::get_zero_points() const {
  auto value_ptr = GetAttr(kAttrKeyLinearQuantParamZeroPoint);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
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
