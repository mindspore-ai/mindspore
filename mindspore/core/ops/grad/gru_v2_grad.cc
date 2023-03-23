/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/grad/gru_v2_grad.h"
#include <algorithm>
#include <vector>
#include <cstdint>
#include "ops/op_utils.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void GRUV2Grad::set_input_size(const int64_t input_size) {
  (void)CheckAndConvertUtils::CheckInteger(kInput_size, input_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kInput_size, api::MakeValue(input_size));
}
int64_t GRUV2Grad::get_input_size() const { return GetValue<int64_t>(GetAttr(kInput_size)); }
void GRUV2Grad::set_hidden_size(const int64_t hidden_size) {
  (void)CheckAndConvertUtils::CheckInteger(kHidden_size, hidden_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kHidden_size, api::MakeValue(hidden_size));
}
int64_t GRUV2Grad::get_hidden_size() const { return GetValue<int64_t>(GetAttr(kHidden_size)); }
void GRUV2Grad::set_num_layers(const int64_t num_layers) {
  (void)CheckAndConvertUtils::CheckInteger(kNumLayers, num_layers, kGreaterThan, 0, this->name());
  (void)AddAttr(kNumLayers, api::MakeValue(num_layers));
}
int64_t GRUV2Grad::get_num_layers() const { return GetValue<int64_t>(GetAttr(kNumLayers)); }
void GRUV2Grad::set_has_bias(const bool has_bias) { (void)AddAttr(kHasBias, api::MakeValue(has_bias)); }
bool GRUV2Grad::get_has_bias() const {
  auto value_ptr = this->GetAttr(kHasBias);
  return GetValue<bool>(value_ptr);
}
void GRUV2Grad::set_dropout(const float dropout) {
  CheckAndConvertUtils::CheckInRange<float>(kDropout, dropout, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)AddAttr(kDropout, api::MakeValue(dropout));
}
float GRUV2Grad::get_dropout() const {
  auto value_ptr = this->GetAttr(kDropout);
  return GetValue<float>(value_ptr);
}
void GRUV2Grad::set_bidirectional(const bool bidirectional) {
  (void)AddAttr(kBidirectional, api::MakeValue(bidirectional));
}
bool GRUV2Grad::get_bidirectional() const {
  auto value_ptr = this->GetAttr(kBidirectional);
  return GetValue<bool>(value_ptr);
}
void GRUV2Grad::set_num_directions(const int64_t num_directions) {
  (void)AddAttr(kNumDirections, api::MakeValue(num_directions));
}
int64_t GRUV2Grad::get_num_directions() const { return GetValue<int64_t>(GetAttr(kNumDirections)); }

void GRUV2Grad::Init(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const bool has_bias,
                     const float dropout, const bool bidirectional) {
  this->set_input_size(input_size);
  this->set_hidden_size(hidden_size);
  this->set_num_layers(num_layers);
  this->set_has_bias(has_bias);
  this->set_dropout(dropout);
  this->set_bidirectional(bidirectional);
  if (bidirectional) {
    constexpr int k2Directions = 2;
    this->set_num_directions(k2Directions);
  } else {
    this->set_num_directions(1);
  }
}

class GruGradInfer : public abstract::OpInferBase {
  const int kInputNum = 9;
  const int64_t kNumber1 = 1;
  const int64_t kNumber2 = 2;
  const int64_t kNumber3 = 3;
  const size_t kShapeSize = 3;
  const int kIndex0 = 0;
  const int kIndex2 = 2;
  const int kHxIdx = 1;
  const int kYIdx = 4;
  const int kDyIdx = 6;
  const int kDhyIdx = 7;

 public:
  GruGradInfer() = default;

  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputNum,
                                             prim_name);
    auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kYIdx]->BuildShape())[kShape];
    auto dy_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDyIdx]->BuildShape())[kShape];
    auto dhy_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDhyIdx]->BuildShape())[kShape];
    (void)CheckAndConvertUtils::CheckInteger("dhy_shape size", SizeToLong(dhy_shape.size()), kEqual, kShapeSize,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("dy_shape size", SizeToLong(dy_shape.size()), kEqual, kShapeSize,
                                             prim_name);

    int64_t num_layers = GetValue<int64_t>(primitive->GetAttr(kNumLayers));
    bool bidirectional = GetValue<bool>(primitive->GetAttr(kBidirectional));
    int64_t num_directions = kNumber1;
    if (bidirectional) {
      num_directions = kNumber2;
    }
    int64_t input_size = GetValue<int64_t>(primitive->GetAttr(kInput_size));
    auto weight_size = GetWeightSize(primitive, num_layers, num_directions);
    ShapeVector dx_shape = {y_shape[kIndex0], y_shape[kIndex2], input_size};
    ShapeVector weight_shape = {weight_size, kNumber1, kNumber1};
    std::vector<abstract::BaseShapePtr> output_shapes;
    output_shapes.push_back(std::make_shared<abstract::Shape>(dx_shape));
    output_shapes.push_back(std::make_shared<abstract::Shape>(dhy_shape));
    output_shapes.push_back(std::make_shared<abstract::Shape>(weight_shape));
    return std::make_shared<abstract::TupleShape>(output_shapes);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto hx_type_ptr = input_args[kHxIdx]->BuildType();
    auto dy_type_ptr = input_args[kDyIdx]->BuildType();
    std::vector<TypePtr> types = {dy_type_ptr, dy_type_ptr, hx_type_ptr};
    return std::make_shared<Tuple>(types);
  }

 private:
  int64_t GetWeightSize(const PrimitivePtr &primitive, int64_t num_layers, int64_t num_directions) const {
    int64_t weight_size = 0;
    bool has_bias = GetValue<bool>(primitive->GetAttr(kHasBias));
    int64_t input_size = GetValue<int64_t>(primitive->GetAttr(kInput_size));
    int64_t hidden_size = GetValue<int64_t>(primitive->GetAttr(kHidden_size));
    int64_t gate_size = hidden_size * kNumber3;
    weight_size += input_size * gate_size * num_directions +
                   (num_layers - 1) * (hidden_size * num_directions) * gate_size * num_directions;
    int64_t temp = num_directions * num_layers;
    weight_size += gate_size * hidden_size * temp;
    if (has_bias) {
      weight_size += gate_size * temp;
    }
    return weight_size;
  }
};

MIND_API_OPERATOR_IMPL(GRUV2Grad, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(GRUV2Grad, prim::kPrimGRUV2Grad, GruGradInfer, false);
}  // namespace ops
}  // namespace mindspore
