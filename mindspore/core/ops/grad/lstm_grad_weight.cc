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

#include "ops/grad/lstm_grad_weight.h"

#include <memory>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
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
void LSTMGradWeight::set_input_size(const int64_t input_size) {
  (void)CheckAndConvertUtils::CheckInteger(kInput_size, input_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kInput_size, api::MakeValue(input_size));
}
int64_t LSTMGradWeight::get_input_size() const { return GetValue<int64_t>(GetAttr(kInput_size)); }
void LSTMGradWeight::set_hidden_size(const int64_t hidden_size) {
  (void)CheckAndConvertUtils::CheckInteger(kHidden_size, hidden_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kHidden_size, api::MakeValue(hidden_size));
}
int64_t LSTMGradWeight::get_hidden_size() const { return GetValue<int64_t>(GetAttr(kHidden_size)); }
void LSTMGradWeight::set_num_layers(const int64_t num_layers) {
  (void)CheckAndConvertUtils::CheckInteger(kNumLayers, num_layers, kGreaterThan, 0, this->name());
  (void)AddAttr(kNumLayers, api::MakeValue(num_layers));
}
int64_t LSTMGradWeight::get_num_layers() const { return GetValue<int64_t>(GetAttr(kNumLayers)); }
void LSTMGradWeight::set_has_bias(const bool has_bias) { (void)AddAttr(kHasBias, api::MakeValue(has_bias)); }
bool LSTMGradWeight::get_has_bias() const {
  auto value_ptr = this->GetAttr(kHasBias);
  return GetValue<bool>(value_ptr);
}
void LSTMGradWeight::set_dropout(const float dropout) {
  CheckAndConvertUtils::CheckInRange<float>(kDropout, dropout, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)AddAttr(kDropout, api::MakeValue(dropout));
}
float LSTMGradWeight::get_dropout() const {
  auto value_ptr = this->GetAttr(kDropout);
  return GetValue<float>(value_ptr);
}
void LSTMGradWeight::set_bidirectional(const bool bidirectional) {
  (void)AddAttr(kBidirectional, api::MakeValue(bidirectional));
}
bool LSTMGradWeight::get_bidirectional() const {
  auto value_ptr = this->GetAttr(kBidirectional);
  return GetValue<bool>(value_ptr);
}
void LSTMGradWeight::set_num_directions(const int64_t num_directions) {
  (void)AddAttr(kNumDirections, api::MakeValue(num_directions));
}
int64_t LSTMGradWeight::get_num_directions() const { return GetValue<int64_t>(GetAttr(kNumDirections)); }
void LSTMGradWeight::set_zoneout_cell(float zoneout_cell) { (void)AddAttr(kZoneoutCell, api::MakeValue(zoneout_cell)); }

float LSTMGradWeight::get_zoneout_cell() const { return GetValue<float>(this->GetAttr(kZoneoutCell)); }

void LSTMGradWeight::set_zoneout_hidden(float zoneout_hidden) {
  (void)AddAttr(kZoneoutHidden, api::MakeValue(zoneout_hidden));
}

float LSTMGradWeight::get_zoneout_hidden() const { return GetValue<float>(this->GetAttr(kZoneoutHidden)); }

void LSTMGradWeight::Init(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers,
                          const bool has_bias, const float dropout, const bool bidirectional, const float zoneout_cell,
                          const float zoneout_hidden) {
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
  this->set_zoneout_cell(zoneout_cell);
  this->set_zoneout_hidden(zoneout_hidden);
}

namespace {
abstract::ShapePtr LstmGradWeightInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  int64_t input_size = GetValue<int64_t>(primitive->GetAttr(kInput_size));
  int64_t hidden_size = GetValue<int64_t>(primitive->GetAttr(kHidden_size));
  int64_t num_layers = GetValue<int64_t>(primitive->GetAttr(kNumLayers));
  bool has_bias = GetValue<bool>(primitive->GetAttr(kHasBias));
  bool bidirectional = GetValue<bool>(primitive->GetAttr(kBidirectional));
  int64_t bidirection_num = 2;
  int64_t num_directions = bidirectional ? bidirection_num : 1;

  int64_t weight_size = 0;
  int64_t gate_size = 4 * hidden_size;
  int64_t bias_broad_size = 2;
  for (int64_t layer = 0; layer < num_layers; layer++) {
    for (int64_t i = 0; i < num_directions; i++) {
      int64_t input_layer_size = layer == 0 ? input_size : hidden_size * num_directions;
      weight_size += gate_size * input_layer_size;
      weight_size += gate_size * hidden_size;
      if (has_bias) {
        weight_size += bias_broad_size * gate_size;
      }
    }
  }
  std::vector<int64_t> out_shape = {weight_size, 1, 1};
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr LstmGradWeightInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto x_dtype = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, prim->name());
  return x_dtype;
}
}  // namespace

AbstractBasePtr LstmGradWeightInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = LstmGradWeightInferType(primitive, input_args);
  auto infer_shape = LstmGradWeightInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(LSTMGradWeight, BaseOperator);

// AG means auto generated
class MIND_API AGLstmGradWeightInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LstmGradWeightInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LstmGradWeightInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LstmGradWeightInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LSTMGradWeight, prim::kPrimLstmGradWeight, AGLstmGradWeightInfer, false);
}  // namespace ops
}  // namespace mindspore
