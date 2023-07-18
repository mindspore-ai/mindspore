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

#include "ops/lstm.h"
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr LSTMInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto h_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto c_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  if (IsDynamicRank(x_input_shape) || IsDynamicRank(h_input_shape) || IsDynamicRank(c_input_shape) ||
      IsDynamicRank(weight_shape)) {
    abstract::ShapePtr output = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{output, output, output, output, output});
  }

  const int64_t shape_size = 3;
  (void)CheckAndConvertUtils::CheckInteger("x_shape.size()", SizeToLong(x_input_shape.size()), kEqual, shape_size,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("h_shape.size()", SizeToLong(h_input_shape.size()), kEqual, shape_size,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("c_shape.size()", SizeToLong(c_input_shape.size()), kEqual, shape_size,
                                           prim_name);

  int64_t input_x_size = GetValue<int64_t>(primitive->GetAttr(kInput_size));
  int64_t num_layers = GetValue<int64_t>(primitive->GetAttr(kNumLayers));
  bool bidirectional = GetValue<bool>(primitive->GetAttr(kBidirectional));
  int64_t num_directions = 1;
  if (bidirectional) {
    num_directions = 2;
  }
  int64_t hidden_size = GetValue<int64_t>(primitive->GetAttr(kHidden_size));
  int64_t proj_size = GetValue<int64_t>(primitive->GetAttr(kProjection_size));
  int64_t real_hidden_size = proj_size > 0 ? proj_size : hidden_size;

  bool x_is_dynamic = IsDynamic(x_input_shape);
  bool h_is_dynamic = IsDynamic(h_input_shape);
  bool c_is_dynamic = IsDynamic(c_input_shape);
  if (!x_is_dynamic) {
    (void)CheckAndConvertUtils::CheckInteger("x_shape[2]", x_input_shape[kIndex2], kEqual, input_x_size, prim_name);
  }
  if (!h_is_dynamic && !c_is_dynamic) {
    if (proj_size == 0) {
      (void)CheckAndConvertUtils::Check("h_shape", h_input_shape, kEqual, c_input_shape, prim_name);
    }
    (void)CheckAndConvertUtils::CheckInteger("h_shape[0]", h_input_shape[0], kEqual, num_layers * num_directions,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("h_shape[2]", h_input_shape[kIndex2], kEqual, real_hidden_size, prim_name);
  }
  if (!x_is_dynamic && !h_is_dynamic) {
    (void)CheckAndConvertUtils::CheckInteger("h_shape[1]", h_input_shape[1], kEqual, x_input_shape[1], prim_name);
  }

  std::vector<int64_t> y_shape = {x_input_shape[0], x_input_shape[1], real_hidden_size * num_directions};
  std::vector<int64_t> h_shape = {h_input_shape};
  std::vector<int64_t> c_shape = {c_input_shape};
  std::vector<int64_t> reverse_shape = {1, 1};
  std::vector<int64_t> state_shape = {1, 1};
  auto output0 = std::make_shared<abstract::Shape>(y_shape);
  auto output1 = std::make_shared<abstract::Shape>(h_shape);
  auto output2 = std::make_shared<abstract::Shape>(c_shape);
  auto output3 = std::make_shared<abstract::Shape>(reverse_shape);
  auto output4 = std::make_shared<abstract::Shape>(state_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{output0, output1, output2, output3, output4});
}

TuplePtr LSTMInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const int64_t output_num = 5;
  auto type = input_args[kInputIndex0]->BuildType()->cast<TensorTypePtr>()->element();
  return std::make_shared<Tuple>(std::vector<TypePtr>(output_num, type));
}
}  // namespace

AbstractBasePtr LSTMInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 4;
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto infer_type = LSTMInferType(primitive, input_args);
  auto infer_shape = LSTMInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

void LSTM::set_input_size(const int64_t input_size) {
  (void)CheckAndConvertUtils::CheckInteger(kInput_size, input_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kInput_size, api::MakeValue(input_size));
}

int64_t LSTM::get_input_size() const { return GetValue<int64_t>(GetAttr(kInput_size)); }

void LSTM::set_hidden_size(const int64_t hidden_size) {
  (void)CheckAndConvertUtils::CheckInteger(kHidden_size, hidden_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kHidden_size, api::MakeValue(hidden_size));
}

int64_t LSTM::get_hidden_size() const { return GetValue<int64_t>(GetAttr(kHidden_size)); }

void LSTM::set_proj_size(const int64_t proj_size) {
  (void)CheckAndConvertUtils::CheckInteger(kProjection_size, proj_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kProjection_size, api::MakeValue(proj_size));
}

int64_t LSTM::get_proj_size() const { return GetValue<int64_t>(GetAttr(kProjection_size)); }

void LSTM::set_num_layers(const int64_t num_layers) {
  (void)CheckAndConvertUtils::CheckInteger(kNumLayers, num_layers, kGreaterThan, 0, this->name());
  (void)AddAttr(kNumLayers, api::MakeValue(num_layers));
}

int64_t LSTM::get_num_layers() const { return GetValue<int64_t>(GetAttr(kNumLayers)); }

void LSTM::set_has_bias(const bool has_bias) { (void)AddAttr(kHasBias, api::MakeValue(has_bias)); }

bool LSTM::get_has_bias() const {
  auto value_ptr = this->GetAttr(kHasBias);
  return GetValue<bool>(value_ptr);
}

void LSTM::set_dropout(const float dropout) {
  CheckAndConvertUtils::CheckInRange<float>(kDropout, dropout, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)AddAttr(kDropout, api::MakeValue(dropout));
}

float LSTM::get_dropout() const {
  auto value_ptr = this->GetAttr(kDropout);
  return GetValue<float>(value_ptr);
}

void LSTM::set_bidirectional(const bool bidirectional) { (void)AddAttr(kBidirectional, api::MakeValue(bidirectional)); }

bool LSTM::get_bidirectional() const {
  auto value_ptr = this->GetAttr(kBidirectional);
  return GetValue<bool>(value_ptr);
}

void LSTM::set_num_directions(const int64_t num_directions) {
  (void)AddAttr(kNumDirections, api::MakeValue(num_directions));
}

int64_t LSTM::get_num_directions() const { return GetValue<int64_t>(GetAttr(kNumDirections)); }

void LSTM::set_zoneout_cell(float zoneout_cell) { (void)AddAttr(kZoneoutCell, api::MakeValue(zoneout_cell)); }

float LSTM::get_zoneout_cell() const { return GetValue<float>(this->GetAttr(kZoneoutCell)); }

void LSTM::set_zoneout_hidden(float zoneout_hidden) { (void)AddAttr(kZoneoutHidden, api::MakeValue(zoneout_hidden)); }

float LSTM::get_zoneout_hidden() const { return GetValue<float>(this->GetAttr(kZoneoutHidden)); }

void LSTM::Init(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const bool has_bias,
                const float dropout, const bool bidirectional, const float zoneout_cell, const float zoneout_hidden) {
  this->set_input_size(input_size);
  this->set_hidden_size(hidden_size);
  this->set_num_layers(num_layers);
  this->set_has_bias(has_bias);
  this->set_dropout(dropout);
  this->set_bidirectional(bidirectional);
  if (bidirectional) {
    this->set_num_directions(2);
  } else {
    this->set_num_directions(1);
  }
  this->set_zoneout_cell(zoneout_cell);
  this->set_zoneout_hidden(zoneout_hidden);
}

MIND_API_OPERATOR_IMPL(LSTM, BaseOperator);
class MIND_API AGLSTMInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LSTMInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LSTMInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LSTMInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LSTM, prim::kPrimLstm, AGLSTMInfer, false);
}  // namespace ops
}  // namespace mindspore
