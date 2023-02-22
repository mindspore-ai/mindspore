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

#include "ops/grad/lstm_grad_data.h"

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
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void LSTMGradData::set_input_size(const int64_t input_size) {
  (void)CheckAndConvertUtils::CheckInteger(kInput_size, input_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kInput_size, api::MakeValue(input_size));
}
int64_t LSTMGradData::get_input_size() const { return GetValue<int64_t>(GetAttr(kInput_size)); }
void LSTMGradData::set_hidden_size(const int64_t hidden_size) {
  (void)CheckAndConvertUtils::CheckInteger(kHidden_size, hidden_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kHidden_size, api::MakeValue(hidden_size));
}
int64_t LSTMGradData::get_hidden_size() const { return GetValue<int64_t>(GetAttr(kHidden_size)); }
void LSTMGradData::set_num_layers(const int64_t num_layers) {
  (void)CheckAndConvertUtils::CheckInteger(kNumLayers, num_layers, kGreaterThan, 0, this->name());
  (void)AddAttr(kNumLayers, api::MakeValue(num_layers));
}
int64_t LSTMGradData::get_num_layers() const { return GetValue<int64_t>(GetAttr(kNumLayers)); }
void LSTMGradData::set_has_bias(const bool has_bias) { (void)AddAttr(kHasBias, api::MakeValue(has_bias)); }
bool LSTMGradData::get_has_bias() const {
  auto value_ptr = this->GetAttr(kHasBias);
  return GetValue<bool>(value_ptr);
}
void LSTMGradData::set_dropout(const float dropout) {
  CheckAndConvertUtils::CheckInRange<float>(kDropout, dropout, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)AddAttr(kDropout, api::MakeValue(dropout));
}
float LSTMGradData::get_dropout() const {
  auto value_ptr = this->GetAttr(kDropout);
  return GetValue<float>(value_ptr);
}
void LSTMGradData::set_bidirectional(const bool bidirectional) {
  (void)AddAttr(kBidirectional, api::MakeValue(bidirectional));
}
bool LSTMGradData::get_bidirectional() const {
  auto value_ptr = this->GetAttr(kBidirectional);
  return GetValue<bool>(value_ptr);
}
void LSTMGradData::set_num_directions(const int64_t num_directions) {
  (void)AddAttr(kNumDirections, api::MakeValue(num_directions));
}
int64_t LSTMGradData::get_num_directions() const { return GetValue<int64_t>(GetAttr(kNumDirections)); }
void LSTMGradData::set_zoneout_cell(float zoneout_cell) { (void)AddAttr(kZoneoutCell, api::MakeValue(zoneout_cell)); }

float LSTMGradData::get_zoneout_cell() const { return GetValue<float>(this->GetAttr(kZoneoutCell)); }

void LSTMGradData::set_zoneout_hidden(float zoneout_hidden) {
  (void)AddAttr(kZoneoutHidden, api::MakeValue(zoneout_hidden));
}

float LSTMGradData::get_zoneout_hidden() const { return GetValue<float>(this->GetAttr(kZoneoutHidden)); }

void LSTMGradData::Init(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers,
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
const size_t kLstmOutputNum = 3;
abstract::TupleShapePtr LstmGradDataInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const size_t input_num = 9;
  auto shape_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  auto unknown_shapes =
    std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>(SizeToLong(kLstmOutputNum), shape_ptr));
  for (size_t i = 0; i < input_num; i++) {
    auto shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[i]->BuildShape())[kShape];
    if (IsDynamicRank(shape)) {
      return unknown_shapes;
    }
  }

  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto dy_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto dhy_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto dcy_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  bool dy_is_dynamic = IsDynamic(dy_shape);
  bool dhy_is_dynamic = IsDynamic(dhy_shape);
  bool dcy_is_dynamic = IsDynamic(dcy_shape);

  int64_t input_size = GetValue<int64_t>(primitive->GetAttr(kInput_size));
  int64_t hidden_size = GetValue<int64_t>(primitive->GetAttr(kHidden_size));
  int64_t num_layers = GetValue<int64_t>(primitive->GetAttr(kNumLayers));
  bool bidirectional = GetValue<bool>(primitive->GetAttr(kBidirectional));
  int64_t bidirection_num = 2;
  int64_t num_directions = bidirectional ? bidirection_num : 1;

  const size_t shape_size = 3;
  if (!dhy_is_dynamic) {
    (void)CheckAndConvertUtils::CheckInteger("dhy_shape.size()", SizeToLong(dhy_shape.size()), kEqual, shape_size,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("h_shape[0]", dhy_shape[0], kEqual, num_layers * num_directions,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("h_shape[2]", dhy_shape[kDim2], kEqual, hidden_size, prim_name);
    if (!dcy_is_dynamic) {
      (void)CheckAndConvertUtils::Check("dhy_shape", dhy_shape, kEqual, dcy_shape, prim_name);
    }
  }
  if (!dy_is_dynamic) {
    (void)CheckAndConvertUtils::CheckInteger("dy_shape.size()", SizeToLong(dy_shape.size()), kEqual, shape_size,
                                             prim_name);
    if (!dhy_is_dynamic) {
      (void)CheckAndConvertUtils::CheckInteger("dy[1]", dy_shape[kDim1], kEqual, dhy_shape[kDim1], prim_name);
    }
    (void)CheckAndConvertUtils::CheckInteger("dy[2]", dy_shape[kDim2], kEqual, hidden_size * num_directions, prim_name);
  }

  std::vector<int64_t> dx_shape = {y_shape[0], y_shape[kDim1], input_size};
  std::vector<abstract::BaseShapePtr> output_shapes;
  output_shapes.push_back(std::make_shared<abstract::Shape>(dx_shape));
  output_shapes.push_back(std::make_shared<abstract::Shape>(dhy_shape));
  output_shapes.push_back(std::make_shared<abstract::Shape>(dcy_shape));
  return std::make_shared<abstract::TupleShape>(output_shapes);
}

TuplePtr LstmGradDataInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto x_dtype = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, prim->name());

  std::vector<TypePtr> type_tuple(kLstmOutputNum, x_dtype);
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

AbstractBasePtr LstmGradDataInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 9;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = LstmGradDataInferType(primitive, input_args);
  auto infer_shape = LstmGradDataInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(LSTMGradData, BaseOperator);

// AG means auto generated
class MIND_API AGLstmGradDataInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LstmGradDataInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LstmGradDataInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LstmGradDataInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LSTMGradData, prim::kPrimLstmGradData, AGLstmGradDataInfer, false);
}  // namespace ops
}  // namespace mindspore
