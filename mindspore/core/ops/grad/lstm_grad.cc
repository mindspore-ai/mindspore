/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/grad/lstm_grad.h"

#include <memory>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
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
void LSTMGrad::set_input_size(const int64_t input_size) {
  (void)CheckAndConvertUtils::CheckInteger(kInput_size, input_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kInput_size, api::MakeValue(input_size));
}
int64_t LSTMGrad::get_input_size() const { return GetValue<int64_t>(GetAttr(kInput_size)); }
void LSTMGrad::set_hidden_size(const int64_t hidden_size) {
  (void)CheckAndConvertUtils::CheckInteger(kHidden_size, hidden_size, kGreaterThan, 0, this->name());
  (void)AddAttr(kHidden_size, api::MakeValue(hidden_size));
}
int64_t LSTMGrad::get_hidden_size() const { return GetValue<int64_t>(GetAttr(kHidden_size)); }
void LSTMGrad::set_num_layers(const int64_t num_layers) {
  (void)CheckAndConvertUtils::CheckInteger(kNumLayers, num_layers, kGreaterThan, 0, this->name());
  (void)AddAttr(kNumLayers, api::MakeValue(num_layers));
}
int64_t LSTMGrad::get_num_layers() const { return GetValue<int64_t>(GetAttr(kNumLayers)); }
void LSTMGrad::set_has_bias(const bool has_bias) { (void)AddAttr(kHasBias, api::MakeValue(has_bias)); }
bool LSTMGrad::get_has_bias() const {
  auto value_ptr = this->GetAttr(kHasBias);
  return GetValue<bool>(value_ptr);
}
void LSTMGrad::set_dropout(const float dropout) {
  CheckAndConvertUtils::CheckInRange<float>(kDropout, dropout, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)AddAttr(kDropout, api::MakeValue(dropout));
}
float LSTMGrad::get_dropout() const {
  auto value_ptr = this->GetAttr(kDropout);
  return GetValue<float>(value_ptr);
}
void LSTMGrad::set_bidirectional(const bool bidirectional) {
  (void)AddAttr(kBidirectional, api::MakeValue(bidirectional));
}
bool LSTMGrad::get_bidirectional() const {
  auto value_ptr = this->GetAttr(kBidirectional);
  return GetValue<bool>(value_ptr);
}
void LSTMGrad::set_num_directions(const int64_t num_directions) {
  (void)AddAttr(kNumDirections, api::MakeValue(num_directions));
}
int64_t LSTMGrad::get_num_directions() const { return GetValue<int64_t>(GetAttr(kNumDirections)); }
void LSTMGrad::set_zoneout_cell(float zoneout_cell) { (void)AddAttr(kZoneoutCell, api::MakeValue(zoneout_cell)); }

float LSTMGrad::get_zoneout_cell() const { return GetValue<float>(this->GetAttr(kZoneoutCell)); }

void LSTMGrad::set_zoneout_hidden(float zoneout_hidden) {
  (void)AddAttr(kZoneoutHidden, api::MakeValue(zoneout_hidden));
}

float LSTMGrad::get_zoneout_hidden() const { return GetValue<float>(this->GetAttr(kZoneoutHidden)); }

void LSTMGrad::Init(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const bool has_bias,
                    const float dropout, const bool bidirectional, const float zoneout_cell,
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

class LstmGradInfer : public abstract::OpInferBase {
 public:
  LstmGradInfer() = default;

  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputNum,
                                             prim_name);
    auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
    auto dy_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex7]->BuildShape())[kShape];
    auto dhy_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex8]->BuildShape())[kShape];
    auto dcy_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex9]->BuildShape())[kShape];
    if (IsDynamicRank(y_shape) || IsDynamicRank(dy_shape) || IsDynamicRank(dhy_shape) || IsDynamicRank(dcy_shape)) {
      auto shape_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
      return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>(kNumber4, shape_ptr));
    }
    (void)CheckAndConvertUtils::CheckInteger("dhy_shape size", SizeToLong(dhy_shape.size()), kEqual, kShapeSize,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("dcy_shape size", SizeToLong(dcy_shape.size()), kEqual, kShapeSize,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("dy_shape size", SizeToLong(dy_shape.size()), kEqual, kShapeSize,
                                             prim_name);
    bool is_dy_shape_dcy = IsDynamic(dy_shape);
    bool is_dynamic_shape_dhy = IsDynamic(dhy_shape);
    bool is_dynamic_shape_dcy = IsDynamic(dcy_shape);

    int64_t num_layers = GetValue<int64_t>(primitive->GetAttr(kNumLayers));
    bool bidirectional = GetValue<bool>(primitive->GetAttr(kBidirectional));
    int64_t num_directions = kNumber1;
    if (bidirectional) {
      num_directions = kNumber2;
    }
    int64_t hidden_size = GetValue<int64_t>(primitive->GetAttr(kHidden_size));
    if (!is_dynamic_shape_dhy) {
      (void)CheckAndConvertUtils::CheckInteger("h_shape[0]", dhy_shape[kDim0], kEqual, num_layers * num_directions,
                                               prim_name);
      (void)CheckAndConvertUtils::CheckInteger("h_shape[2]", dhy_shape[kDim2], kEqual, hidden_size, prim_name);
      if (!is_dynamic_shape_dcy) {
        (void)CheckAndConvertUtils::Check("dhy and dcy shape", dhy_shape, kEqual, dcy_shape, prim_name);
      }
    }
    if (!is_dy_shape_dcy) {
      if (dhy_shape[kDim1] != abstract::Shape::kShapeDimAny) {
        (void)CheckAndConvertUtils::CheckInteger("dy_shape[1]", dy_shape[kDim1], kEqual, dhy_shape[kDim1], prim_name);
      }
      (void)CheckAndConvertUtils::CheckInteger("dy_shape[2]", dy_shape[kDim2], kEqual, hidden_size * num_directions,
                                               prim_name);
    }
    int64_t input_size = GetValue<int64_t>(primitive->GetAttr(kInput_size));
    auto weight_size = GetWeightSize(primitive, num_layers, num_directions);
    ShapeVector dx_shape = {y_shape[kDim0], y_shape[kDim1], input_size};
    ShapeVector weight_shape = {weight_size, kNumber1, kNumber1};
    std::vector<abstract::BaseShapePtr> output_shapes;
    output_shapes.push_back(std::make_shared<abstract::Shape>(dx_shape));
    output_shapes.push_back(std::make_shared<abstract::Shape>(dhy_shape));
    output_shapes.push_back(std::make_shared<abstract::Shape>(dcy_shape));
    output_shapes.push_back(std::make_shared<abstract::Shape>(weight_shape));
    return std::make_shared<abstract::TupleShape>(output_shapes);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto hx_type_ptr = input_args[kInputIndex1]->BuildType();
    auto dy_type_ptr = input_args[kInputIndex7]->BuildType();
    std::vector<TypePtr> types = {dy_type_ptr, dy_type_ptr, dy_type_ptr, hx_type_ptr};
    return std::make_shared<Tuple>(types);
  }

 private:
  int64_t GetWeightSize(const PrimitivePtr &primitive, int64_t num_layers, int64_t num_directions) const {
    int64_t weight_size = 0;
    bool has_bias = GetValue<bool>(primitive->GetAttr(kHasBias));
    int64_t input_size = GetValue<int64_t>(primitive->GetAttr(kInput_size));
    int64_t hidden_size = GetValue<int64_t>(primitive->GetAttr(kHidden_size));
    int64_t gate_size = hidden_size * kNumber4;
    for (int i = 0; i < num_layers; ++i) {
      for (int j = 0; j < num_directions; ++j) {
        int64_t input_layer_size = input_size;
        if (i != 0) {
          input_layer_size = hidden_size * num_directions;
        }
        weight_size += gate_size * input_layer_size;
        weight_size += gate_size * hidden_size;
        if (has_bias) {
          weight_size += gate_size;
        }
      }
    }
    return weight_size;
  }

  const int kInputNum = 11;
  const int64_t kNumber1 = 1;
  const int64_t kNumber2 = 2;
  const int64_t kNumber4 = 4;
  const size_t kShapeSize = 3;
};

MIND_API_OPERATOR_IMPL(LSTMGrad, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(LSTMGrad, prim::kPrimLstmGrad, LstmGradInfer, false);
}  // namespace ops
}  // namespace mindspore
