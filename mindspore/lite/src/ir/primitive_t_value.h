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

#ifndef MINDSPORE_LITE_SRC_ANF_IMPORTER_PRIMITIVET_H_
#define MINDSPORE_LITE_SRC_ANF_IMPORTER_PRIMITIVET_H_

#include <vector>
#include "ir/value.h"
#include "mindspore/lite/schema/inner/model_generated.h"

namespace mindspore::lite {

class PrimitiveTValue : public Value {
 public:
  explicit PrimitiveTValue(schema::PrimitiveT *primt) : primitive(primt) {}
  // not responsible to free primitive, the one created the dynamic memory is responsible to free it.
  ~PrimitiveTValue() override = default;

  MS_DECLARE_PARENT(PrimitiveTValue, Value)

  schema::PrimitiveT *GetPrimitiveT() const { return this->primitive; }

  void SetPrimitiveT(schema::PrimitiveT *primIn) { this->primitive = primIn; }

  bool operator==(const Value &rhs) const override {
    if (rhs.isa<PrimitiveTValue>()) {
      auto other_prim = static_cast<const PrimitiveTValue &>(rhs);
      auto a = this->primitive->value.type;
      auto b = other_prim.primitive->value.type;
      return a == b;
    } else {
      return false;
    }
  }

  void SetInputQuantParam(std::vector<std::vector<schema::QuantParamT>> vec_quant_param) {
  }

  void AddInputQuantParam(std::vector<schema::QuantParamT> quant_param) {
    this->input_quant_param_.emplace_back(quant_param);
  }
  std::vector<std::vector<schema::QuantParamT>> GetInputQuantParams() const {
    return input_quant_param_;
  }

  void AddOutputQuantParam(std::vector<schema::QuantParamT> quant_param) {
    this->output_quant_param_.emplace_back(quant_param);
  }
  std::vector<std::vector<schema::QuantParamT>> GetOutputQuantParams() const {
    return output_quant_param_;
  }

  void SetQuantType(schema::QuantType quant_type) { this->quant_type_ = quant_type; }

  schema::QuantType GetQuantType() const { return quant_type_; }

 protected:
  schema::PrimitiveT *primitive = nullptr;
  std::vector<std::vector<schema::QuantParamT>> input_quant_param_;
  std::vector<std::vector<schema::QuantParamT>> output_quant_param_;
  schema::QuantType quant_type_{schema::QuantType_QUANT_NONE};
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_ANF_IMPORTER_PRIMITIVET_H_

