/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CALC_QUANT_PARAM_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CALC_QUANT_PARAM_H

#include <unordered_map>
#include <memory>
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore::lite {
static constexpr int CONVLUTION_INPUT_NUM = 3;

class QuantParamCalcer {
 public:
  virtual ~QuantParamCalcer() = default;
  virtual int Calc(schema::MetaGraphT *graph, const schema::CNodeT &node);

 protected:
  STATUS ComputeConstQuantParam(const schema::TensorT &tensor, schema::QuantParamT *quantParam);

 protected:
  size_t inputParamDone = 0;
  size_t outputParamDone = 0;
};

class CommonCalcer : public QuantParamCalcer {
 public:
  CommonCalcer() = default;
  ~CommonCalcer() override = default;
  int Calc(schema::MetaGraphT *subGraph, const schema::CNodeT &node) override;
};

class ConvCalcer : public CommonCalcer {
 public:
  ConvCalcer() = default;
  ~ConvCalcer() override = default;
  int Calc(schema::MetaGraphT *subGraph, const schema::CNodeT &node) override;
};

class BiasAddCalcer : public CommonCalcer {
 public:
  BiasAddCalcer() = default;
  ~BiasAddCalcer() override = default;
  int Calc(schema::MetaGraphT *subGraph, const schema::CNodeT &node) override;
};

class LinearCalcer : public QuantParamCalcer {
 public:
  LinearCalcer() = default;
  ~LinearCalcer() override = default;
  int Calc(schema::MetaGraphT *graph, const schema::CNodeT &node) override;
};

class QuantParamCalcRegister {
 public:
  virtual ~QuantParamCalcRegister();
  std::shared_ptr<QuantParamCalcer> GetQuantParamCalcer(schema::PrimitiveType opType);
  static QuantParamCalcRegister *GetInstance();

 private:
  QuantParamCalcRegister();
  std::unordered_map<schema::PrimitiveType, std::shared_ptr<QuantParamCalcer>> _registerMap;
};
}  // namespace mindspore::lite
#endif
