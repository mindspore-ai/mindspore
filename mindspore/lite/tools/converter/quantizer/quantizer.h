/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZER_H

#include <unordered_map>
#include <utility>
#include <memory>
#include "schema/inner/model_generated.h"
#include "include/errorcode.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "base/base.h"
#include "src/param_value_lite.h"
#include "tools/converter/converter_flags.h"
#include "tools/converter/quant_param_holder.h"

namespace mindspore::lite::quant {
using STATUS = int;
enum QuantType {
  QuantType_QUANT_NONE = 0,
  QuantType_AwareTraining = 1,
  QuantType_WeightQuant = 2,
  QuantType_PostTraining = 3,
  QuantType_MIN = QuantType_QUANT_NONE,
  QuantType_MAX = QuantType_PostTraining
};

class Quantizer {
 public:
  explicit Quantizer(FuncGraphPtr graph) : funcGraph(std::move(graph)) {}

  virtual ~Quantizer() = default;

  virtual STATUS RemoveFakeQuant();

  virtual STATUS GenerateQuantParam();

  virtual STATUS DetermineNodeQuantType();

  virtual STATUS DoQuantize(FuncGraphPtr func_graph) = 0;

  mindspore::lite::converter::Flags flags;

 protected:
  FuncGraphPtr funcGraph = nullptr;
};

class FbQuantizer {
 public:
  explicit FbQuantizer(schema::MetaGraphT *graph) : graph(graph) {}

  virtual ~FbQuantizer() = default;

  virtual STATUS RemoveFakeQuant();

  virtual STATUS GenerateQuantParam();

  virtual STATUS DetermineNodeQuantType();

  virtual STATUS DoQuantize() = 0;

 protected:
  schema::MetaGraphT *graph = nullptr;
};
}  // namespace mindspore::lite::quant
#endif
