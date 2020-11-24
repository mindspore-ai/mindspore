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

#include "tools/converter/parser/tflite/tflite_fullyconnected_parser.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {
PrimitiveC *TfliteFullyConnectedParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                           const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is null";
    return nullptr;
  }

  std::unique_ptr<schema::FullConnectionT> attr = std::make_unique<schema::FullConnectionT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsFullyConnectedOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op fully connect attr failed";
    return nullptr;
  }

  bool hasBias = tflite_op->inputs.size() > 2 && tflite_op->inputs[2] != -1;

  attr->hasBias = hasBias;
  attr->axis = 1;
  attr->useAxis = false;
  attr->activationType = GetActivationFunctionType(tflite_attr->fused_activation_function);

  primitive->value.type = schema::PrimitiveType_FullConnection;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteFullyConnectedParser(tflite::BuiltinOperator_FULLY_CONNECTED,
                                                new TfliteFullyConnectedParser());
TfliteNodeRegister g_tfliteFakeQuantParser(tflite::BuiltinOperator_FAKE_QUANT, new TfliteFullyConnectedParser());
}  // namespace lite
}  // namespace mindspore
