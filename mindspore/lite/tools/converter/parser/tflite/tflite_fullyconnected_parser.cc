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
#include "ops/fusion/full_connection.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteFullyConnectedParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                   const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::FullConnection>();

  prim->set_axis(1);
  prim->set_use_axis(false);
  prim->set_has_bias(tflite_op->inputs.size() > 2 && tflite_op->inputs.at(2) != -1);

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsFullyConnectedOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get FullConnection attr failed";
    return nullptr;
  }
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  return prim.release();
}

TfliteNodeRegister g_tfliteFullyConnectedParser(tflite::BuiltinOperator_FULLY_CONNECTED,
                                                new TfliteFullyConnectedParser());
TfliteNodeRegister g_tfliteFakeQuantParser(tflite::BuiltinOperator_FAKE_QUANT, new TfliteFullyConnectedParser());
}  // namespace lite
}  // namespace mindspore
