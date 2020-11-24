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

#include "tools/converter/parser/tflite/tflite_activation_parser.h"
#include <memory>
#include <vector>
#include <string>
#include "src/ops/activation.h"
#include "src/ops/primitive_c.h"
#include "tools/converter/parser/tflite/tflite_util.h"

namespace mindspore::lite {
lite::PrimitiveC *TfliteActivationParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                             const std::unique_ptr<tflite::ModelT> &tflite_model) {
  std::unique_ptr<schema::ActivationT> attr = std::make_unique<schema::ActivationT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  auto tflite_op_type = (tflite_model->operator_codes[tflite_op->opcode_index])->builtin_code;
  auto ms_op_type = GetMSOpType(tflite_op_type);
  if (kActivationTypeMap.find(ms_op_type) == kActivationTypeMap.end()) {
    MS_LOG(ERROR) << ms_op_type << "is a not supported activation type";
    return nullptr;
  }
  attr->type = kActivationTypeMap.find(GetMSOpType(tflite_op_type))->second;
  if (attr->type == schema::ActivationType_LEAKY_RELU) {
    const auto &tflite_attr = tflite_op->builtin_options.AsLeakyReluOptions();
    if (tflite_attr == nullptr) {
      MS_LOG(ERROR) << "get op: " << GetMSOpType(tflite_op_type) << " attr failed";
      return nullptr;
    }
    attr->alpha = tflite_attr->alpha;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Activation;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_TfliteReluParser(tflite::BuiltinOperator_RELU, new TfliteActivationParser());
TfliteNodeRegister g_TfliteRelu6Parser(tflite::BuiltinOperator_RELU6, new TfliteActivationParser());
TfliteNodeRegister g_TfliteTanhParser(tflite::BuiltinOperator_TANH, new TfliteActivationParser());
TfliteNodeRegister g_TfliteSwishParser(tflite::BuiltinOperator_HARD_SWISH, new TfliteActivationParser());
TfliteNodeRegister g_tfliteLogisticParser(tflite::BuiltinOperator_LOGISTIC, new TfliteActivationParser());
TfliteNodeRegister g_TfliteLeakyReluParser(tflite::BuiltinOperator_LEAKY_RELU, new TfliteActivationParser());
}  // namespace mindspore::lite
