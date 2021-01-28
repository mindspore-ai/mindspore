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
#include "tools/converter/parser/tflite/tflite_util.h"
#include "ops/fusion/prelu_fusion.h"
#include "ops/fusion/activation.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteReluParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Activation>();

  prim->set_activation_type(mindspore::ActivationType::RELU);

  return prim.release();
}

ops::PrimitiveC *TfliteRelu6Parser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Activation>();

  prim->set_activation_type(mindspore::ActivationType::RELU6);

  return prim.release();
}

ops::PrimitiveC *TfliteLeakyReluParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                              const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Activation>();

  prim->set_activation_type(mindspore::ActivationType::LEAKY_RELU);

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsLeakyReluOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get LeakyRelu attr failed";
    return nullptr;
  }
  prim->set_alpha(tflite_attr->alpha);

  return prim.release();
}

ops::PrimitiveC *TflitePReLUParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::PReLUFusion>();

  prim->set_channel_shared(true);

  return prim.release();
}

ops::PrimitiveC *TfliteTanhParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Activation>();

  prim->set_activation_type(mindspore::ActivationType::TANH);

  return prim.release();
}

ops::PrimitiveC *TfliteHardSwishParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                              const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Activation>();

  prim->set_activation_type(mindspore::ActivationType::HSWISH);

  return prim.release();
}

ops::PrimitiveC *TfliteLogisticParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                             const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Activation>();

  prim->set_activation_type(mindspore::ActivationType::SIGMOID);

  return prim.release();
}

TfliteNodeRegister g_TfliteReluParser(tflite::BuiltinOperator_RELU, new TfliteReluParser());
TfliteNodeRegister g_TfliteRelu6Parser(tflite::BuiltinOperator_RELU6, new TfliteRelu6Parser());
TfliteNodeRegister g_TflitePReLUParser(tflite::BuiltinOperator_PRELU, new TflitePReLUParser());
TfliteNodeRegister g_TfliteLeakyReluParser(tflite::BuiltinOperator_LEAKY_RELU, new TfliteLeakyReluParser());
TfliteNodeRegister g_TfliteTanhParser(tflite::BuiltinOperator_TANH, new TfliteTanhParser());
TfliteNodeRegister g_TfliteSwishParser(tflite::BuiltinOperator_HARD_SWISH, new TfliteHardSwishParser());
TfliteNodeRegister g_tfliteLogisticParser(tflite::BuiltinOperator_LOGISTIC, new TfliteLogisticParser());
}  // namespace lite
}  // namespace mindspore
