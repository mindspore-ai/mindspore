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
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TfliteReluParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                      const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                      const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::RELU);
  prim->set_min_val(0);
  prim->set_max_val(FLT_MAX);

  return prim->GetPrim();
}

PrimitiveCPtr TfliteRelu6Parser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                       const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                       const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::RELU6);
  prim->set_min_val(0);
  prim->set_max_val(kValueThreshold6);

  return prim->GetPrim();
}

PrimitiveCPtr TfliteLeakyReluParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                           const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                           const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::LEAKY_RELU);

  const auto &tflite_attr = tflite_op->builtin_options.AsLeakyReluOptions();
  MS_CHECK_TRUE_MSG(tflite_attr != nullptr, nullptr, "Get LeakyRelu attr failed.");
  prim->set_alpha(tflite_attr->alpha);

  return prim->GetPrim();
}

PrimitiveCPtr TflitePReLUParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                       const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                       const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::PReLUFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_channel_shared(true);

  return prim->GetPrim();
}

PrimitiveCPtr TfliteTanhParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                      const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                      const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::TANH);

  return prim->GetPrim();
}

PrimitiveCPtr TfliteHardSwishParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                           const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                           const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::HSWISH);

  return prim->GetPrim();
}

PrimitiveCPtr TfliteLogisticParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::SIGMOID);

  return prim->GetPrim();
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
