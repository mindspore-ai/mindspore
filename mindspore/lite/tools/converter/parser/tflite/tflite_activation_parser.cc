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

namespace mindspore {
namespace lite {
STATUS TfliteActivationParser::Parse(TfliteTensorsInfo *tensors_info,
                                     const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::unique_ptr<tflite::ModelT> &tflite_model,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  MS_ASSERT(tflite_subgraph != nullptr);
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::ActivationT> attr = std::make_unique<schema::ActivationT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  std::vector<std::string> node_name_str;
  Split(op->name, &node_name_str, "-");
  const char *node_name = node_name_str.data()->c_str();
  if (std::strcmp(node_name, "Relu") == 0) {
    MS_LOG(DEBUG) << "parse TfliteReluParser";
    attr->type = schema::ActivationType_RELU;
  } else if (std::strcmp(node_name, "Relu6") == 0) {
    MS_LOG(DEBUG) << "parse TfliteRelu6Parser";
    attr->type = schema::ActivationType_RELU6;
  } else if (std::strcmp(node_name, "Tanh") == 0) {
    MS_LOG(DEBUG) << "parse TfliteTanhParser";
    attr->type = schema::ActivationType_TANH;
  } else if (std::strcmp(node_name, "Logistic") == 0) {
    MS_LOG(DEBUG) << "parse TfliteLogisticParser";
    attr->type = schema::ActivationType_SIGMOID;
  } else if (std::strcmp(node_name, "Swish") == 0) {
    MS_LOG(DEBUG) << "parse TfliteSwishParser";
    attr->type = schema::ActivationType_SWISH;
  } else if (std::strcmp(node_name, "HardSwish") == 0) {
    MS_LOG(DEBUG) << "parse TfliteHardSwishParser";
    attr->type = schema::ActivationType_HSWISH;
  } else if (std::strcmp(node_name, "LeakyRelu") == 0) {
    const auto &tflite_attr = tflite_op->builtin_options.AsLeakyReluOptions();
    if (tflite_attr == nullptr) {
      MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
      return RET_NULL_PTR;
    }
    attr->alpha = tflite_attr->alpha;
    attr->type = schema::ActivationType_LEAKY_RELU;
  } else {
    MS_LOG(ERROR) << node_name << " hasn't been supported";
    return RET_NOT_FIND_OP;
  }

  op->primitive->value.type = schema::PrimitiveType_Activation;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_info, tflite_op->inputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  AddOpOutput(op, tensors_info, tflite_op->outputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteReluParser("Relu", new TfliteActivationParser());
TfliteNodeRegister g_tfliteRelu6Parser("Relu6", new TfliteActivationParser());
TfliteNodeRegister g_tfliteTanhParser("Tanh", new TfliteActivationParser());
TfliteNodeRegister g_tfliteSwishParser("Swish", new TfliteActivationParser());
TfliteNodeRegister g_tfliteHardSwishParser("HardSwish", new TfliteActivationParser());
TfliteNodeRegister g_tfliteLogisticParser("Logistic", new TfliteActivationParser());
TfliteNodeRegister g_tfliteLeakyReluParser("LeakyRelu", new TfliteActivationParser());
}  // namespace lite
}  // namespace mindspore
