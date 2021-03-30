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
#include "tools/converter/parser/tf/tf_activation_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/fusion/activation.h"
#include "ops/leaky_relu.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFActivationParser::Parse(const tensorflow::NodeDef &tf_op,
                                           const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                           std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Activation>();

  if (tf_op.op() == "Relu") {
    prim->set_activation_type(mindspore::ActivationType::RELU);
  } else if (tf_op.op() == "Relu6") {
    prim->set_activation_type(mindspore::ActivationType::RELU6);
  } else if (tf_op.op() == "Sigmoid") {
    prim->set_activation_type(mindspore::ActivationType::SIGMOID);
  } else if (tf_op.op() == "Tanh") {
    prim->set_activation_type(mindspore::ActivationType::TANH);
  } else if (tf_op.op() == "Selu") {
    prim->set_activation_type(mindspore::ActivationType::SELU);
  } else if (tf_op.op() == "Softplus") {
    prim->set_activation_type(mindspore::ActivationType::SOFTPLUS);
  } else if (tf_op.op() == "LeakyRelu") {
    prim->set_activation_type(mindspore::ActivationType::LEAKY_RELU);
    tensorflow::AttrValue attr_value;
    if (!TensorFlowUtils::FindAttrValue(tf_op, "alpha", &attr_value)) {
      MS_LOG(ERROR) << "The attribute alpha should be specified.";
      return nullptr;
    }
    prim->set_alpha(attr_value.f());
  } else {
    MS_LOG(ERROR) << "unsupported activation type:" << tf_op.op();
    return nullptr;
  }

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

TFNodeRegistrar g_tfReluParser("Relu", new TFActivationParser());
TFNodeRegistrar g_tfRelu6Parser("Relu6", new TFActivationParser());
TFNodeRegistrar g_tfSigmoidParser("Sigmoid", new TFActivationParser());
TFNodeRegistrar g_tfTanhParser("Tanh", new TFActivationParser());
TFNodeRegistrar g_tfSeLUParser("Selu", new TFActivationParser());
TFNodeRegistrar g_tfLeakyReluParser("LeakyRelu", new TFActivationParser());
TFNodeRegistrar g_tfSoftplusParser("Softplus", new TFActivationParser());
}  // namespace lite
}  // namespace mindspore
