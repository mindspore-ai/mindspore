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
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/fusion/activation.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFActivationParser::Parse(const tensorflow::NodeDef &tf_op,
                                           const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                           std::vector<std::string> *inputs, int *output_size) {
  auto primitive_c = new (std::nothrow) ops::Activation();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Activation failed";
    return nullptr;
  }

  if (tf_op.op() == "Relu") {
    primitive_c->set_activation_type(mindspore::ActivationType::RELU);
  } else if (tf_op.op() == "Relu6") {
    primitive_c->set_activation_type(mindspore::ActivationType::RELU6);
  } else if (tf_op.op() == "Sigmoid") {
    primitive_c->set_activation_type(mindspore::ActivationType::SIGMOID);
  } else if (tf_op.op() == "Tanh") {
    primitive_c->set_activation_type(mindspore::ActivationType::TANH);
  } else {
    MS_LOG(ERROR) << "unsupported activation type:" << tf_op.op();
    return nullptr;
  }

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }
  return primitive_c;
}

TFNodeRegistrar g_tfReluParser("Relu", new TFActivationParser());
TFNodeRegistrar g_tfRelu6Parser("Relu6", new TFActivationParser());
TFNodeRegistrar g_tfSigmoidParser("Sigmoid", new TFActivationParser());
TFNodeRegistrar g_tfTanhParser("Tanh", new TFActivationParser());
}  // namespace lite
}  // namespace mindspore
