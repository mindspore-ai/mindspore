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

namespace mindspore {
namespace lite {
STATUS TFActivationParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 PrimitiveC **primitiveC, std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF ActivationParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::ActivationT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  if (tf_op.op() == "Relu") {
    attr->type = schema::ActivationType_RELU;
  } else if (tf_op.op() == "Relu6") {
    attr->type = schema::ActivationType_RELU6;
  } else if (tf_op.op() == "Sigmoid") {
    attr->type = schema::ActivationType_SIGMOID;
  } else if (tf_op.op() == "Tanh") {
    attr->type = schema::ActivationType_TANH;
  } else {
    MS_LOG(ERROR) << "unsupported activation type:" << tf_op.op();
    return RET_ERROR;
  }

  primitive->value.type = schema::PrimitiveType_Activation;
  primitive->value.value = attr.release();
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  *output_size = 1;
  auto status = AddOpInput(tf_op, 0, inputs);
  return status;
}
TFNodeRegistrar g_tfReluParser("Relu", new TFActivationParser());
TFNodeRegistrar g_tfRelu6Parser("Relu6", new TFActivationParser());
TFNodeRegistrar g_tfSigmoidParser("Sigmoid", new TFActivationParser());
TFNodeRegistrar g_tfTanhParser("Tanh", new TFActivationParser());
}  // namespace lite
}  // namespace mindspore
