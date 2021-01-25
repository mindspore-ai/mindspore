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

#include "tools/converter/parser/caffe/caffe_activation_parser.h"
#include <memory>
#include "ops/fusion/activation.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeReluParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto primitive_c = new (std::nothrow) ops::Activation();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new ReLU failed";
    return nullptr;
  }

  primitive_c->set_activation_type(mindspore::ActivationType::RELU);

  if (proto.has_relu_param() && proto.relu_param().has_negative_slope()) {
    float negative_slope = proto.relu_param().negative_slope();
    if (negative_slope != 0) {
      primitive_c->set_activation_type(mindspore::ActivationType::LEAKY_RELU);
      primitive_c->set_alpha(negative_slope);
    }
  }

  return primitive_c;
}

ops::PrimitiveC *CaffeRelu6Parser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto primitive_c = new (std::nothrow) ops::Activation();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Relu6 failed";
    return nullptr;
  }

  primitive_c->set_activation_type(mindspore::ActivationType::RELU6);

  return primitive_c;
}

ops::PrimitiveC *CaffeSigmoidParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto primitive_c = new (std::nothrow) ops::Activation();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Sigmoid failed";
    return nullptr;
  }

  primitive_c->set_activation_type(mindspore::ActivationType::SIGMOID);

  return primitive_c;
}

ops::PrimitiveC *CaffeTanhParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto primitive_c = new (std::nothrow) ops::Activation();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Tanh failed";
    return nullptr;
  }

  primitive_c->set_activation_type(mindspore::ActivationType::TANH);

  return primitive_c;
}

CaffeNodeRegistrar g_caffeReluParser("ReLU", new CaffeReluParser());
CaffeNodeRegistrar g_caffeRelu6Parser("ReLU6", new CaffeRelu6Parser());
CaffeNodeRegistrar g_caffeSigmoidParser("Sigmoid", new CaffeSigmoidParser());
CaffeNodeRegistrar g_caffeTanhParser("TanH", new CaffeTanhParser());
}  // namespace lite
}  // namespace mindspore
