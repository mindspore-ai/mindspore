/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "parser/caffe/caffe_activation_parser.h"
#include <memory>
#include <limits>
#include <cmath>
#include "common/op_attr.h"
#include "ops/fusion/activation.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeReluParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Activation>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_activation_type(mindspore::ActivationType::RELU);

  if (proto.has_relu_param() && proto.relu_param().has_negative_slope()) {
    float negative_slope = proto.relu_param().negative_slope();
    if (std::fabs(negative_slope) > std::numeric_limits<float>::epsilon()) {
      prim->set_activation_type(mindspore::ActivationType::LEAKY_RELU);
      prim->set_alpha(negative_slope);
    }
  }

  return prim;
}

BaseOperatorPtr CaffeRelu6Parser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Activation>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_activation_type(mindspore::ActivationType::RELU6);
  if (proto.has_relu6_param() && proto.relu6_param().has_negative_slope()) {
    float negative_slope = proto.relu6_param().negative_slope();
    if (std::fabs(negative_slope) > std::numeric_limits<float>::epsilon()) {
      prim->set_alpha(negative_slope);
    }
  }
  return prim;
}

BaseOperatorPtr CaffeSigmoidParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Activation>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_activation_type(mindspore::ActivationType::SIGMOID);

  return prim;
}

BaseOperatorPtr CaffeTanhParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Activation>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_activation_type(mindspore::ActivationType::TANH);

  return prim;
}

BaseOperatorPtr CaffeEluParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Activation>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_activation_type(mindspore::ActivationType::ELU);

  if (proto.has_elu_param()) {
    const caffe::ELUParameter &eluParameter = proto.elu_param();
    if (eluParameter.has_alpha()) {
      prim->set_alpha(eluParameter.alpha());
    }
  }

  return prim;
}

BaseOperatorPtr CaffeHswishParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Activation>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_activation_type(mindspore::ActivationType::HSWISH);
  if (proto.has_hswish_param()) {
    const caffe::HswishParameter &hswishParameter = proto.hswish_param();
    if (hswishParameter.has_negative_slope()) {
      (void)prim->AddAttr(dpico::kNegativeSlope, api::MakeValue<float>(hswishParameter.negative_slope()));
    }
  }
  return prim;
}

CaffeNodeRegistrar g_caffeReluParser("ReLU", new CaffeReluParser());
CaffeNodeRegistrar g_caffeRelu6Parser("ReLU6", new CaffeRelu6Parser());
CaffeNodeRegistrar g_caffeSigmoidParser("Sigmoid", new CaffeSigmoidParser());
CaffeNodeRegistrar g_caffeTanhParser("TanH", new CaffeTanhParser());
CaffeNodeRegistrar g_caffeEluParser("ELU", new CaffeEluParser());
CaffeNodeRegistrar g_caffeHswishParser("Hswish", new CaffeHswishParser());
}  // namespace lite
}  // namespace mindspore
