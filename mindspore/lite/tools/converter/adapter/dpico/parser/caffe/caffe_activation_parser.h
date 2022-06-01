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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_CAFFE_CAFFE_ACTIVATION_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_CAFFE_CAFFE_ACTIVATION_PARSER_H_

#include <vector>
#include "parser/caffe/caffe_node_parser.h"
#include "parser/caffe/caffe_node_parser_registry.h"

namespace mindspore {
namespace lite {
class CaffeReluParser : public CaffeNodeParser {
 public:
  CaffeReluParser() : CaffeNodeParser("relu") {}
  ~CaffeReluParser() override = default;

  BaseOperatorPtr Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) override;
};

class CaffeRelu6Parser : public CaffeNodeParser {
 public:
  CaffeRelu6Parser() : CaffeNodeParser("relu6") {}
  ~CaffeRelu6Parser() override = default;

  BaseOperatorPtr Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) override;
};

class CaffeSigmoidParser : public CaffeNodeParser {
 public:
  CaffeSigmoidParser() : CaffeNodeParser("sigmoid") {}
  ~CaffeSigmoidParser() override = default;

  BaseOperatorPtr Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) override;
};

class CaffeTanhParser : public CaffeNodeParser {
 public:
  CaffeTanhParser() : CaffeNodeParser("tanh") {}
  ~CaffeTanhParser() override = default;

  BaseOperatorPtr Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) override;
};

class CaffeEluParser : public CaffeNodeParser {
 public:
  CaffeEluParser() : CaffeNodeParser("elu") {}
  ~CaffeEluParser() override = default;

  BaseOperatorPtr Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) override;
};

class CaffeHswishParser : public CaffeNodeParser {
 public:
  CaffeHswishParser() : CaffeNodeParser("Hswish") {}
  ~CaffeHswishParser() override = default;

  BaseOperatorPtr Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_CAFFE_CAFFE_ACTIVATION_PARSER_H_
