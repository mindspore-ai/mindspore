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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_CONV_BASE_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_CONV_BASE_PARSER_H_

#include <string>
#include <vector>
#include "tools/converter/parser/caffe/caffe_node_parser.h"
#include "tools/converter/parser/caffe/caffe_node_parser_registry.h"

namespace mindspore {
namespace lite {
class CaffeConvBaseParser {
 public:
  CaffeConvBaseParser() = default;

  virtual ~CaffeConvBaseParser() = default;

  static STATUS ParsePads(const caffe::ConvolutionParameter &conv_param, std::vector<int64_t> *pad);

  static STATUS ParseStrides(const caffe::ConvolutionParameter &conv_param, std::vector<int64_t> *stride);

  static STATUS ParseDilations(const caffe::ConvolutionParameter &conv_param, std::vector<int64_t> *dilation);

  static STATUS ParseKernels(const caffe::ConvolutionParameter &conv_param, std::vector<int64_t> *kernel);

  static int ParseGroup(const caffe::ConvolutionParameter &convParam, const std::string &layerType);

  static int ParseChannelOut(const caffe::ConvolutionParameter &convParam, int32_t *channelOut);

  static STATUS ParseWeight(const caffe::LayerParameter &weight, std::vector<schema::TensorT *> *weightVec);
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_CONV_BASE_PARSER_H_
