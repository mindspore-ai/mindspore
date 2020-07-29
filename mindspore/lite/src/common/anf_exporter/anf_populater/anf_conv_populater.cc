/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/common/anf_exporter/anf_populater/anf_conv_populater.h"
#include <string>
#include <vector>
#include <memory>
#include "src/common/anf_exporter/anf_populater/anf_node_populater_registry.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"

namespace mindspore::lite {
int mindspore::lite::AnfConvPopulater::Parse(mindspore::CNodePtr cnodePtr, schema::CNodeT *node,
                                          std::vector<schema::TensorT *> *outputs) {
  auto p = GetCNodePrimitive(cnodePtr);
  int group = GetValue<int>(p->GetAttr("group"));

  if (group > 1) {
    auto attr = std::make_unique<schema::DepthwiseConv2DT>();
    auto format = GetValue<std::string>(p->GetAttr("data_format"));
    if (format == "NCHW") {
      attr->format = schema::Format_NCHW;
    } else if (format == "NHWC") {
      attr->format = schema::Format_NHWC;
    } else {
      attr->format = schema::Format_NUM_OF_FORMAT;
    }
    auto pad_list = GetValue<std::vector<int>>(p->GetAttr("pad_list"));
    attr->padUp = pad_list[0];
    attr->padDown = pad_list[1];
    attr->padLeft = pad_list[2];
    attr->padRight = pad_list[3];

    auto dilation = GetValue<std::vector<int>>(p->GetAttr("dilation"));
    attr->dilateH = dilation[0];
    attr->dilateW = dilation[1];

    auto kernel_size = GetValue<std::vector<int>>(p->GetAttr("kernel_size"));
    attr->kernelH = kernel_size[0];
    attr->kernelW = kernel_size[1];

    auto stride = GetValue<std::vector<int>>(p->GetAttr("stride"));
    attr->strideH = stride[2];
    attr->strideW = stride[3];

    auto pad_mode = GetValue<std::string>(p->GetAttr("pad_mode"));
    if (pad_mode == "valid") {
      attr->padMode = schema::PadMode_VALID;
    } else if (pad_mode == "same") {
      attr->padMode = schema::PadMode_SAME;
    } else {
      attr->padMode = schema::PadMode_NOTSET;
    }

    node->nodeType = schema::NodeType_CNode;
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_DepthwiseConv2D;
    node->primitive->value.value = attr.release();
  } else {
    auto attr = std::make_unique<schema::Conv2DT>();
    attr->group = group;
    auto format = GetValue<std::string>(p->GetAttr("data_format"));
    if (format == "NCHW") {
      attr->format = schema::Format_NCHW;
    } else if (format == "NHWC") {
      attr->format = schema::Format_NHWC;
    } else {
      attr->format = schema::Format_NUM_OF_FORMAT;
    }
    auto pad_list = GetValue<std::vector<int>>(p->GetAttr("pad_list"));
    attr->padUp = pad_list[0];
    attr->padDown = pad_list[1];
    attr->padLeft = pad_list[2];
    attr->padRight = pad_list[3];

    auto dilation = GetValue<std::vector<int>>(p->GetAttr("dilation"));
    attr->dilateH = dilation[0];
    attr->dilateW = dilation[1];

    auto kernel_size = GetValue<std::vector<int>>(p->GetAttr("kernel_size"));
    attr->kernelH = kernel_size[0];
    attr->kernelW = kernel_size[1];

    auto stride = GetValue<std::vector<int>>(p->GetAttr("stride"));
    attr->strideH = stride[2];
    attr->strideW = stride[3];

    attr->channelOut = GetValue<int>(p->GetAttr("out_channel"));

    auto pad_mode = GetValue<std::string>(p->GetAttr("pad_mode"));
    if (pad_mode == "valid") {
      attr->padMode = schema::PadMode_VALID;
    } else if (pad_mode == "same") {
      attr->padMode = schema::PadMode_SAME;
    } else {
      attr->padMode = schema::PadMode_NOTSET;
    }
    node->primitive = std::make_unique<schema::PrimitiveT>();
    node->primitive->value.type = schema::PrimitiveType_Conv2D;
    node->primitive->value.value = attr.release();
  }
  return 0;
}

AnfNodePopulaterRegistrar anfConvParser("Conv2D", new AnfConvPopulater());
}  // namespace mindspore::lite
