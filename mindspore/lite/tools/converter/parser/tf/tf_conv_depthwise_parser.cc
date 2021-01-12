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
#include "tools/converter/parser/tf/tf_conv_depthwise_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "tools/converter/parser/tf/tf_util.h"

namespace mindspore {
namespace lite {
STATUS TFConvDepthwiseParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    PrimitiveC **primitiveC, std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(DEBUG) << "TF ConvDepthwiseParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "New PrimitiveT failed";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::DepthwiseConv2DT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return RET_NULL_PTR;
  }

  attr->format = TensorFlowUtils::ParseNodeFormat(tf_op);
  if (attr->format == schema::Format_NCHW) {
    MS_LOG(ERROR) << "TF Conv2D with data_format=NCHW is not supported now";
    return RET_ERROR;
  }

  std::vector<int64_t> dilations(2);
  auto status = ParseDilations(tf_op, attr->format, &dilations);
  if (status != RET_OK) {
    return status;
  }
  attr->dilateH = dilations[0];
  attr->dilateW = dilations[1];

  std::vector<int64_t> strides(2);
  status = ParseStrides(tf_op, attr->format, &strides);
  if (status != RET_OK) {
    return status;
  }
  attr->strideH = strides[0];
  attr->strideW = strides[1];

  auto weight_node = GetConstInputNode(tf_node_map, tf_op.input(1));
  if (weight_node != nullptr) {
    std::vector<int64_t> kernels(4);
    status = ParseKernels(*weight_node, attr->format, &kernels);
    if (status != RET_OK) {
      return status;
    }
    attr->kernelH = kernels[0];
    attr->kernelW = kernels[1];
    attr->channelIn = kernels[2];
    attr->channelMultiplier = kernels[3];
  } else {
    attr->kernelH = -1;
    attr->kernelW = -1;
    attr->channelIn = -1;
    attr->channelMultiplier = -1;
    MS_LOG(WARNING) << "parsing of kernelH/W channelIn/Out is delayed";
  }

  status = ParsePadMode(tf_op, &attr->padMode);
  if (status != RET_OK) {
    return status;
  }

  primitive->value.type = schema::PrimitiveType_DepthwiseConv2D;
  primitive->value.value = attr.release();
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  *output_size = 1;
  status = AddOpInput(tf_op, 0, inputs);
  if (status != RET_OK) {
    return status;
  }
  status = AddOpInput(tf_op, 1, inputs);  // weights
  return status;
}
TFNodeRegistrar g_tfConvDepthwiseParser("DepthwiseConv2dNative", new TFConvDepthwiseParser());
}  // namespace lite
}  // namespace mindspore
