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
#include "tools/converter/parser/tf/tf_conv_base_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "schema/inner/model_generated.h"
namespace mindspore {
namespace lite {
STATUS TFConvBaseParser::ParseKernels(const tensorflow::NodeDef &node_def, const mindspore::Format &format,
                                      std::vector<int64_t> *kernel) {
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(node_def, "value", &attr_value)) {
    MS_LOG(ERROR) << "The kernels should be specified";
    return RET_PARAM_INVALID;
  }
  auto shape = attr_value.tensor().tensor_shape();
  if (shape.dim().size() != 4) {
    MS_LOG(ERROR) << "Dims of Kernel should be 4.";
    return RET_PARAM_INVALID;
  }
  kernel->at(0) = shape.dim(0).size();
  kernel->at(1) = shape.dim(1).size();
  kernel->at(2) = shape.dim(2).size();
  kernel->at(3) = shape.dim(3).size();
  return RET_OK;
}

STATUS TFConvBaseParser::ParseStrides(const tensorflow::NodeDef &node_def, const mindspore::Format &format,
                                      std::vector<int64_t> *strides) {
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(node_def, "strides", &attr_value)) {
    strides->at(0) = 1;
    strides->at(1) = 1;
  } else {
    auto stride_list = attr_value.list();
    if (format == mindspore::NHWC) {
      strides->at(0) = stride_list.i(1);
      strides->at(1) = stride_list.i(2);
    } else {
      strides->at(0) = stride_list.i(2);
      strides->at(1) = stride_list.i(3);
    }
  }
  return RET_OK;
}

STATUS TFConvBaseParser::ParseDilations(const tensorflow::NodeDef &node_def, const mindspore::Format &format,
                                        std::vector<int64_t> *dilations) {
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(node_def, "dilations", &attr_value)) {
    dilations->at(0) = 1;
    dilations->at(1) = 1;
  } else {
    auto dilation_list = attr_value.list();
    if (format == mindspore::NHWC) {
      dilations->at(0) = dilation_list.i(1);
      dilations->at(1) = dilation_list.i(2);
    } else {
      dilations->at(0) = dilation_list.i(2);
      dilations->at(1) = dilation_list.i(3);
    }
  }
  return RET_OK;
}

mindspore::PadMode TFConvBaseParser::ParsePadMode(const tensorflow::NodeDef &node_def) {
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(node_def, "padding", &attr_value)) {
    MS_LOG(ERROR) << "The attr padding should be specified";
    return mindspore::PadMode::VALID;
  }
  if (attr_value.s() == "SAME") {
    return mindspore::PadMode::SAME;
  }
  return mindspore::PadMode::VALID;
}
}  // namespace lite
}  // namespace mindspore
