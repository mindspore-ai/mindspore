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
#include "tools/converter/parser/tf/tf_conv_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "tools/converter/parser/tf/tf_util.h"
#include "ops/fusion/conv2d_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFConvParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto primitive_c = new (std::nothrow) ops::Conv2DFusion;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Conv2DFusion failed";
    return nullptr;
  }

  primitive_c->set_pad({0, 0, 0, 0});
  primitive_c->set_group(1);

  // parse format
  auto format = TensorFlowUtils::ParseNodeFormat(tf_op);
  if (format == mindspore::Format::NCHW) {
    MS_LOG(ERROR) << "TF Conv2D with data_format=NCHW is not supported now";
    return nullptr;
  }
  primitive_c->set_format(format);

  // parse kernel
  auto weight_node = GetConstInputNode(tf_node_map, tf_op.input(1));
  if (weight_node == nullptr) {
    MS_LOG(ERROR) << "Find Conv2D input weights failed";
    return nullptr;
  }
  std::vector<int64_t> kernels(4);
  if (ParseKernels(*weight_node, format, &kernels) != RET_OK) {
    MS_LOG(ERROR) << "parse kernels failed";
    return nullptr;
  }
  primitive_c->set_kernel_size({kernels[0], kernels[1]});
  primitive_c->set_out_channel(kernels[3]);
  primitive_c->set_in_channel(kernels[2]);

  // parse stride
  std::vector<int64_t> strides(2);
  if (ParseStrides(tf_op, format, &strides) != RET_OK) {
    MS_LOG(ERROR) << "parse strides failed";
    return nullptr;
  }
  primitive_c->set_stride(strides);

  // parse dilation
  std::vector<int64_t> dilations(2);
  if (ParseDilations(tf_op, format, &dilations) != RET_OK) {
    MS_LOG(ERROR) << "parse dilations failed";
    return nullptr;
  }
  primitive_c->set_dilation(dilations);

  // parse pad
  auto padMode = ParsePadMode(tf_op);
  primitive_c->set_pad_mode(padMode);

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return primitive_c;
}

TFNodeRegistrar g_tfConvParser("Conv2D", new TFConvParser());
}  // namespace lite
}  // namespace mindspore
