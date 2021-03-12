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
#include "tools/converter/parser/tf/tf_deconv_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "tools/converter/parser/tf/tf_util.h"
#include "ops/fusion/conv2d_transpose_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFDeconvParser::Parse(const tensorflow::NodeDef &tf_op,
                                       const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                       std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Conv2dTransposeFusion>();

  prim->set_group(1);
  prim->set_pad({0, 0, 0, 0});
  auto format = TensorFlowUtils::ParseNodeFormat(tf_op);
  prim->set_format(format);
  prim->set_output_paddings({0, 0});

  std::vector<int64_t> dilations(2);
  if (ParseDilations(tf_op, format, &dilations) != RET_OK) {
    MS_LOG(ERROR) << "parse dilations failed";
    return nullptr;
  }
  prim->set_dilation({dilations[0], dilations[1]});

  std::vector<int64_t> strides(2);
  if (ParseStrides(tf_op, format, &strides) != RET_OK) {
    MS_LOG(ERROR) << "parse strides failed";
    return nullptr;
  }
  prim->set_stride({strides[0], strides[1]});

  auto weight_node = GetConstInputNode(tf_node_map, tf_op.input(1));
  if (weight_node != nullptr) {
    std::vector<int64_t> kernels(4);
    if (ParseKernels(*weight_node, format, &kernels) != RET_OK) {
      MS_LOG(ERROR) << "parse kernels failed";
      return nullptr;
    }
    prim->set_kernel_size({kernels[0], kernels[1]});
    prim->set_out_channel(kernels[2]);
    prim->set_in_channel(kernels[3]);
  } else {
    prim->set_kernel_size({-1, -1});
    prim->set_out_channel(-1);
    prim->set_in_channel(-1);
    MS_LOG(WARNING) << "parsing of kernelH/W channelIn/Out is delayed";
  }

  prim->set_pad_mode(ParsePadMode(tf_op));

  *output_size = 1;
  if (AddOpInput(tf_op, 2, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}
TFNodeRegistrar g_tf_deconv_parser("Conv2DBackpropInput", new TFDeconvParser());
}  // namespace lite
}  // namespace mindspore
