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

#include "tools/converter/parser/onnx/onnx_conv_transpose_parser.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
const size_t kInputSizeThree = 3;
}
STATUS OnnxDeConvParser::ParseOnnxAttr(const onnx::NodeProto &onnx_node, int64_t *group, mindspore::PadMode *pad_mode,
                                       std::vector<int64_t> *output_paddings) {
  MS_ASSERT(group != nullptr);
  MS_ASSERT(pad_mode != nullptr);
  MS_ASSERT(output_paddings != nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "group") {
      *group = onnx_node_attr.i();
    } else if (onnx_node_attr.name() == "auto_pad") {
      *pad_mode = GetOnnxPadMode(onnx_node_attr);
    } else if (onnx_node_attr.name() == "output_padding") {
      MS_CHECK_GE(onnx_node_attr.ints_size(), kInputSize1, RET_ERROR);
      output_paddings->push_back(static_cast<int32_t>(onnx_node_attr.ints(0)));
      output_paddings->push_back(static_cast<int32_t>(onnx_node_attr.ints(1)));
    } else if (onnx_node_attr.name() == "order" && onnx_node_attr.s() != "NHWC") {
      MS_LOG(ERROR) << "Unsupported format: " << onnx_node_attr.s().c_str();
      return RET_ERROR;
    }
  }
  return RET_OK;
}

PrimitiveCPtr OnnxDeConvParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  MS_CHECK_GE(onnx_node.input_size(), kInputSize1, nullptr);
  auto prim = std::make_unique<ops::Conv2dTransposeFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  prim->set_pad({0, 0, 0, 0});
  mindspore::PadMode pad_mode = mindspore::PadMode::PAD;
  std::vector<int64_t> kernel, dilate, stride, pads, output_paddings;
  int64_t group = 1;
  auto status = ParseOnnxAttr(onnx_node, &group, &pad_mode, &output_paddings);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Parse onnx attribute failed.";
    return nullptr;
  }
  (void)prim_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(mindspore::Format::NCHW));
  if (onnx_node.input_size() == kInputSizeThree) {
    (void)prim_c->AddAttr(ops::kHasBias, MakeValue<bool>(true));
  }
  prim->set_group(group);
  prim->set_pad_mode(pad_mode);

  bool conv1d = false;
  if (ParseVecAttr(onnx_node, &kernel, &stride, &dilate, &pads, &conv1d) != RET_OK) {
    return nullptr;
  }
  if (conv1d) {
    (void)prim_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(NCW));
  }
  if (!dilate.empty()) {
    prim->set_dilation(dilate);
  }
  if (!pads.empty()) {
    prim->set_pad_list(pads);
  }
  if (!kernel.empty()) {
    prim->set_kernel_size(kernel);
  }
  if (!stride.empty()) {
    prim->set_stride(stride);
  }
  if (!output_paddings.empty()) {
    prim->set_output_paddings(output_paddings);
  } else {
    prim->set_output_paddings({0, 0});
  }

  const auto &onnx_conv_weight = onnx_node.input(1);
  auto node_iter =
    std::find_if(onnx_graph.initializer().begin(), onnx_graph.initializer().end(),
                 [onnx_conv_weight](const onnx::TensorProto &proto) { return proto.name() == onnx_conv_weight; });
  if (node_iter == onnx_graph.initializer().end()) {
    MS_LOG(WARNING) << "parsing of channelIn/Out is delayed.";
  } else {
    std::vector<int> weight_shape;
    auto size = (*node_iter).dims_size();
    weight_shape.reserve(size);
    for (int i = 0; i < size; ++i) {
      weight_shape.emplace_back((*node_iter).dims(i));
    }
    if (weight_shape.size() < DIMENSION_2D) {
      MS_LOG(ERROR) << "weight_shape.size() should not be less than 2, but is " << weight_shape.size();
      return nullptr;
    }
    prim->set_in_channel(weight_shape[0]);
    if (INT_MUL_OVERFLOW_THRESHOLD(weight_shape[1], group, INT64_MAX)) {
      MS_LOG(ERROR) << "channel out overflow";
      return nullptr;
    }
    prim->set_out_channel(weight_shape[1] * group);

    if (group != 1 && weight_shape[1] == 1) {
      (void)prim_c->AddAttr(ops::kIsDepthWise, MakeValue<bool>(true));
    }
  }

  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxDeConvParser("ConvTranspose", new OnnxDeConvParser());
}  // namespace lite
}  // namespace mindspore
