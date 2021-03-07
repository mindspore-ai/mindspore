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

#include "tools/converter/parser/onnx/onnx_pad_parser.h"
#include <memory>
#include <vector>
#include "ops/fusion/pad_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxPadParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::PadFusion>();

  mindspore::PaddingMode padding_mode;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "pads") {
      const int size = onnx_node_attr.ints_size();
      std::vector<std::vector<int64_t>> paddings(size / 2, std::vector<int64_t>(2, 0));
      // begin1, begin2, begin3... end1, end2, end3... to
      // begin1, end1, begin2, end2, begin3, end3...
      for (int i = 0; i < size / 2; i++) {
        paddings[i][0] = static_cast<int64_t>(onnx_node_attr.ints(i));
        paddings[i][1] = static_cast<int64_t>(onnx_node_attr.ints(i + size / 2));
      }
      prim->set_paddings(paddings);
    } else if (attribute_name == "mode") {
      const auto &mode = onnx_node_attr.s();
      if (mode == "constant") {
        padding_mode = mindspore::PaddingMode::CONSTANT;
      } else if (mode == "reflect") {
        padding_mode = mindspore::PaddingMode::REFLECT;
      } else if (mode == "edge") {
        padding_mode = mindspore::PaddingMode::SYMMETRIC;
      }
      prim->set_padding_mode(padding_mode);
    }
  }

  return prim.release();
}

OnnxNodeRegistrar g_onnxPadParser("Pad", new OnnxPadParser());
}  // namespace lite
}  // namespace mindspore
