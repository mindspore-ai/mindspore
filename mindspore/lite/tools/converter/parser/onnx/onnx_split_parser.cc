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

#include "tools/converter/parser/onnx/onnx_split_parser.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "ops/split.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxSplitParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Split>();

  prim->set_axis(0);
  std::vector<int64_t> size_splits;
  int64_t split_num = 0;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "axis") {
      prim->set_axis(onnx_node_attr.i());
    } else if (attribute_name == "split") {
      size_splits.resize(onnx_node_attr.ints_size());
      std::copy(onnx_node_attr.ints().begin(), onnx_node_attr.ints().end(), size_splits.begin());
      prim->set_size_splits(size_splits);
      split_num = onnx_node_attr.ints_size();
    }
  }
  if (split_num == 0) {
    split_num = onnx_node.output_size();
  }
  prim->set_output_num(split_num);

  return prim.release();
}

OnnxNodeRegistrar g_onnxSplitParser("Split", new OnnxSplitParser());
}  // namespace lite
}  // namespace mindspore
