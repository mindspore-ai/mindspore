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

#include "tools/converter/parser/onnx/onnx_expand_parser.h"
#include <memory>
#include <vector>
#include "ops/broadcast_to.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxExpandParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::BroadcastTo>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  std::vector<int64_t> dst_shape;
  if (onnx_node.input_size() != kInputSize1) {
    for (const auto &onnx_node_attr : onnx_node.attribute()) {
      const auto &attribute_name = onnx_node_attr.name();
      if (attribute_name == "shape") {
        for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
          dst_shape.push_back(static_cast<int64_t>(onnx_node_attr.ints(i)));
        }
      }
    }
  } else {
    const auto &onnx_expand_power = onnx_node.input(1);
    auto node_iter =
      std::find_if(onnx_graph.node().begin(), onnx_graph.node().end(),
                   [onnx_expand_power](const onnx::NodeProto &proto) { return proto.output(0) == onnx_expand_power; });
    if (node_iter != onnx_graph.node().end()) {
      for (const auto &attr_power : node_iter->attribute()) {
        if (attr_power.name() == "value") {
          const auto &t = attr_power.t();
          auto *shape_data = t.raw_data().data();
          MS_CHECK_TRUE_RET(shape_data != nullptr, nullptr);
          auto *data_ptr = reinterpret_cast<const int64_t *>(shape_data);
          for (int i = 0; i < t.dims(0); ++i) {
            dst_shape.emplace_back(data_ptr[i]);
          }
        }
      }
    }
  }
  if (!dst_shape.empty()) {
    prim->set_shape(dst_shape);
  }

  return prim.release();
}

OnnxNodeRegistrar g_onnxExpandSpaceParser("Expand", new OnnxExpandParser());
}  // namespace lite
}  // namespace mindspore
