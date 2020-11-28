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

namespace mindspore {
namespace lite {
STATUS OnnxExpandParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                               schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx ExpandParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::BroadcastToT> attr = std::make_unique<schema::BroadcastToT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  std::vector<int> dst_shape;
  const auto &onnx_expand_power = onnx_node.input(1);
  auto node_iter =
    std::find_if(onnx_graph.node().begin(), onnx_graph.node().end(),
                 [onnx_expand_power](const onnx::NodeProto &proto) { return proto.output(0) == onnx_expand_power; });
  if (node_iter == onnx_graph.node().end()) {
    MS_LOG(ERROR) << "can not find node: " << onnx_expand_power;
    return RET_ERROR;
  }
  for (const auto &attrPower : node_iter->attribute()) {
    if (attrPower.name() == "value") {
      const auto &t = attrPower.t();
      auto *dataPtr = reinterpret_cast<const int64_t *>(t.raw_data().data());
      for (int i = 0; i < t.dims(0); ++i) {
        dst_shape.emplace_back(dataPtr[i]);
      }
    }
  }
  attr->dst_shape = dst_shape;
  op->primitive->value.type = schema::PrimitiveType_BroadcastTo;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

OnnxNodeRegistrar g_onnxExpandSpaceParser("Expand", new OnnxExpandParser());
}  // namespace lite
}  // namespace mindspore
