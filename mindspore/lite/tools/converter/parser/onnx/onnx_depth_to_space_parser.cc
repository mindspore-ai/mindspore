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

#include <memory>
#include "tools/converter/parser/onnx/onnx_depth_to_space_parser.h"

namespace mindspore {
namespace lite {
STATUS OnnxDepthToSpaceParser::Parse(const onnx::GraphProto &onnx_graph,
                                     const onnx::NodeProto &onnx_node,
                                     schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx DepthToSpaceParser";
  std::unique_ptr<schema::DepthToSpaceT> attr(new schema::DepthToSpaceT());
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto& attribute_name = onnx_node_attr.name();
    if (attribute_name == "blocksize") {
      attr->blockSize = static_cast<int32_t>(onnx_node_attr.i());
    }
  }
  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_DepthToSpace;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxDepthToSpaceParser("DepthToSpace", new OnnxDepthToSpaceParser());
}  // namespace lite
}  // namespace mindspore

