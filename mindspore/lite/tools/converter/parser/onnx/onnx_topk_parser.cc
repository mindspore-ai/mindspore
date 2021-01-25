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

#include "tools/converter/parser/onnx/onnx_topk_parser.h"
#include <memory>
#include "ops/fusion/topk_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxTopkParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::TopKFusion;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new TopKFusion failed";
    return nullptr;
  }

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "k") {
      primitive_c->AddAttr("k", MakeValue(static_cast<int32_t>(onnx_node_attr.i())));
    }
  }

  return primitive_c;
}

OnnxNodeRegistrar g_onnxTopkParser("TopK", new OnnxTopkParser());
}  // namespace lite
}  // namespace mindspore
