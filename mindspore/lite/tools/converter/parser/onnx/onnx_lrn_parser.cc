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

#include "tools/converter/parser/onnx/onnx_lrn_parser.h"
#include <memory>
#include "ops/lrn.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxLrnParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::LRN>();

  int64_t size = 0;
  float alpha = 0;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "alpha") {
      alpha = onnx_node_attr.f();
    } else if (attribute_name == "beta") {
      prim->set_beta(onnx_node_attr.f());
    } else if (attribute_name == "bias") {
      prim->set_bias(onnx_node_attr.f());
    } else if (attribute_name == "size") {
      size = onnx_node_attr.i();
      prim->set_depth_radius(size / 2);
    }
  }

  if (size == 0) {
    MS_LOG(ERROR) << "Divide-by-zero error.";
    return nullptr;
  }
  alpha /= size;
  prim->set_alpha(alpha);

  return prim.release();
}

OnnxNodeRegistrar g_onnxLrnxParser("Lrn", new OnnxLrnParser());
OnnxNodeRegistrar g_onnxLRNxParser("LRN", new OnnxLrnParser());
}  // namespace lite
}  // namespace mindspore
