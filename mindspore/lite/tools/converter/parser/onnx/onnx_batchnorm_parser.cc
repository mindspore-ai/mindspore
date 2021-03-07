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

#include "tools/converter/parser/onnx/onnx_batchnorm_parser.h"
#include <memory>
#include "ops/fused_batch_norm.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxBatchNormParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::FusedBatchNorm>();

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "epsilon") {
      prim->set_epsilon(onnx_node_attr.f());
    } else if (onnx_node_attr.name() == "momentum") {
      prim->set_momentum(onnx_node_attr.f());
    }
  }

  return prim.release();
}

OnnxNodeRegistrar g_onnxBatchNormParser("BatchNormalization", new OnnxBatchNormParser());
}  // namespace lite
}  // namespace mindspore
