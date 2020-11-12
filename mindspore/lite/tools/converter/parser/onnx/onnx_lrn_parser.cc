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

namespace mindspore {
namespace lite {
STATUS OnnxLrnParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx LrnParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::LocalResponseNormalizationT> attr = std::make_unique<schema::LocalResponseNormalizationT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  auto onnx_node_attr = onnx_node.attribute();
  int32_t size = 0;
  for (int i = 0; i < onnx_node_attr.size(); ++i) {
    MS_ASSERT(onnx_node_attr.at(i) != nullptr);
    if (onnx_node_attr.at(i).name() == "alpha") {
      attr->alpha = onnx_node_attr.at(i).f();
    } else if (onnx_node_attr.at(i).name() == "beta") {
      attr->beta = onnx_node_attr.at(i).f();
    } else if (onnx_node_attr.at(i).name() == "bias") {
      attr->bias = onnx_node_attr.at(i).f();
    } else if (onnx_node_attr.at(i).name() == "size") {
      size = static_cast<int32_t>(onnx_node_attr.at(i).i());
      attr->depth_radius = size / 2;
    }
  }

  if (size == 0) {
    MS_LOG(ERROR) << "Divide-by-zero error.";
    return RET_ERROR;
  }
  attr->alpha /= size;

  op->primitive->value.type = schema::PrimitiveType_LocalResponseNormalization;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

OnnxNodeRegistrar g_onnxLrnxParser("Lrn", new OnnxLrnParser());
OnnxNodeRegistrar g_onnxLRNxParser("LRN", new OnnxLrnParser());
}  // namespace lite
}  // namespace mindspore
