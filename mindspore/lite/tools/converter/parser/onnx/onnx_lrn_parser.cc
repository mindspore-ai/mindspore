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
STATUS OnnxLrnParser::Parse(const onnx::GraphProto &onnx_graph,
                            const onnx::NodeProto &onnx_node,
                            schema::CNodeT *op) {
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

  std::unique_ptr<schema::LocalResponseNormalizationT> attr
    = std::make_unique<schema::LocalResponseNormalizationT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  auto onnx_node_attr = onnx_node.attribute().at(0);
  int32_t size = 0;
  if (onnx_node_attr.name() == "size") {
    size = static_cast<int32_t>(onnx_node_attr.i());
    attr->depth_radius = static_cast<int32_t>(size / 2);
  } else {
    MS_LOG(ERROR) << "the first attr is not size";
    return RET_ERROR;
  }

  onnx_node_attr = onnx_node.attribute().at(1);
  if (onnx_node_attr.name() == "alpha") {
    auto alpha = onnx_node_attr.f();
    attr->alpha = alpha / size;
  } else {
    MS_LOG(ERROR) << "the second attr is not alpha";
    return RET_ERROR;
  }

  onnx_node_attr = onnx_node.attribute().at(2);
  if (onnx_node_attr.name() == "beta") {
    attr->beta = onnx_node_attr.f();
  } else {
    MS_LOG(ERROR) << "the third attr is not beta";
    return RET_ERROR;
  }

  onnx_node_attr = onnx_node.attribute().at(3);
  if (onnx_node_attr.name() == "bias") {
    attr->bias = onnx_node_attr.f();
  } else {
    MS_LOG(ERROR) << "the third attr is not bias";
    return RET_ERROR;
  }

  op->primitive->value.type = schema::PrimitiveType_LocalResponseNormalization;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

OnnxNodeRegistrar g_onnxLrnxParser("Lrn", new OnnxLrnParser());
}  // namespace lite
}  // namespace mindspore

