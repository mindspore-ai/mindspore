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

#include "tools/converter/parser/onnx/onnx_reshape_parser.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {
STATUS OnnxReshapeParser::Parse(const onnx::GraphProto &onnx_graph,
                                const onnx::NodeProto &onnx_node,
                                schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx ReshapeParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::ReshapeT> attr = std::make_unique<schema::ReshapeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  attr->format = schema::Format_NCHW;
  std::vector<onnx::TensorProto> params;
  for (int i = 0; i < onnx_node.input_size(); ++i) {
    const auto &input_name = onnx_node.input(i);
    for (const auto &it : onnx_graph.initializer()) {
      if (it.name() == input_name) {
        params.emplace_back(it);
        break;
      }
    }
  }
  if (params.empty()) {
    MS_LOG(DEBUG) << "shape from another op other than const initializer";
  } else {
    if (params.size() != 1) {
      MS_LOG(ERROR) << "shape param num is " << params.size() << ", not equal to 1";
      return RET_ERROR;
    }

    for (int i = 0; i < params[0].int64_data_size(); ++i) {
      attr->shape.emplace_back(params[0].int64_data(i));
    }
  }

  op->primitive->value.type = schema::PrimitiveType_Reshape;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

OnnxNodeRegistrar g_onnxReshapeParser("Reshape", new OnnxReshapeParser());
}  // namespace lite
}  // namespace mindspore
