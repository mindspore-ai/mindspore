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
STATUS OnnxReshapeParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
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
  std::vector<int64_t> shape;
  shape.clear();
  if (onnx_node.input_size() != 2) {
    for (const auto &onnx_node_attr : onnx_node.attribute()) {
      const auto &attribute_name = onnx_node_attr.name();
      if (attribute_name == "shape") {
        for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
          shape.push_back(static_cast<int64_t>(onnx_node_attr.ints(i)));
        }
      }
    }
  } else {
    onnx::TensorProto input_shape;
    const auto &shape_name = onnx_node.input(1);
    for (const auto &it : onnx_graph.initializer()) {
      if (it.name() == shape_name) {
        input_shape = it;
        break;
      }
    }
    if (input_shape.int64_data_size() == 0) {
      MS_LOG(INFO) << "shape maybe from another op other than const initializer";
    } else {
      for (int i = 0; i < input_shape.int64_data_size(); ++i) {
        shape.push_back(input_shape.int64_data(i));
      }
    }
  }
  attr->shape = shape;

  op->primitive->value.type = schema::PrimitiveType_Reshape;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

OnnxNodeRegistrar g_onnxReshapeParser("Reshape", new OnnxReshapeParser());
}  // namespace lite
}  // namespace mindspore
