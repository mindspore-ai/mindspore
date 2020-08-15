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

#include <vector>
#include <memory>
#include "tools/converter/parser/onnx/onnx_reshape_parser.h"

namespace mindspore {
namespace lite {
STATUS OnnxReshapeParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx ReshapeParser";
  std::unique_ptr<schema::ReshapeT> attr(new schema::ReshapeT());
  attr->format = schema::Format_NCHW;
  std::vector<onnx::TensorProto> params;
  // TODO(wangzhe) shape may also come from other op, there need refactor to introduce tensor_cache
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
      return RET_PARAM_INVALID;
    }

    for (int i = 0; i < params[0].int64_data_size(); ++i) {
      attr->shape.emplace_back(params[0].int64_data(i));
    }
  }
  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Reshape;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxReshapeParser("Reshape", new OnnxReshapeParser());
}  // namespace lite
}  // namespace mindspore
