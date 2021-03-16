/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_splice_parser.h"
#include <vector>
#include <string>
#include <memory>
#include "ops/splice.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxSpliceParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx Splice Parser";
  auto primitive = std::make_unique<ops::Splice>();
  std::vector<int64_t> context;
  std::vector<int64_t> forward_indexes;
  int64_t output_dim = 0;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const std::string attribute_name = onnx_node_attr.name();
    if (attribute_name == "context") {
      const int32_t size = onnx_node_attr.ints_size();
      context.resize(size);
      for (int32_t i = 0; i < size; i++) {
        context[i] = static_cast<int>(onnx_node_attr.ints(i));
      }
    } else if (attribute_name == "forward_indexes") {
      const int32_t size = onnx_node_attr.ints_size();
      forward_indexes.resize(size);
      for (int32_t i = 0; i < size; i++) {
        forward_indexes[i] = static_cast<int>(onnx_node_attr.ints(i));
      }
    } else if (attribute_name == "output_dim") {
      output_dim = static_cast<int>(onnx_node_attr.i());
    } else {
      MS_LOG(ERROR) << "unsupported attribute in splice " << attribute_name;
      return nullptr;
    }
  }
  primitive->Init(context, forward_indexes, output_dim);
  return primitive.release();
}

OnnxNodeRegistrar g_onnxSpliceParser("Splice", new OnnxSpliceParser());
}  // namespace lite
}  // namespace mindspore
