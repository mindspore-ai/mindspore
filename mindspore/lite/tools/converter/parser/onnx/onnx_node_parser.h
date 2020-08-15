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

#ifndef MS_ONNX_NODE_PARSER_H
#define MS_ONNX_NODE_PARSER_H

#include <string>
#include "google/protobuf/message.h"
#include "tools/converter/parser/onnx/onnx.pb.h"
#include "tools/common/node_util.h"
#include "schema/inner/model_generated.h"

// using namespace std;

namespace mindspore {
namespace lite {
class OnnxNodeParser {
 public:
  explicit OnnxNodeParser(const std::string &nodeName) : name(nodeName) {}
  virtual ~OnnxNodeParser() = default;
  virtual STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) = 0;

 protected:
  schema::PadMode GetOnnxPadMode(const onnx::AttributeProto &onnx_node_attr);
  const std::string &name;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MS_ONNX_NODE_PARSER_H

