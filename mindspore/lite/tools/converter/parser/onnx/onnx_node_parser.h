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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_NODE_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_NODE_PARSER_H

#include <string>
#include <vector>
#include "google/protobuf/message.h"
#include "tools/converter/parser/onnx/onnx.pb.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
class OnnxNodeParser {
 public:
  explicit OnnxNodeParser(const std::string &nodeName) : name(nodeName) {}

  virtual ~OnnxNodeParser() = default;

  virtual STATUS Parse(const onnx::GraphProto &onnx_graph,
                       const onnx::NodeProto &onnx_node,
                       schema::CNodeT *op) = 0;

 protected:
  schema::PadMode GetOnnxPadMode(const onnx::AttributeProto &onnx_node_attr);

  void Split(const std::string &src_str,
             std::vector<std::string> *dst_str,
             const std::string &chr);

  const std::string &name;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_NODE_PARSER_H

