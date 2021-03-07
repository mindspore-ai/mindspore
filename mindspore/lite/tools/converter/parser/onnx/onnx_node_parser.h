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
#include <utility>
#include <vector>
#include "google/protobuf/message.h"
#include "proto/onnx.pb.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "schema/inner/model_generated.h"
#include "ir/dtype/type_id.h"
#include "ops/primitive_c.h"
#include "mindspore/core/utils/check_convert_utils.h"

namespace mindspore {
namespace lite {
class OnnxNodeParser {
 public:
  explicit OnnxNodeParser(std::string nodeName) : name(std::move(nodeName)) {}

  virtual ~OnnxNodeParser() = default;

  virtual ops::PrimitiveC *Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
    return nullptr;
  }

  static STATUS set_opset_version(int version) {
    opset_version_ = version;
    return RET_OK;
  }
  static int opset_version() { return opset_version_; }

 protected:
  static mindspore::PadMode GetOnnxPadMode(const onnx::AttributeProto &onnx_node_attr);

  static STATUS GetPadMode(const onnx::AttributeProto &onnx_node_attr, std::string *mode);

  STATUS GetTensorDataFromOnnx(const onnx::TensorProto &onnx_tensor, std::vector<float> *value, int *type);

  static void Split(const std::string &src_str, std::vector<std::string> *dst_str, const std::string &chr);

  const std::string name;

 private:
  static int opset_version_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_NODE_PARSER_H
