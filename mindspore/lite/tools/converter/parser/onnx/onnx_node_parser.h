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
  explicit OnnxNodeParser(std::string node_name) : name_(std::move(node_name)) {}

  virtual ~OnnxNodeParser() = default;

  virtual ops::PrimitiveC *Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
    return nullptr;
  }

  static STATUS set_opset_version(int64_t version) {
    opset_version_ = version;
    return RET_OK;
  }

  static int64_t opset_version() { return opset_version_; }

  static STATUS CopyOnnxTensorData(const onnx::TensorProto &onnx_const_tensor, const tensor::TensorPtr &tensor_info);

  static TypeId GetDataTypeFromOnnx(onnx::TensorProto_DataType onnx_type);

  static size_t GetOnnxElementNum(const onnx::TensorProto &onnx_tensor, bool *overflowed);

 protected:
  static mindspore::PadMode GetOnnxPadMode(const onnx::AttributeProto &onnx_node_attr);

  static STATUS GetTensorDataFromOnnx(const onnx::TensorProto &onnx_tensor, std::vector<float> *value, int *type);

  static const void *GetOnnxRawData(const onnx::TensorProto &onnx_const_tensor, TypeId data_type, size_t data_count,
                                    size_t *data_size);

  const std::string name_{};

 private:
  static int64_t opset_version_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_NODE_PARSER_H
