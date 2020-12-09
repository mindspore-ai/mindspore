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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_MODEL_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_MODEL_PARSER_H

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <unordered_map>
#include "securec/include/securec.h"
#include "tools/converter/model_parser.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"
#include "proto/onnx.pb.h"
#include "src/param_value_lite.h"

namespace mindspore {
namespace lite {
class OnnxModelParser : public ModelParser {
 public:
  OnnxModelParser() = default;

  ~OnnxModelParser() override = default;

  MetaGraphT *ParseToFb(const std::string &model_file, const std::string &weight_file,
                        const QuantType &quant_type) override {
    return nullptr;
  }

  FuncGraphPtr Parse(const std::string &model_file, const std::string &weight_file,
                     const QuantType &quant_type) override;
  static TypeId GetDataTypeFromOnnx(onnx::TensorProto_DataType onnx_type);
  static STATUS CopyOnnxTensorData(const onnx::TensorProto &onnx_const_value,
                                   const ParamValueLitePtr &param_value_lite);

 private:
  STATUS InitOriginModel(const std::string &model_file);
  STATUS ConvertNodes();
  STATUS ConvertConstTensors();
  STATUS ConvertGraphInputs();
  STATUS ConvertGraphOutputs();
  STATUS BuildReturnNode(const std::vector<AnfNodePtr> &return_inputs);
  STATUS BuildParameterNode(const ParameterPtr &parameter_node, const onnx::TensorProto &tensor);
  STATUS BuildParameterNodeForQuantParam(void *data, const std::string &name, TypeId type);
  STATUS BuildCNode(const onnx::NodeProto &onnx_node, lite::PrimitiveC *primitive_c);
  STATUS BuildOpOutputs(const onnx::NodeProto &onnx_node, const CNodePtr &cnode);
  STATUS ConvertSpecialOnnxNode(const onnx::NodeProto &onnx_node, lite::PrimitiveC *primitive_c);
  STATUS ConvertOnnxGemmNode(const onnx::NodeProto &onnx_node, lite::PrimitiveC *primitive_c);
  STATUS BuildCNodeForGemm(const onnx::NodeProto &onnx_node, lite::PrimitiveC *primitive_c, const std::string &name);
  STATUS ConvertOpQuantParams(const onnx::NodeProto &onnx_node, lite::PrimitiveC *primitive_c);
  STATUS ParseQuantParam(const onnx::NodeProto &onnx_node);
  STATUS SetTensorQuantParam(const std::string &tensor_name, std::vector<QuantParamT> *quant_params);
  STATUS SetTensorQuantParamFromNode(const std::string &tensor_name, std::vector<QuantParamT> *quant_params);
  STATUS CopyTensorQuantParam(const std::string &tensor_name, QuantParamT *quant_param, bool scale_or_not);
  bool IsSpecialOnnxNode(const onnx::NodeProto &onnx_node);

  onnx::ModelProto onnx_model_;
  onnx::GraphProto onnx_graph_;
  std::unordered_map<std::string, AnfNodePtr> nodes_;
  FuncGraphPtr func_graph_ptr_ = nullptr;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_MODEL_PARSER_H
