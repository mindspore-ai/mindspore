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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_ANF_IMPORTER_IMPORTER_FROM_PROTOBUF_H_
#define MINDSPORE_LITE_TOOLS_COMMON_ANF_IMPORTER_IMPORTER_FROM_PROTOBUF_H_

#include <map>
#include <string>
#include <unordered_map>
#include <utility>

#include "include/errorcode.h"
#include "proto/onnx.pb.h"
#include "tools/converter/converter_context.h"
#include "tools/anf_importer/anf_importer.h"
#include "abstract/abstract_value.h"

namespace mindspore::lite {
class AnfImporterFromMindir : public AnfImporter {
 public:
  AnfImporterFromMindir() = default;

  ~AnfImporterFromMindir() override { delete onnx_model_; }

  static onnx::ModelProto *ReadOnnxFromBinary(const std::string &model_path);

  FuncGraphPtr GetResult() override;

  int Import(const converter::Flags *flag) override;

 private:
  int ConverterConstTensor() override { return RET_ERROR; };
  int ConverterCNode() override { return RET_ERROR; };
  int AddReturnCNode() override { return RET_ERROR; };
  int ParseModelConfigureInfo(const onnx::ModelProto &model_proto);
  int BuildFuncGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto,
                     const schema::QuantType &quantType);
  int ImportParametersForGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto);
  int ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto,
                          const schema::QuantType &quantType);
  int BuildParameterForFuncGraph(const ParameterPtr &node, const onnx::ValueInfoProto &value_proto);
  CNodePtr BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph, const onnx::NodeProto &node_proto,
                                  const schema::QuantType &quantType);
  bool BuildReturnForFuncGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto,
                               const CNodePtr &cnode_ptr);
  static bool GetAttrValueForCNode(const PrimitivePtr &prim, const onnx::AttributeProto &attr_proto);
  static bool ObtainCNodeAttrInTypeForm(const PrimitivePtr &prim, const std::string &attr_name,
                                        const onnx::TensorProto &attr_tensor);
  static ValuePtr ObtainCNodeAttrInScalarForm(const onnx::TensorProto &attr_tensor);
  static bool ObtainCNodeAttrInTensorForm(const PrimitivePtr &prim, const std::string &attr_name,
                                          const onnx::TensorProto &attr_tensor);
  bool BuildValueNodeForFuncGraph(const onnx::NodeProto &node_proto);
  bool ObtainValueNodeInTensorForm(const string &value_node_name, const onnx::TensorProto &attr_tensor);
  bool GetAttrValueForValueNode(const std::string &value_node_name, const onnx::AttributeProto &attr_proto);
  bool ObtainValueNodeInTypeForm(const string &value_node_name, const onnx::TensorProto &attr_tensor);
  static std::unordered_map<std::string, abstract::AbstractTensorPtr> GetAbstractForCNode(
    const onnx::AttributeProto &attr_proto);

 private:
  std::string producer_name_;
  int model_version_{};
  int ir_version_{};
  std::unordered_map<std::string, AnfNodePtr> anfnode_build_map_;
  std::map<std::string, onnx::TensorProto> default_para_map_;
  onnx::ModelProto *onnx_model_ = nullptr;
  FuncGraphPtr func_graph_ = nullptr;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_TOOLS_COMMON_ANF_IMPORTER_IMPORTER_FROM_PROTOBUF_H_
