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

#ifndef MINDSPORE_LITE_SRC_ANF_IMPORTER_IMPORTER_FROM_PROTOBUF_H_
#define MINDSPORE_LITE_SRC_ANF_IMPORTER_IMPORTER_FROM_PROTOBUF_H_

#include <map>
#include <string>
#include <unordered_map>
#include <utility>

#include "abstract/abstract_value.h"
#include "src/common/anf_importer/anf_importer.h"
#include "tools/converter/parser/onnx/onnx.pb.h"

namespace mindspore::lite {
class AnfImporterFromProtobuf : public AnfImporter {
 public:
  explicit AnfImporterFromProtobuf(onnx::ModelProto *onnx_model,
                                   FuncGraphPtr func_graph)
      : onnx_model_(onnx_model), func_graph_(std::move(func_graph)) {}

  ~AnfImporterFromProtobuf() override = default;

  static onnx::ModelProto *ReadOnnxFromBinary(const std::string &model_path);

  FuncGraphPtr GetResult() override;

  int Import(const schema::QuantType &quantType =
                 schema::QuantType_QUANT_NONE) override;

 private:
  void ConverterConstTensor() override{};
  int ConverterCNode() override{};
  void AddReturnCNode() override{};
  bool ParseModelConfigureInfo(const onnx::ModelProto &model_proto);
  bool BuildFuncGraph(const FuncGraphPtr &outputFuncGraph,
                      const onnx::GraphProto &importProto,
                      const schema::QuantType &quantType);
#if 0
  bool ImportParametersForGraph(const FuncGraphPtr &outputFuncGraph,
                                const onnx::GraphProto &importProto);
  bool ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph,
                           const onnx::GraphProto &importProto);
  bool BuildParameterForFuncGraph(const ParameterPtr &node,
                                  const onnx::ValueInfoProto &value_proto);
  CNodePtr BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                                  const onnx::NodeProto &node_proto);
  bool BuildReturnForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                               const onnx::GraphProto &importProto,
                               const CNodePtr &cnode_ptr);
  bool GetAttrValueForCNode(const PrimitivePtr &prim,
                            const onnx::AttributeProto &attr_proto);
  bool ObtainCNodeAttrInTypeForm(const PrimitivePtr &prim,
                                 const std::string &attr_name,
                                 const onnx::TensorProto &attr_tensor);
  ValuePtr ObtainCNodeAttrInScalarForm(const onnx::TensorProto &attr_tensor);
  bool ObtainCNodeAttrInTensorForm(const PrimitivePtr &prim,
                                   const std::string &attr_name,
                                   const onnx::TensorProto &attr_tensor);
  bool BuildValueNodeForFuncGraph(const onnx::NodeProto &node_proto);
  bool ObtainValueNodeInTensorForm(const std::string &value_node_name,
                                   const onnx::TensorProto &attr_tensor);
  bool GetAttrValueForValueNode(const std::string &value_node_name,
                                const onnx::AttributeProto &attr_tensor);
  bool ObtainValueNodeInTypeForm(const std::string &value_node_name,
                                 const onnx::TensorProto &attr_tensor);
  std::unordered_map<std::string, abstract::AbstractTensorPtr>
          GetAbstractForCNode(const onnx::AttributeProto &attr_proto);
#else
  bool ImportParametersForGraph(const FuncGraphPtr &outputFuncGraph,
                                const onnx::GraphProto &importProto);
  bool ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph,
                           const onnx::GraphProto &importProto,
                           const schema::QuantType &quantType);
  bool BuildParameterForFuncGraph(const ParameterPtr &node,
                                  const onnx::ValueInfoProto &value_proto);
  CNodePtr BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                                  const onnx::NodeProto &node_proto,
                                  const schema::QuantType &quantType);
  bool BuildReturnForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                               const onnx::GraphProto &importProto,
                               const CNodePtr &cnode_ptr);
  bool GetAttrValueForCNode(const PrimitivePtr &prim,
                            const onnx::AttributeProto &attr_proto);
  bool ObtainCNodeAttrInTypeForm(const PrimitivePtr &prim,
                                 const std::string &attr_name,
                                 const onnx::TensorProto &attr_tensor);
  bool ObtainCNodeAttrInScalarForm(const PrimitivePtr &prim,
                                   const std::string &attr_name,
                                   const onnx::TensorProto &attr_tensor);
  bool ObtainCNodeAttrInTensorForm(const PrimitivePtr &prim,
                                   const std::string &attr_name,
                                   const onnx::TensorProto &attr_tensor);
  bool BuildValueNodeForFuncGraph(const onnx::NodeProto &node_proto);
  bool ObtainValueNodeInTensorForm(const string &value_node_name,
                                   const onnx::TensorProto &attr_tensor);

  bool ObtainValueNodeInScalarForm(const string &value_node_name,
                                   const onnx::TensorProto &attr_tensor);
  bool GetAttrValueForValueNode(const string &ref_attr_name,
                                const std::string &value_node_name,
                                const onnx::TensorProto &attr_tensor);
  bool ObtainValueNodeInTypeForm(const string &value_node_name,
                                 const onnx::TensorProto &attr_tensor);
  abstract::AbstractTensorPtr GetAbstractForCNode(
      const onnx::AttributeProto &attr_proto);

#endif

 private:
  std::string producer_name_;
  int model_version_{};
  int ir_version_{};
  std::unordered_map<std::string, AnfNodePtr> anfnode_build_map_;
  std::map<std::string, onnx::TensorProto> default_para_map_;
  onnx::ModelProto *onnx_model_;
  FuncGraphPtr func_graph_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_ANF_IMPORTER_IMPORTER_FROM_PROTOBUF_H_
