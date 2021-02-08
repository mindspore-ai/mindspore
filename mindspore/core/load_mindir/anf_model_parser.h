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

#ifndef MINDSPORE_CORE_LOAD_MINDIR_ANF_MODEL_PARSER_H
#define MINDSPORE_CORE_LOAD_MINDIR_ANF_MODEL_PARSER_H

#include <string>
#include <map>
#include <unordered_map>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "ir/func_graph.h"
#include "proto/mind_ir.pb.h"

namespace mindspore {
using int32 = int32_t;
using int64 = int64_t;
using uint64 = uint64_t;
class MSANFModelParser {
 public:
  MSANFModelParser() : producer_name_(""), model_version_(""), ir_version_("") {}
  ~MSANFModelParser() = default;

  FuncGraphPtr Parse(const mind_ir::ModelProto &model_proto);
  bool MSANFParseModelConfigureInfo(const mind_ir::ModelProto &model_proto);

  std::string GetProducerName() { return producer_name_; }
  std::string GetProducerVersion() { return model_version_; }
  std::string GetIrVersion() { return ir_version_; }
  void SetLite() { is_lite_ = true; }
  bool IsLite() { return is_lite_; }

 private:
  bool BuildFuncGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool ImportParametersForGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool BuildParameterForFuncGraph(const ParameterPtr &node, const mind_ir::TensorProto &tensor_proto);
  bool BuildInputForFuncGraph(const ParameterPtr &node, const mind_ir::ValueInfoProto &value_proto);
  tensor::TensorPtr BuildTensorInfoForFuncGraph(const mind_ir::TensorProto &tensor_proto);
  CNodePtr BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::NodeProto &node_proto);
  bool BuildReturnForFuncGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto,
                               const CNodePtr &cnode_ptr);
  bool GetAttrValueForCNode(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto);
  bool ObtainCNodeAttrInTypeForm(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto);
  void ObtainCNodeAttrInScalarForm(const mind_ir::AttributeProto &attr_proto,
                                   std::unordered_map<std::string, ValuePtr> *multi_value_map);
  ValuePtr ParseAttrInScalarForm(const mind_ir::AttributeProto &attr_proto, int index);
  ValuePtr ObtainCNodeAttrInSingleScalarForm(const mind_ir::AttributeProto &attr_proto);
  bool ObtainCNodeAttrInTensorForm(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto);
  bool BuildValueNodeForFuncGraph(const mind_ir::NodeProto &node_proto);
  bool ObtainValueNodeInTensorForm(const string &value_node_name, const mind_ir::TensorProto &attr_tensor);
  bool GetAttrValueForValueNode(const std::string &value_node_name, const mind_ir::AttributeProto &attr_tensor);
  bool ObtainValueNodeInTypeForm(const string &value_node_name, const mind_ir::TensorProto &attr_tensor);
  bool ObtainValueNodeInNoneForm(const std::string &value_node_name, const mind_ir::AttributeProto &attr_proto);
  bool ObtainValueNodeInMonadForm(const std::string &value_node_name, const mind_ir::AttributeProto &attr_proto);
  std::unordered_map<std::string, abstract::AbstractBasePtr> GetAbstractForCNode(
    const mind_ir::AttributeProto &attr_proto);

  std::string producer_name_;
  std::string model_version_;
  std::string ir_version_;
  bool is_lite_ = false;
  std::unordered_map<std::string, AnfNodePtr> anfnode_build_map_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_LOAD_MINDIR_ANF_MODEL_PARSER_H
