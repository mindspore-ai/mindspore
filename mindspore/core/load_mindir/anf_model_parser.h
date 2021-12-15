/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <memory>
#include "utils/hash_map.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "ir/func_graph.h"
#include "proto/mind_ir.pb.h"
#include "utils/crypto.h"
#include "load_mindir/load_model.h"
namespace mindspore {
using int32 = int32_t;
using int64 = int64_t;
using uint64 = uint64_t;
class MSANFModelParser {
 public:
  MSANFModelParser() : producer_name_(""), model_version_(""), ir_version_("") {}
  ~MSANFModelParser() = default;

  static void LoadTensorMapClear() { load_tensor_map_.clear(); }
  FuncGraphPtr Parse(const mind_ir::ModelProto &model_proto, const std::map<std::string, ValuePtr> &weights = {});
  const LayoutMap ParseLayout(const mind_ir::ModelProto &model_proto);
  bool MSANFParseModelConfigureInfo(const mind_ir::ModelProto &model_proto);

  std::string GetProducerName() { return producer_name_; }
  std::string GetProducerVersion() { return model_version_; }
  std::string GetIrVersion() { return ir_version_; }
  void SetLite() { is_lite_ = true; }
  bool IsLite() const { return is_lite_; }
  void SetIncLoad() { inc_load_ = true; }
  bool IsIncLoad() const { return inc_load_; }
  void set_need_renormalize(bool need_renormalize) { need_renormalize_ = need_renormalize; }
  bool need_renormalize() const { return need_renormalize_; }
  void SetMindIRPath(const std::string &file_path) { mindir_path_ = file_path; }
  void SetMindIRDecKey(const unsigned char *dec_key) { mindir_dec_key_ = dec_key; }
  void SetMindIRKeySize(size_t size) { mindir_key_size_ = size; }
  void SetMindIRDecMode(const std::string &dec_mode) { mindir_dec_mode_ = dec_mode; }

 private:
  bool BuildFuncGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool BuildAttrForFuncGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool ImportParametersForGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool BuildParameterForFuncGraph(const ParameterPtr &node, const mind_ir::TensorProto &tensor_proto);
  bool SetValueForTopGraphParameter(const FuncGraphPtr &topGraph, const std::map<std::string, ValuePtr> &weights);
  bool GetTensorDataFromExternal(const mind_ir::TensorProto &tensor_proto, const tensor::TensorPtr &tensor_info);
  bool BuildInputForFuncGraph(const ParameterPtr &node, const mind_ir::ValueInfoProto &value_proto);
  abstract::AbstractTensorPtr GetAbsTensorFromTensorProto(const mind_ir::TensorProto &tensor_proto);
  CNodePtr BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::NodeProto &node_proto);
  bool BuildReturnForFuncGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool GetAttrValueForCNode(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto);
  bool ObtainCNodeAttrInTypeForm(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto);
  void ObtainCNodeAttrInScalarForm(const mind_ir::AttributeProto &attr_proto,
                                   mindspore::HashMap<std::string, ValuePtr> *multi_value_map);
  ValuePtr ParseAttrInScalarForm(const mind_ir::AttributeProto &attr_proto, int index);
  ValuePtr ObtainCNodeAttrInSingleScalarForm(const mind_ir::AttributeProto &attr_proto);
  bool ObtainCNodeAttrInTensorForm(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto);
  bool BuildValueNodeForFuncGraph(const mind_ir::NodeProto &node_proto);
  AnfNodePtr BuildOperatorNode(const mind_ir::NodeProto &node_proto);
  bool CheckCNodePrim(CNodePtr cnode_ptr);
  bool SetEmptyTensorProtoCNodeAbstract(const AnfNodePtr &node_ptr);
  bool SetCNodeAbstract(const mind_ir::AttributeProto &attr_proto, const CNodePtr &cnode_ptr);
  bool SetNodeAbstractFromAttrProto(const mind_ir::AttributeProto &attr_proto, const AnfNodePtr &node_ptr);
  void SetCNodePrimAttrAndAbstract(const mind_ir::NodeProto &node_proto, const CNodePtr &cnode_ptr);
  bool ObtainValueNodeInTensorForm(const string &value_node_name, const mind_ir::TensorProto &attr_tensor);
  bool ObtainValueNodeInTupleTensorForm(const string &value_node_name, const mind_ir::AttributeProto &attr_proto);
  bool GetAttrValueForValueNode(const std::string &value_node_name, const mind_ir::AttributeProto &attr_tensor);
  bool ObtainValueNodeInTypeForm(const string &value_node_name, const mind_ir::TensorProto &attr_tensor);
  bool ObtainValueNodeInNoneForm(const std::string &value_node_name, const mind_ir::AttributeProto &attr_proto);
  bool ObtainValueNodeInMonadForm(const std::string &value_node_name, const mind_ir::AttributeProto &attr_proto);
  bool little_endian() { return little_endian_; }
  mindspore::HashMap<std::string, abstract::AbstractBasePtr> GetAbstractForNode(
    const mind_ir::AttributeProto &attr_proto);
  AnfNodePtr GetAnfNode(const std::string &node_name);
  tensor::TensorPtr GenerateTensorPtrFromTensorProto(const mind_ir::TensorProto &attr_tensor,
                                                     bool need_load_data = true);

  FuncGraphPtr top_graph_ = nullptr;
  std::string producer_name_;
  std::string model_version_;
  std::string ir_version_;
  bool is_lite_ = false;
  bool inc_load_ = false;
  bool need_renormalize_ = true;
  mindspore::HashMap<std::string, AnfNodePtr> anfnode_build_map_;
  std::string mindir_path_;
  const unsigned char *mindir_dec_key_{nullptr};
  size_t mindir_key_size_;
  std::string mindir_dec_mode_;
  bool little_endian_ = common::IsLittleByteOrder();
  std::map<std::string, std::unique_ptr<Byte[]>> tenor_data_;
  static std::map<std::string, tensor::TensorPtr> load_tensor_map_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_LOAD_MINDIR_ANF_MODEL_PARSER_H
