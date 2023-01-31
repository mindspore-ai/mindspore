/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <vector>
#include "utils/hash_map.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "ir/func_graph.h"
#include "proto/mind_ir.pb.h"
#include "utils/crypto.h"

namespace mindspore {
using int32 = int32_t;
using int64 = int64_t;
using uint64 = uint64_t;

class Layout {
 public:
  Layout() = default;

  const std::vector<int64_t> &get_device_arrangement() const { return device_arrangement_; }
  void set_device_arrangement(const std::vector<int64_t> &device_arrangement) {
    device_arrangement_ = device_arrangement;
  }
  const std::vector<int64_t> &get_tensor_map() const { return tensor_map_; }
  void set_tensor_map(const std::vector<int64_t> &tensor_map) { tensor_map_ = tensor_map; }
  const std::vector<int64_t> &get_slice_shape() const { return slice_shape_; }
  void set_slice_shape(const std::vector<int64_t> &slice_shape) { slice_shape_ = slice_shape; }
  int64_t get_field_size() const { return field_size_; }
  void set_field_size(int64_t field_size) { field_size_ = field_size; }
  bool get_uniform_split() const { return uniform_split_; }
  void set_uniform_split(bool uniform_split) { uniform_split_ = uniform_split; }
  const std::string &get_opt_shard_group() const { return opt_shard_group_; }
  void set_opt_shard_group(const std::string &opt_shard_group) { opt_shard_group_ = opt_shard_group; }

 private:
  std::vector<int64_t> device_arrangement_{};
  std::vector<int64_t> tensor_map_{};
  std::vector<int64_t> slice_shape_{};
  int64_t field_size_ = 0;
  bool uniform_split_ = false;
  std::string opt_shard_group_ = "";
};
using LayoutPtr = std::shared_ptr<Layout>;
using LayoutMap = std::map<string, LayoutPtr>;

class MSANFModelParser {
 public:
  MSANFModelParser() = default;
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
  void SetMindIRPath(const std::string &file_path) { mindir_path_ = file_path; }
  void SetMindIRDecKey(const unsigned char *dec_key) { mindir_dec_key_ = dec_key; }
  void SetMindIRKeySize(size_t size) { mindir_key_size_ = size; }
  void SetMindIRDecMode(const std::string &dec_mode) { mindir_dec_mode_ = dec_mode; }

 private:
  bool BuildPrimitiveNode(const mind_ir::PrimitiveProto &primitive_proto);
  abstract::AbstractBasePtr BuildAbstractFunction(const mind_ir::AttributeProto &attr_proto);
  void CorrectFuncGraph(const FuncGraphPtr &root);
  bool BuildFuncGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool BuildAttrForFuncGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool BuildAttrForCNode(const CNodePtr &cnode, const mind_ir::NodeProto &node_proto);
  ValuePtr GetValueFromAttributeProto(const mind_ir::AttributeProto &attr_proto);
  bool ImportParametersForGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool ImportMapParametersForGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool BuildParameterForFuncGraph(const ParameterPtr &node, const mind_ir::TensorProto &parameter_proto);
  bool BuildMapParameterFromMapTensorProto(const ParameterPtr &node,
                                           const mind_ir::MapTensorProto &map_parameter_proto);
  abstract::AbstractMapTensorPtr BuildAbstractMapTensorFromAttrProto(const mind_ir::AttributeProto &attr_proto);
  abstract::AbstractCOOTensorPtr BuildAbstractCOOTensorFromAttrProto(const mind_ir::AttributeProto &attr_proto);
  abstract::AbstractCSRTensorPtr BuildAbstractCSRTensorFromAttrProto(const mind_ir::AttributeProto &attr_proto);
  bool SetValueForTopGraphParameter(const FuncGraphPtr &topGraph, const std::map<std::string, ValuePtr> &weights);
  bool GetTensorDataFromExternal(const mind_ir::TensorProto &tensor_proto, const tensor::TensorPtr &tensor_info);
  bool BuildInputForFuncGraph(const ParameterPtr &node, const mind_ir::ValueInfoProto &value_proto);
  abstract::AbstractTensorPtr GetAbsTensorFromTensorProto(const mind_ir::TensorProto &tensor_proto);
  CNodePtr BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::NodeProto &node_proto);
  bool BuildReturnForFuncGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto);
  bool GetAttrValueForCNode(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto);
  bool SetPrimitiveAttrWithType(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto);
  bool ObtainCNodeAttrInTypeForm(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto);
  void ObtainCNodeAttrInScalarForm(const mind_ir::AttributeProto &attr_proto,
                                   mindspore::HashMap<std::string, ValuePtr> *multi_value_map);
  ValuePtr ParseAttrInScalarForm(const mind_ir::AttributeProto &attr_proto, int index);
  ValuePtr ObtainCNodeAttrInSingleScalarForm(const mind_ir::AttributeProto &attr_proto);
  bool ObtainCNodeAttrInTensorForm(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto);
  bool BuildValueNodeForFuncGraph(const mind_ir::NodeProto &node_proto);
  ValuePtr BuildValueFromAttributeProto(const mind_ir::AttributeProto &attr_proto);
  AnfNodePtr BuildOperatorNode(const mind_ir::NodeProto &node_proto);
  bool SetEmptyTensorProtoCNodeAbstract(const AnfNodePtr &node_ptr);
  void SetCNodeAbstract(const mind_ir::AttributeProto &attr_proto, const CNodePtr &cnode_ptr);
  bool SetNodeAbstractFromAttrProto(const mind_ir::AttributeProto &attr_proto, const AnfNodePtr &node_ptr);
  abstract::AbstractBasePtr GetNodeAbstractFromAttrProtoWithType(const mind_ir::AttributeProto &attr_proto);
  void SetCNodePrimAttrAndAbstract(const mind_ir::NodeProto &node_proto, const CNodePtr &cnode_ptr);
  bool ObtainValueNodeInTensorForm(const string &value_node_name, const mind_ir::TensorProto &attr_tensor);
  bool ObtainValueNodeInTupleTensorForm(const string &value_node_name, const mind_ir::AttributeProto &attr_proto);
  bool GetAttrValueForValueNode(const std::string &value_node_name, const mind_ir::AttributeProto &attr_proto);
  bool GetAttrValueForValueNodeWithType(const std::string &value_node_name, const mind_ir::AttributeProto &attr_proto);
  bool ObtainValueNodeInTypeForm(const string &value_node_name, const mind_ir::TensorProto &attr_tensor);
  bool ObtainValueNodeInNoneForm(const std::string &value_node_name);
  bool ObtainValueNodeInMonadForm(const std::string &value_node_name, const mind_ir::AttributeProto &attr_proto);
  ValuePtr ObtainValueInSequenceForm(const mind_ir::AttributeProto &attr_proto);
  ValuePtr ObtainValueInDictionaryForm(const mind_ir::AttributeProto &attr_proto);
  std::vector<std::shared_ptr<mindspore::QuantizationParam>> GenerateQuantizationParam(
    const mind_ir::TensorProto &attr_tensor);
  bool little_endian() const { return little_endian_; }
  mindspore::HashMap<std::string, abstract::AbstractBasePtr> GetAbstractForNode(
    const mind_ir::AttributeProto &attr_proto);
  AnfNodePtr GetAnfNode(const std::string &node_name);
  tensor::TensorPtr GenerateTensorPtrFromTensorProto(const mind_ir::TensorProto &attr_tensor);

  FuncGraphPtr top_graph_ = nullptr;
  std::string producer_name_;
  std::string model_version_;
  std::string ir_version_;
  bool is_lite_ = false;
  bool inc_load_ = false;
  bool abstract_valid_ = false;
  mindspore::HashMap<std::string, AnfNodePtr> anfnode_build_map_;
  std::string mindir_path_;
  const unsigned char *mindir_dec_key_{nullptr};
  size_t mindir_key_size_{0};
  std::string mindir_dec_mode_;
  bool little_endian_ = common::IsLittleByteOrder();
  std::map<std::string, std::unique_ptr<Byte[]>> tenor_data_;
  static std::map<std::string, tensor::TensorPtr> load_tensor_map_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_LOAD_MINDIR_ANF_MODEL_PARSER_H
