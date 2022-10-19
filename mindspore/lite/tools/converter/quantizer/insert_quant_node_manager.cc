/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API

#include "mindspore/lite/tools/converter/quantizer/insert_quant_node_manager.h"
#include <memory>
#include <set>
#include <vector>
#include <string>
#include <algorithm>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/common/node_util.h"

namespace mindspore::lite::quant {
namespace {
constexpr size_t kMinSize3 = 3;
constexpr size_t kPrimitiveCOffset = 1;
}  // namespace
int InsertQuantNodeManager::SetCastNodeAbstract(const CNodePtr &cnode, const AnfNodePtr &input_node,
                                                const CNodePtr &cast_cnode) {
  CHECK_NULL_RETURN(cnode);
  CHECK_NULL_RETURN(input_node);
  CHECK_NULL_RETURN(cast_cnode);

  AbstractBasePtr abstract;
  if (cnode->abstract() != nullptr) {
    abstract = cnode->abstract()->Clone();
  } else if (input_node->abstract() != nullptr) {
    abstract = input_node->abstract()->Clone();
  } else {
    MS_LOG(ERROR) << "Abstract is nullptr, cnode name: " << cnode->fullname_with_scope()
                  << " input node: " << input_node->fullname_with_scope();
    return RET_NULL_PTR;
  }
  cast_cnode->set_abstract(abstract);
  return RET_OK;
}

// If dtype can be fetched, check data type, otherwise return RET_OK
int InsertQuantNodeManager::CheckDataType(const AnfNodePtr &input_node, TypeId check_type_id) const {
  bool is_graph_input = IsGraphInput(input_node);
  if (!input_node->isa<mindspore::CNode>() && !is_graph_input) {
    return RET_NO_CHANGE;
  }
  bool is_special_node =
    input_node->isa<mindspore::CNode>() && opt::IsSpecialType(input_node->cast<mindspore::CNodePtr>());
  if (!is_special_node || is_graph_input) {
    TypeId type_id;
    auto ret = opt::GetDataTypeFromAnfNode(input_node, &type_id);
    if (ret != RET_OK) {
      MS_LOG(WARNING) << "Fetch DataType from cnode failed.";
      return RET_OK;
    }
    if (type_id != check_type_id) {
      return RET_NO_CHANGE;
    }
  }
  return RET_OK;
}

int InsertQuantNodeManager::InsertDynamicQuantWithIndex(const FuncGraphPtr &graph, const CNodePtr &cnode,
                                                        size_t index) {
  auto primitive = std::make_shared<ops::DynamicQuant>();
  auto primitive_c = primitive->GetPrim();
  primitive->set_dst_type(dst_type_);
  primitive->set_symmetric(symmetric_);
  auto dynamic_quant_cnode = graph->NewCNode(primitive_c, {cnode->input(index)});
  auto name = cnode->fullname_with_scope() + "_dynamic_cast_node_" + std::to_string(index);
  dynamic_quant_cnode->set_fullname_with_scope(name);
  CHECK_NULL_RETURN(cnode->abstract());
  auto abstract = cnode->abstract()->Clone();
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Abstract of node is nullptr, " << cnode->fullname_with_scope();
    return RET_NULL_PTR;
  }
  dynamic_quant_cnode->set_abstract(abstract);
  auto ret = UpdateDataType(cnode, dst_type_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " set new dtype failed.";
    return ret;
  }
  MarkDynamicQuantize(dynamic_quant_cnode);
  cnode->set_input(index, dynamic_quant_cnode);
  return RET_OK;
}

int InsertQuantNodeManager::NewDynamicQuantNode(const FuncGraphPtr &graph, const CNodePtr &cnode) {
  auto op_name = cnode->fullname_with_scope();
  if (cnode->size() < kMinSize3) {
    MS_LOG(ERROR) << op_name << " cnode size:" << cnode->size() << " < 3.";
    return RET_ERROR;
  }
  auto input = cnode->input(kInputIndex + kPrimitiveCOffset);
  if (input->isa<mindspore::CNode>() || IsGraphInput(input)) {
    auto ret = InsertDynamicQuantWithIndex(graph, cnode, kInputIndex + kPrimitiveCOffset);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert dynamic quant with index failed.";
    }
  }
  auto weight = cnode->input(kWeightIndex + kPrimitiveCOffset);
  if (weight->isa<mindspore::CNode>() || IsGraphInput(weight)) {
    auto ret = InsertDynamicQuantWithIndex(graph, cnode, kWeightIndex + kPrimitiveCOffset);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert dynamic quant with index failed.";
    }
  }
  return RET_OK;
}

int InsertQuantNodeManager::MarkDynamicQuantize(const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_NULL_PTR);
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return RET_ERROR;
  }
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  quant_param_holder->set_quant_type(schema::QuantType_QUANT_DYNAMIC);
  return RET_OK;
}

int InsertQuantNodeManager::InsertDynamicQuantNode(const FuncGraphPtr &graph,
                                                   const std::set<PrimitivePtr> &support_dynamic_quant_ops,
                                                   const std::set<std::string> &skip_quant_node) {
  MS_ASSERT(graph != nullptr);
  auto cnodes = graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto op_name = cnode->fullname_with_scope();
    if (skip_quant_node.find(op_name) != skip_quant_node.end()) {
      MS_LOG(INFO) << op_name << " is skip dynamic quant.";
      continue;
    }
    auto ret = CheckDataType(cnode, kNumberTypeFloat32);
    if (ret == RET_NO_CHANGE) {
      continue;
    }
    if (opt::IsSpecialType(cnode)) {
      continue;
    }
    auto is_support_node = CheckNodeInSet(cnode, support_dynamic_quant_ops);
    if (!is_support_node) {
      auto type = NodePrimitiveType(cnode);
      MS_LOG(INFO) << "node:" << op_name << " type:" << type << " will not quantify.";
      continue;
    }
    ret = NewDynamicQuantNode(graph, cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "node:" << op_name << " new dynamic quant node failed.";
      return ret;
    }
    ret = MarkDynamicQuantize(cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "node:" << op_name << " new mark dynamic quant node failed.";
      return ret;
    }
    ret = UpdateDataType(cnode, kNumberTypeFloat32);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "node:" << op_name << " update datatype failed.";
      return ret;
    }
  }
  return RET_OK;
}

int InsertQuantNodeManager::InsertFP32DtypeCastNode(const FuncGraphPtr &graph) {
  CHECK_NULL_RETURN(graph);
  auto cnodes = graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    schema::QuantType curr_quant_type;
    if (GetQuantType(cnode, &curr_quant_type) != RET_OK) {
      MS_LOG(ERROR) << "Get quant type failed, cnode name: " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (curr_quant_type != schema::QuantType_QUANT_ALL) {
      MS_LOG(INFO) << "Invalid cnode quant type, cnode name: " << cnode->fullname_with_scope()
                   << " quant type: " << curr_quant_type;
      continue;
    }
    auto status = InsertForwardCastNode(graph, cnode, kNumberTypeFloat32, curr_quant_type);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "InsertForwardCastNode failed, cnode name: " << cnode->fullname_with_scope();
      return status;
    }
    // DetectionPostProcess op(Uint8toFp32, not need backward cast node)
    if (!CheckNodeInSet(cnode, kUint8toFP32Operator)) {
      status = InsertBackwardCastNode(graph, cnode, kNumberTypeFloat32, curr_quant_type);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertBackwardCastNode failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
    }
  }  // for
  return RET_OK;
}

int InsertQuantNodeManager::InserQuantCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode,
                                               InsertDirection insert_direction, TypeId cast_dtype,
                                               CastNodeType cast_node_type, size_t index,
                                               const AnfNodePtr &output_node) {
  if (insert_direction == FORWARD) {
    return InsertForwardQuantCastNode(graph, cnode, cast_dtype, index, cast_node_type);
  } else if (insert_direction == BACKWARD && cast_node_type == kDeQuant) {
    return InsertBackwardDeQuantCastNode(graph, cnode, cast_dtype, index, output_node);
  }
  MS_LOG(ERROR) << "Invalid insert direction: " << insert_direction;
  return RET_NOT_SUPPORT;
}

int InsertQuantNodeManager::InsertForwardQuantCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode,
                                                       TypeId cast_dtype, size_t index, CastNodeType cast_node_type) {
  if (cast_dtype != kNumberTypeUInt8 && cast_dtype != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Invalid cast dtype: " << cast_dtype;
    return RET_NOT_SUPPORT;
  }

  auto input_node = cnode->input(index);
  CHECK_NULL_RETURN(input_node);
  if (!input_node->isa<mindspore::CNode>() && !IsGraphInput(input_node)) {
    MS_LOG(ERROR) << "Invalid input node, input node name: " << input_node->fullname_with_scope();
    return RET_ERROR;
  }
  if (CheckDataType(input_node, cast_dtype) != RET_OK) {
    return RET_NO_CHANGE;
  }
  // insert forward cast_node
  TypeId src_dtype;
  TypeId dst_dtype;
  std::vector<schema::QuantParamT> input_quant_params;
  std::vector<schema::QuantParamT> output_quant_params;
  if (cast_node_type == kQuant) {
    src_dtype = cast_dtype;
    dst_dtype = kNumberTypeInt8;
    auto curr_primitive_quant_param_holder = GetCNodeQuantHolder(cnode);
    CHECK_NULL_RETURN(curr_primitive_quant_param_holder);
    if (curr_primitive_quant_param_holder->get_input_quant_params().size() < index) {
      MS_LOG(ERROR) << "quant param is invalid.";
      return RET_ERROR;
    }
    output_quant_params = curr_primitive_quant_param_holder->get_input_quant_params()[index - 1];
    std::copy(output_quant_params.cbegin(), output_quant_params.cend(), std::back_inserter(input_quant_params));
    // Uint8toInt8
    if (src_dtype == kNumberTypeUInt8) {
      for (auto &quant_param : input_quant_params) {
        quant_param.zeroPoint += kU8ZeroPointOffset;
      }
    }
  } else {
    src_dtype = kNumberTypeInt8;
    dst_dtype = cast_dtype;
    auto input_cnode = std::dynamic_pointer_cast<mindspore::CNode>(input_node);
    auto input_cnode_primitive_c = GetValueNode<std::shared_ptr<mindspore::Primitive>>(input_cnode->input(0));
    if (input_cnode_primitive_c == nullptr) {
      MS_LOG(DEBUG) << "input: " << index << " " << input_cnode->fullname_with_scope() << ": "
                    << " PrimitiveC is null";
      return RET_NO_CHANGE;
    }
    auto input_primitive_quant_param_holder = GetCNodeQuantHolder(input_cnode_primitive_c);
    if (input_primitive_quant_param_holder->get_output_quant_params().empty()) {
      MS_LOG(ERROR) << "output quant param is empty.";
      return RET_ERROR;
    }
    input_quant_params = input_primitive_quant_param_holder->get_output_quant_params()[0];
    std::copy(input_quant_params.cbegin(), input_quant_params.cend(), std::back_inserter(output_quant_params));
  }
  ValueNodePtr new_primitive = NewQuantCastPrimitive(src_dtype, dst_dtype, input_quant_params, output_quant_params);
  std::vector<AnfNodePtr> op_inputs = {new_primitive, input_node};
  auto quant_cast_cnode = graph->NewCNode(op_inputs);
  MS_CHECK_TRUE_MSG(quant_cast_cnode != nullptr, RET_NULL_PTR, "quant_cast_cnode is nullptr.");
  quant_cast_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_dtype_cast_" + std::to_string(index) +
                                            "_pre");
  // set abstract
  if (input_node->abstract() != nullptr) {
    auto abstract = input_node->abstract()->Clone();
    quant_cast_cnode->set_abstract(abstract);
    if (quant::UpdateDataType(quant_cast_cnode, dst_dtype) != RET_OK) {
      MS_LOG(ERROR) << "UpdateDataType failed, cnode name: " << quant_cast_cnode->fullname_with_scope();
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "input node abstract nullptr, input node name: " << input_node->fullname_with_scope();
  }
  auto manager = graph->manager();
  if (manager == nullptr) {
    manager = Manage(graph, true);
  }
  MS_CHECK_TRUE_RET(manager != nullptr, RET_NULL_PTR);
  manager->SetEdge(cnode, index, quant_cast_cnode);
  MS_LOG(INFO) << "InsertForwardQuantCastNode cnode name: " << cnode->fullname_with_scope()
               << " src dtype:" << src_dtype << " dst_type: " << dst_dtype;
  return RET_OK;
}

int InsertQuantNodeManager::InsertBackwardDeQuantCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode,
                                                          TypeId cast_dtype, size_t index,
                                                          const AnfNodePtr &output_node) {
  if (cast_dtype != kNumberTypeUInt8 && cast_dtype != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Invalid cast dtype: " << cast_dtype;
    return RET_NOT_SUPPORT;
  }
  CHECK_NULL_RETURN(output_node);
  // If cnode or outputnode is QuantDTypeCast, do nothing.
  if (opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast) ||
      opt::CheckPrimitiveType(output_node, prim::kPrimQuantDTypeCast)) {
    return RET_NO_CHANGE;
  }
  auto ret = CheckDataType(output_node, cast_dtype);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check data type failed, cnode name: " << output_node->fullname_with_scope();
    return ret;
  }
  auto manager = graph->manager();
  if (manager == nullptr) {
    manager = Manage(graph, true);
  }
  CHECK_NULL_RETURN(manager);

  // insert backward cast_node
  TypeId src_dtype = kNumberTypeInt8;
  TypeId dst_dtype = cast_dtype;
  std::vector<schema::QuantParamT> input_quant_params;
  std::vector<schema::QuantParamT> output_quant_params;

  auto curr_primitive_quant_param_holder = GetCNodeQuantHolder(cnode);
  CHECK_NULL_RETURN(curr_primitive_quant_param_holder);
  if (curr_primitive_quant_param_holder->get_output_quant_params().empty()) {
    MS_LOG(ERROR) << "quant param is invalid.";
    return RET_ERROR;
  }
  input_quant_params = curr_primitive_quant_param_holder->get_output_quant_params().front();
  std::copy(input_quant_params.cbegin(), input_quant_params.cend(), std::back_inserter(output_quant_params));
  // Int8toUint8
  if (dst_dtype == kNumberTypeUInt8) {
    for (auto &quant_param : output_quant_params) {
      quant_param.zeroPoint += kU8ZeroPointOffset;
    }
  }
  ValueNodePtr new_primitive = NewQuantCastPrimitive(src_dtype, dst_dtype, input_quant_params, output_quant_params);
  std::vector<AnfNodePtr> op_inputs = {new_primitive, cnode->cast<AnfNodePtr>()};
  auto quant_cast_cnode = graph->NewCNode(op_inputs);
  MS_CHECK_TRUE_MSG(quant_cast_cnode != nullptr, RET_NULL_PTR, "quant_cast_cnode is nullptr.");
  quant_cast_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_dtype_cast_" + std::to_string(index) +
                                            "_post");
  if (SetCastNodeAbstract(cnode, output_node, quant_cast_cnode) != RET_OK) {
    MS_LOG(ERROR) << "SetCastNodeAbstract failed.";
    return RET_ERROR;
  }
  if (quant::UpdateDataType(quant_cast_cnode, dst_dtype) != RET_OK) {
    MS_LOG(ERROR) << "UpdateDataType failed, cnode name: " << quant_cast_cnode->fullname_with_scope();
    return RET_ERROR;
  }
  manager->SetEdge(output_node, index, quant_cast_cnode);
  MS_LOG(INFO) << "InsertBackwardDeQuantCastNode cnode name: " << cnode->fullname_with_scope()
               << " src dtype:" << src_dtype << " dst_type: " << dst_dtype;
  return RET_OK;
}

int InsertQuantNodeManager::InsertForwardCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype,
                                                  schema::QuantType curr_quant_type) {
  // inputs
  for (size_t index = 1; index < cnode->size(); index++) {
    auto input_node = cnode->input(index);
    CHECK_NULL_RETURN(input_node);
    if (!input_node->isa<mindspore::CNode>() && !IsGraphInput(input_node)) {
      MS_LOG(DEBUG) << "Invalid input node, not CNode and graph input.";
      continue;
    }
    schema::QuantType pre_quant_type = schema::QuantType_QUANT_NONE;
    if (input_node->isa<mindspore::CNode>()) {
      if (GetQuantType(input_node->cast<mindspore::CNodePtr>(), &pre_quant_type) != RET_OK) {
        MS_LOG(ERROR) << "Get quant type failed, cnode name: " << cnode->fullname_with_scope();
        return RET_ERROR;
      }
    }
    if (pre_quant_type == schema::QuantType_QUANT_NONE && curr_quant_type == schema::QuantType_QUANT_ALL) {
      auto status = InserQuantCastNode(graph, cnode, FORWARD, cast_dtype, kQuant, index, nullptr);
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "InserQuantCastNode kQuant failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
    }
  }
  return RET_OK;
}

int InsertQuantNodeManager::InsertCastNodeForFullQuant(const FuncGraphPtr &graph, const CNodePtr &cnode,
                                                       TypeId cast_dtype, schema::QuantType curr_quant_type) {
  // inputs
  for (size_t index = 1; index < cnode->size(); index++) {
    auto input_node = cnode->input(index);
    CHECK_NULL_RETURN(input_node);
    if (!input_node->isa<mindspore::CNode>() && !IsGraphInput(input_node)) {
      MS_LOG(DEBUG) << "Invalid input node, not CNode and graph input.";
      continue;
    }
    schema::QuantType pre_quant_type = schema::QuantType_QUANT_NONE;
    if (input_node->isa<mindspore::CNode>()) {
      if (GetQuantType(input_node->cast<mindspore::CNodePtr>(), &pre_quant_type) != RET_OK) {
        MS_LOG(ERROR) << "Get quant type failed, cnode name: " << cnode->fullname_with_scope();
        return RET_ERROR;
      }
    }
    if (pre_quant_type == schema::QuantType_QUANT_NONE && curr_quant_type == schema::QuantType_QUANT_ALL) {
      auto status = InserQuantCastNode(graph, cnode, FORWARD, cast_dtype, kQuant, index, nullptr);
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "InserQuantCastNode kQuant failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
    } else if (pre_quant_type == schema::QuantType_QUANT_ALL && curr_quant_type == schema::QuantType_QUANT_NONE) {
      auto status = InserQuantCastNode(graph, cnode, FORWARD, cast_dtype, kDeQuant, index, nullptr);
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "InserQuantCastNode kDeQuant failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
    }
  }
  return RET_OK;
}

int InsertQuantNodeManager::InsertBackwardCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype,
                                                   schema::QuantType curr_quant_type) {
  // outputs
  auto manager = graph->manager();
  if (manager == nullptr) {
    manager = Manage(graph, true);
  }
  CHECK_NULL_RETURN(manager);
  auto node_users = manager->node_users()[cnode];
  for (auto &node_user : node_users) {
    auto output_cnode = node_user.first->cast<CNodePtr>();
    schema::QuantType post_quant_type;
    if (GetQuantType(output_cnode, &post_quant_type) != RET_OK) {
      MS_LOG(ERROR) << "Get quant type failed, cnode name: " << output_cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (curr_quant_type == schema::QuantType_QUANT_ALL && post_quant_type == schema::QuantType_QUANT_NONE) {
      auto status = InserQuantCastNode(graph, cnode, BACKWARD, cast_dtype, kDeQuant, node_user.second, node_user.first);
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "InserQuantCastNode dequant failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
    }
  }  // node_users
  return RET_OK;
}
}  // namespace mindspore::lite::quant
