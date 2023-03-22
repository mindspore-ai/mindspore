/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/node_infershape.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/quantizer/fse_decoder.h"
#include "ops/fse_decode.h"
#include "ops/op_name.h"
#include "ir/dtype.h"

namespace mindspore::lite::quant {
namespace {
constexpr size_t kMinSize2 = 2;
constexpr size_t kMinSize3 = 3;
constexpr size_t kTableExtend = 3;
constexpr size_t kAlignOffset = 7;
constexpr size_t kInt32Mask = 31;
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
  abstract->set_shape(cnode->input(index)->Shape());
  auto ret = UpdateDataType(dynamic_quant_cnode, dst_type_);
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
  auto input = cnode->input(kInputIndex + kPrimOffset);
  if (input->isa<mindspore::CNode>() || IsGraphInput(input)) {
    auto ret = InsertDynamicQuantWithIndex(graph, cnode, kInputIndex + kPrimOffset);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert dynamic quant with index failed.";
    }
  }
  auto weight = cnode->input(kWeightIndex + kPrimOffset);
  if (weight->isa<mindspore::CNode>() || IsGraphInput(weight)) {
    auto ret = InsertDynamicQuantWithIndex(graph, cnode, kWeightIndex + kPrimOffset);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert dynamic quant with index failed.";
    }
  }
  return RET_OK;
}

int InsertQuantNodeManager::MarkDynamicQuantize(const CNodePtr &cnode) {
  CHECK_NULL_RETURN(cnode);
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(primitive);
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  quant_param_holder->set_quant_type(quant::QUANT_DYNAMIC);
  return RET_OK;
}

int InsertQuantNodeManager::InsertDynamicQuantNode(const FuncGraphPtr &graph,
                                                   const std::set<PrimitivePtr> &support_dynamic_quant_ops,
                                                   const std::set<std::string> &skip_quant_node) {
  CHECK_NULL_RETURN(graph);
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

int InsertQuantNodeManager::InsertDequantNode(const FuncGraphPtr &graph) {
  CHECK_NULL_RETURN(graph);
  auto cnodes = graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    quant::QuantType curr_quant_type;
    if (GetQuantType(cnode, &curr_quant_type) != RET_OK) {
      MS_LOG(ERROR) << "Get quant type failed, cnode name: " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (curr_quant_type != quant::QUANT_ALL) {
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

int InsertQuantNodeManager::InsertQuantDtypeCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode,
                                                     InsertDirection insert_direction, TypeId cast_dtype,
                                                     CastNodeType cast_node_type, size_t index,
                                                     const AnfNodePtr &output_node) {
  CHECK_NULL_RETURN(graph);
  CHECK_NULL_RETURN(cnode);
  if (insert_direction == FORWARD) {
    return InsertForwardQuantNode(graph, cnode, cast_dtype, index, cast_node_type);
  } else if (insert_direction == BACKWARD && cast_node_type == kDeQuant) {
    return InsertBackwardDeQuantNode(graph, cnode, cast_dtype, index, output_node);
  }
  MS_LOG(ERROR) << "Invalid insert direction: " << insert_direction;
  return RET_NOT_SUPPORT;
}

int InsertQuantNodeManager::InsertForwardQuantNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype,
                                                   size_t index, CastNodeType cast_node_type) {
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
  ValueNodePtr new_primitive =
    NewQuantCastPrimitive(src_dtype, dst_dtype, input_quant_params, output_quant_params, 0, false);
  std::vector<AnfNodePtr> op_inputs = {new_primitive, input_node};
  auto quant_cast_cnode = graph->NewCNode(op_inputs);
  CHECK_NULL_RETURN(quant_cast_cnode);
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
    MS_LOG(INFO) << "input node abstract nullptr, input node name: " << input_node->fullname_with_scope();
  }
  auto manager = graph->manager();
  if (manager == nullptr) {
    manager = Manage(graph, true);
  }
  CHECK_NULL_RETURN(manager);
  manager->SetEdge(cnode, index, quant_cast_cnode);
  MS_LOG(INFO) << "InsertForwardQuantNode cnode name: " << cnode->fullname_with_scope() << " src dtype:" << src_dtype
               << " dst_type: " << dst_dtype;
  return RET_OK;
}

int InsertQuantNodeManager::InsertBackwardDeQuantNode(const FuncGraphPtr &graph, const CNodePtr &cnode,
                                                      TypeId cast_dtype, size_t index, const AnfNodePtr &output_node) {
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
  ValueNodePtr new_primitive =
    NewQuantCastPrimitive(src_dtype, dst_dtype, input_quant_params, output_quant_params, 0, false);
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
  MS_LOG(INFO) << "InsertBackwardDeQuantNode cnode name: " << cnode->fullname_with_scope() << " src dtype:" << src_dtype
               << " dst_type: " << dst_dtype;
  return RET_OK;
}

int InsertQuantNodeManager::InsertForwardCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype,
                                                  quant::QuantType curr_quant_type) {
  // inputs
  for (size_t index = 1; index < cnode->size(); index++) {
    auto input_node = cnode->input(index);
    CHECK_NULL_RETURN(input_node);
    if (!input_node->isa<mindspore::CNode>() && !IsGraphInput(input_node)) {
      MS_LOG(DEBUG) << "Invalid input node, not CNode and graph input.";
      continue;
    }
    quant::QuantType pre_quant_type = quant::QUANT_NONE;
    if (input_node->isa<mindspore::CNode>()) {
      if (GetQuantType(input_node->cast<mindspore::CNodePtr>(), &pre_quant_type) != RET_OK) {
        MS_LOG(ERROR) << "Get quant type failed, cnode name: " << cnode->fullname_with_scope();
        return RET_ERROR;
      }
    }
    if (pre_quant_type == quant::QUANT_NONE && curr_quant_type == quant::QUANT_ALL) {
      auto status = InsertQuantDtypeCastNode(graph, cnode, FORWARD, cast_dtype, kQuant, index, nullptr);
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "InsertQuantDtypeCastNode kQuant failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
    }
  }
  return RET_OK;
}

int InsertQuantNodeManager::InsertCastNodeForFullQuant(const FuncGraphPtr &graph, const CNodePtr &cnode,
                                                       TypeId cast_dtype, quant::QuantType curr_quant_type) {
  // inputs
  for (size_t index = 1; index < cnode->size(); index++) {
    auto input_node = cnode->input(index);
    CHECK_NULL_RETURN(input_node);
    if (!input_node->isa<mindspore::CNode>() && !IsGraphInput(input_node)) {
      MS_LOG(DEBUG) << "Invalid input node, not CNode and graph input.";
      continue;
    }
    quant::QuantType pre_quant_type = quant::QUANT_NONE;
    if (input_node->isa<mindspore::CNode>()) {
      if (GetQuantType(input_node->cast<mindspore::CNodePtr>(), &pre_quant_type) != RET_OK) {
        MS_LOG(ERROR) << "Get quant type failed, cnode name: " << cnode->fullname_with_scope();
        return RET_ERROR;
      }
    }
    if (pre_quant_type == quant::QUANT_NONE && curr_quant_type == quant::QUANT_ALL) {
      auto status = InsertQuantDtypeCastNode(graph, cnode, FORWARD, cast_dtype, kQuant, index, nullptr);
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "InsertQuantDtypeCastNode kQuant failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
    } else if (pre_quant_type == quant::QUANT_ALL && curr_quant_type == quant::QUANT_NONE) {
      auto status = InsertQuantDtypeCastNode(graph, cnode, FORWARD, cast_dtype, kDeQuant, index, nullptr);
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "InsertQuantDtypeCastNode kDeQuant failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
    }
  }
  return RET_OK;
}

int InsertQuantNodeManager::InsertBackwardCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype,
                                                   quant::QuantType curr_quant_type) {
  // outputs
  auto manager = graph->manager();
  if (manager == nullptr) {
    manager = Manage(graph, true);
  }
  CHECK_NULL_RETURN(manager);
  auto node_users = manager->node_users()[cnode];
  for (auto &node_user : node_users) {
    auto output_cnode = node_user.first->cast<CNodePtr>();
    quant::QuantType post_quant_type;
    if (GetQuantType(output_cnode, &post_quant_type) != RET_OK) {
      MS_LOG(ERROR) << "Get quant type failed, cnode name: " << output_cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (curr_quant_type == quant::QUANT_ALL && post_quant_type == quant::QUANT_NONE) {
      auto status =
        InsertQuantDtypeCastNode(graph, cnode, BACKWARD, cast_dtype, kDeQuant, node_user.second, node_user.first);
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "InsertQuantDtypeCastNode dequant failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
    }
  }  // node_users
  return RET_OK;
}
int InsertQuantNodeManager::InsertQuantDtypeCastFlyNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                        size_t input_index, TypeId src_dtype, TypeId dst_dtype,
                                                        int axis) {
  auto primitive = GetValueNode<std::shared_ptr<mindspore::Primitive>>(cnode->input(kPrimIndex));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto input_node = cnode->input(input_index);
  if (!input_node->isa<mindspore::Parameter>()) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " input " << input_index << " is not parameter node.";
    return RET_ERROR;
  }

  auto curr_primitive_quant_param_holder = GetCNodeQuantHolder(primitive);
  if (curr_primitive_quant_param_holder == nullptr ||
      curr_primitive_quant_param_holder->get_input_quant_params().size() < input_index) {
    MS_LOG(ERROR) << input_node->fullname_with_scope() << " quant param is invalid.";
    return RET_ERROR;
  }
  auto input_quant_params = curr_primitive_quant_param_holder->get_input_quant_params().at(input_index - kPrimOffset);

  ValueNodePtr new_primitive = NewQuantCastPrimitive(src_dtype, dst_dtype, input_quant_params, {}, axis, false);
  std::vector<float> scales;
  std::vector<int> zps;
  std::vector<float> mean_corrs;
  std::vector<float> var_corrs;
  for (size_t i = 0; i < input_quant_params.size(); ++i) {
    scales.push_back(static_cast<float>(input_quant_params.at(i).scale));
    zps.push_back(static_cast<int64_t>(input_quant_params.at(i).zeroPoint));
    mean_corrs.push_back(static_cast<float>(input_quant_params.at(i).meanCorr));
    var_corrs.push_back(static_cast<float>(input_quant_params.at(i).varCorr));
  }
  auto scales_node = opt::BuildFloatVecParameterNode(func_graph, scales, "scales");
  auto zps_node = opt::BuildIntVecParameterNode(func_graph, zps, "zps");
  auto mean_corrs_node = opt::BuildFloatVecParameterNode(func_graph, mean_corrs, "mean_corrs");
  auto var_corrs_node = opt::BuildFloatVecParameterNode(func_graph, var_corrs, "var_corrs");

  std::vector<AnfNodePtr> op_inputs = {new_primitive, input_node,      scales_node,
                                       zps_node,      mean_corrs_node, var_corrs_node};
  auto quant_cast_cnode = func_graph->NewCNode(op_inputs);
  CHECK_NULL_RETURN(quant_cast_cnode);
  auto strings = SplitStringToVector(cnode->fullname_with_scope(), "-op");
  int index = 0;
  if (!ConvertIntNum(strings.at(strings.size() - 1), &index)) {
    index = 0;
  }
  const int quant_dtype_cast_offset = 10000;
  quant_cast_cnode->set_fullname_with_scope(strings.at(0) + "-QuantDtypeCast-op" +
                                            std::to_string(index + quant_dtype_cast_offset));
  opt::NodeInferShape infer;
  auto status = infer.InferShape(quant_cast_cnode);
  if (status != RET_OK) {
    MS_LOG(ERROR) << quant_cast_cnode->fullname_with_scope() << " InferShape failed.";
    return RET_ERROR;
  }
  auto manager = func_graph->manager();
  CHECK_NULL_RETURN(manager);
  auto ret = manager->Replace(input_node, quant_cast_cnode);
  if (!ret) {
    MS_LOG(ERROR) << "Replace QuantDtypeCast failed.";
    return RET_ERROR;
  }
  curr_primitive_quant_param_holder->ClearQuantParams();
  MS_LOG(INFO) << "InsertCastNode cnode name: " << quant_cast_cnode->fullname_with_scope()
               << " src_dtype: " << src_dtype << " dst_dtype: " << dst_dtype;

  return RET_OK;
}

int InsertQuantNodeManager::InsertFSEDecodeNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                size_t input_index, TypeId dst_dtype) {
  auto primitive = GetValueNode<std::shared_ptr<mindspore::Primitive>>(cnode->input(kPrimIndex));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto input_node = cnode->input(input_index);
  if (!input_node->isa<mindspore::Parameter>()) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " input " << input_index << " is not parameter node.";
    return RET_ERROR;
  }
  auto shape = input_node->Shape();
  std::vector<AnfNodePtr> op_inputs;
  int ret = CreateFSEInputs(func_graph, input_node, &op_inputs, dst_dtype);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CreateFSEInputs failed.";
    return RET_ERROR;
  }

  auto fse_decode_cnode = func_graph->NewCNode(op_inputs);
  CHECK_NULL_RETURN(fse_decode_cnode);
  auto strings = SplitStringToVector(cnode->fullname_with_scope(), "-op");
  int index = 0;
  if (!ConvertIntNum(strings.at(strings.size() - 1), &index)) {
    index = 0;
  }
  const int fse_decode_offset = 20000;
  fse_decode_cnode->set_fullname_with_scope(strings.at(0) + "-FSEDecode-op" +
                                            std::to_string(index + fse_decode_offset));
  CHECK_NULL_RETURN(cnode->abstract());
  auto fse_abstract = cnode->abstract()->Clone();
  fse_abstract->set_shape(shape);
  fse_decode_cnode->set_abstract(fse_abstract);

  auto manager = func_graph->manager();
  CHECK_NULL_RETURN(manager);
  ret = manager->Replace(input_node, fse_decode_cnode);
  if (!ret) {
    MS_LOG(ERROR) << "Replace QuantDtypeCast failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

int InsertQuantNodeManager::CreateFSEInputs(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                                            std::vector<AnfNodePtr> *op_inputs, TypeId dst_dtype) {
  if (!input_node->isa<mindspore::Parameter>()) {
    MS_LOG(ERROR) << "FSEDecode input is not parameter node.";
    return RET_ERROR;
  }
  auto parameter_ptr = input_node->cast<ParameterPtr>();
  CHECK_NULL_RETURN(parameter_ptr);
  if (!parameter_ptr->has_default()) {
    MS_LOG(ERROR) << input_node->fullname_with_scope() << " parameter dont have default.";
    return RET_ERROR;
  }
  auto tensor = parameter_ptr->default_param()->cast<tensor::TensorPtr>();
  int8_t *data8 = reinterpret_cast<int8_t *>(tensor->data_c());
  size_t data_size = tensor->DataSize();
  FSEBuffer fse_buffer;
  auto ret = FSEDecoder::DecodeBuffer(data8, data_size, &fse_buffer);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << input_node->fullname_with_scope() << " buffer decode failed.";
    return RET_ERROR;
  }
  ValueNodePtr new_primitive = NewFSEDecodePrimitive(dst_dtype, fse_buffer.curr_chunk, fse_buffer.curr_chunk_index,
                                                     fse_buffer.curr_bit_count, fse_buffer.table_log);
  op_inputs->push_back(new_primitive);

  // make shape to (1,chunk_size)
  ShapeVector shape_vector;
  shape_vector.push_back(1);
  shape_vector.push_back(fse_buffer.chunk_size);
  auto chunk_tensor_info =
    lite::CreateTensorInfo(fse_buffer.chunks, fse_buffer.chunk_size, shape_vector, kNumberTypeInt8);
  parameter_ptr->set_default_param(chunk_tensor_info);
  parameter_ptr->set_abstract(chunk_tensor_info->ToAbstract());
  op_inputs->push_back(input_node);

  size_t table_size = 1u << fse_buffer.table_log;
  uint16_t *states_table = static_cast<uint16_t *>(malloc(table_size * sizeof(uint16_t)));
  CHECK_NULL_RETURN(states_table);
  uint8_t *bit_count_table = static_cast<uint8_t *>(malloc(table_size * sizeof(uint8_t)));
  CHECK_NULL_RETURN(bit_count_table);
  uint16_t *symbol_table = static_cast<uint16_t *>(malloc(table_size * sizeof(uint16_t)));
  CHECK_NULL_RETURN(symbol_table);

  ret = FSEDecoder::FSECreateStatesForDecoding(fse_buffer.frequency, fse_buffer.frequency_count, fse_buffer.table_log,
                                               states_table, bit_count_table, symbol_table);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FSE create states for decoding failed.";
    free(states_table);
    free(bit_count_table);
    free(symbol_table);
    return RET_ERROR;
  }
  std::vector<int64_t> shape = {static_cast<int64_t>(table_size)};

  auto states_table_tensor_info =
    lite::CreateTensorInfo(states_table, sizeof(uint16_t) * table_size, shape, kNumberTypeUInt16);
  auto states_table_node = opt::BuildParameterNode(func_graph, states_table_tensor_info, "states_table");
  op_inputs->push_back(states_table_node);

  auto bit_count_table_tensor_info =
    lite::CreateTensorInfo(bit_count_table, sizeof(uint8_t) * table_size, shape, kNumberTypeUInt8);
  auto bit_count_table_node = opt::BuildParameterNode(func_graph, bit_count_table_tensor_info, "bit_count_table");
  op_inputs->push_back(bit_count_table_node);

  auto symbol_table_tensor_info =
    lite::CreateTensorInfo(symbol_table, sizeof(uint16_t) * table_size, shape, kNumberTypeUInt16);
  auto symbol_table_node = opt::BuildParameterNode(func_graph, symbol_table_tensor_info, "symbol_table");
  op_inputs->push_back(symbol_table_node);

  auto centroids_tensor_info =
    lite::CreateTensorInfo(fse_buffer.centroids, sizeof(float) * fse_buffer.centroid_size,
                           {static_cast<int64_t>(fse_buffer.centroid_size)}, kNumberTypeFloat32);
  auto centroids_node = opt::BuildParameterNode(func_graph, centroids_tensor_info, "centroids");
  op_inputs->push_back(centroids_node);

  auto shape_tensor_info = lite::CreateTensorInfo(ConvertShapeVectorToInt32(tensor->shape_c()).data(),
                                                  sizeof(int32_t) * tensor->shape_c().size(),
                                                  {static_cast<int64_t>(tensor->shape_c().size())}, kNumberTypeInt32);
  auto shape_node = opt::BuildParameterNode(func_graph, shape_tensor_info, "input_shape");
  op_inputs->push_back(shape_node);

  auto chunk_ends_tensor_info =
    lite::CreateTensorInfo(fse_buffer.chunk_ends, sizeof(uint64_t) * fse_buffer.chunk_ends_count,
                           {static_cast<int64_t>(fse_buffer.chunk_ends_count)}, kNumberTypeUInt64);
  auto chunk_ends_node = opt::BuildParameterNode(func_graph, chunk_ends_tensor_info, "chunk_ends");
  op_inputs->push_back(chunk_ends_node);

  // Free buffer
  free(states_table);
  free(bit_count_table);
  free(symbol_table);
  return RET_OK;
}

ValueNodePtr InsertQuantNodeManager::NewQuantCastPrimitive(int src_type, int dst_type,
                                                           const std::vector<schema::QuantParamT> &input_quant_params,
                                                           const std::vector<schema::QuantParamT> &output_quant_params,
                                                           int axis, bool set_quant_flag) {
  auto prim_c = std::make_shared<ops::QuantDTypeCast>();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr.");
  prim_c->Init(src_type, dst_type);
  prim_c->set_axis(axis);
  auto quant_params_holder = std::make_shared<QuantParamHolder>(input_quant_params.size(), output_quant_params.size());
  MS_CHECK_TRUE_MSG(quant_params_holder != nullptr, nullptr, "quant_params_holder is nullptr.");
  if (set_quant_flag) {
    quant_params_holder->set_quant_type(quant::QUANT_ALL);
  }
  quant_params_holder->set_input_quant_param(0, input_quant_params);
  quant_params_holder->set_output_quant_param(0, output_quant_params);
  auto prim = prim_c->GetPrim();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is nullptr");
  prim->AddAttr("quant_params", quant_params_holder);
  return NewValueNode(prim);
}

ValueNodePtr InsertQuantNodeManager::NewFSEDecodePrimitive(int dst_type, uint64_t curr_chunk, int64_t curr_chunk_index,
                                                           int64_t curr_bit_count, int64_t table_log) {
  auto prim_c = std::make_shared<ops::FSEDecode>();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr.");
  prim_c->Init(dst_type, curr_chunk, curr_chunk_index, curr_bit_count, table_log);

  auto prim = prim_c->GetPrim();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is nullptr");
  return NewValueNode(prim);
}

int InsertQuantNodeManager::InsertAscendQuantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  for (size_t i = 1; i < cnode->size(); i++) {
    if (cnode->input(i)->isa<CNode>() || IsGraphInput(cnode->input(i))) {
      auto ret = InsertAscendQuantNode(func_graph, cnode, i);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "InsertAscendQuantNode failed.";
        return ret;
      }
    }
  }
  return RET_OK;
}

int InsertQuantNodeManager::InsertAscendQuantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                  size_t input_index) {
  CHECK_NULL_RETURN(func_graph);
  CHECK_NULL_RETURN(cnode);
  auto curr_quant_param_holder = GetCNodeQuantHolder(cnode);
  CHECK_NULL_RETURN(curr_quant_param_holder);
  auto input_quant_param = curr_quant_param_holder->get_input_quant_params();
  if (input_quant_param.size() < kMinSize2) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " input quant params " << input_quant_param.size() << " < 2";
    return RET_ERROR;
  }
  auto x_q_param = input_quant_param.at(input_index - kPrimOffset);
  if (x_q_param.size() != kPerTensor) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " x quant param size " << x_q_param.size() << " != 1";
    return RET_ERROR;
  }
  x_q_param.at(0).scale = 1 / x_q_param.at(0).scale;
  ValueNodePtr new_primitive = NewQuantCastPrimitive(kNumberTypeFloat32, kNumberTypeInt8, x_q_param, x_q_param);

  std::vector<AnfNodePtr> op_inputs = {new_primitive, cnode->input(input_index)};
  auto quant_cast_cnode = func_graph->NewCNode(op_inputs);
  CHECK_NULL_RETURN(quant_cast_cnode);
  quant_cast_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "-quant-" + std::to_string(input_index));
  // set abstract
  if (cnode->input(input_index)->abstract() != nullptr) {
    auto abstract = cnode->input(input_index)->abstract()->Clone();
    quant_cast_cnode->set_abstract(abstract);
    if (quant::UpdateDataType(quant_cast_cnode, kNumberTypeInt8) != RET_OK) {
      MS_LOG(ERROR) << "UpdateDataType failed, cnode name: " << quant_cast_cnode->fullname_with_scope();
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "input node abstract nullptr, input node name: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  CHECK_NULL_RETURN(manager);
  manager->SetEdge(cnode, input_index, quant_cast_cnode);
  MS_LOG(INFO) << cnode->fullname_with_scope() << " Insert Ascend QuantNode.";
  return RET_OK;
}

int InsertQuantNodeManager::InsertAscendDeQuantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  CHECK_NULL_RETURN(func_graph);
  CHECK_NULL_RETURN(cnode);
  auto curr_quant_param_holder = GetCNodeQuantHolder(cnode);
  CHECK_NULL_RETURN(curr_quant_param_holder);
  auto input_quant_param = curr_quant_param_holder->get_input_quant_params();
  if (input_quant_param.size() < kMinSize2) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " input quant params " << input_quant_param.size() << " < 2";
    return RET_ERROR;
  }
  auto x_q_param = input_quant_param.at(0);
  if (x_q_param.size() != kPerTensor) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " x quant param size " << x_q_param.size() << " != 1";
    return RET_ERROR;
  }
  auto w_q_params = input_quant_param.at(1);
  if (w_q_params.empty()) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " w quant param is empty.";
    return RET_ERROR;
  }
  std::vector<uint64_t> deq_scales(w_q_params.size());
  for (size_t i = 0; i < w_q_params.size(); ++i) {
    float float32_deq_scale = static_cast<float>(x_q_param.at(0).scale * w_q_params.at(i).scale);
    void *ptr = &float32_deq_scale;
    uint32_t *uint32_deq_scale = reinterpret_cast<uint32_t *>(ptr);
    uint64_t u64_deq_scale = 0;
    u64_deq_scale |= *uint32_deq_scale;
    deq_scales[i] = u64_deq_scale;
  }
  auto dtype = kNumberTypeFloat32;
  if (cnode->HasAttr("origin_type")) {
    auto value = cnode->GetAttr("origin_type");
    dtype = static_cast<TypeId>(opt::CastToInt(value).front());
  }
  auto prim_c = std::make_shared<ops::QuantDTypeCast>();
  CHECK_NULL_RETURN(prim_c);

  prim_c->Init(kNumberTypeInt32, dtype);
  auto prim = prim_c->GetPrim();
  auto quant_dtype_cast_primitive = NewValueNode(prim);
  std::vector<AnfNodePtr> op_inputs;
  op_inputs.push_back(quant_dtype_cast_primitive);
  op_inputs.push_back(cnode);
  auto deq_scales_tensor_info = lite::CreateTensorInfo(deq_scales.data(), sizeof(uint64_t) * deq_scales.size(),
                                                       {static_cast<int64_t>(deq_scales.size())}, kNumberTypeUInt64);
  auto deq_scales_node =
    opt::BuildParameterNode(func_graph, deq_scales_tensor_info, cnode->fullname_with_scope() + "-deq_scales");
  op_inputs.push_back(deq_scales_node);

  auto quant_cast_cnode = func_graph->NewCNode(op_inputs);
  CHECK_NULL_RETURN(quant_cast_cnode);
  quant_cast_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "-dequant");
  // set abstract
  if (cnode->abstract() != nullptr) {
    auto abstract = cnode->abstract()->Clone();
    quant_cast_cnode->set_abstract(abstract);
    if (quant::UpdateDataType(quant_cast_cnode, dtype) != RET_OK) {
      MS_LOG(ERROR) << "UpdateDataType failed, cnode name: " << quant_cast_cnode->fullname_with_scope();
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "input node abstract nullptr, input node name: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  CHECK_NULL_RETURN(manager);
  auto node_users = manager->node_users()[cnode];
  for (auto &node_user : node_users) {
    manager->SetEdge(node_user.first, node_user.second, quant_cast_cnode);
  }
  MS_LOG(INFO) << cnode->fullname_with_scope() << " Insert Ascend DeQuant Node.";
  return RET_OK;
}

int InsertQuantNodeManager::AdjustTransposeNodeForMatMul(const FuncGraphPtr &func_graph) {
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (opt::CheckPrimitiveType(cnode, prim::kPrimMatMulFusion)) {
      auto prim_ptr = GetCNodePrimitive(cnode);
      CHECK_NULL_RETURN(prim_ptr);

      auto transpose_a = prim_ptr->GetAttr(mindspore::ops::kTransposeA);
      auto transpose_b = prim_ptr->GetAttr(mindspore::ops::kTransposeB);

      if (transpose_a != nullptr && GetValue<bool>(transpose_a)) {
        MS_LOG(ERROR) << cnode->fullname_with_scope() << " transposeA is true.";
        return RET_ERROR;
      }
      if (transpose_b != nullptr && GetValue<bool>(transpose_b)) {
        int ret = RET_ERROR;
        MS_LOG(INFO) << cnode->fullname_with_scope() << ":" << cnode->input(kWeightIndex + kPrimOffset)->type_name();
        if (cnode->input(kWeightIndex + kPrimOffset)->isa<CNode>()) {
          ret = InsertTransposeNode(func_graph, cnode, kWeightQuant + kPrimOffset);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << cnode->fullname_with_scope() << " insert transpose node failed";
            return ret;
          }
        } else if (cnode->input(kWeightIndex + kPrimOffset)->isa<Parameter>()) {
          auto manager = Manage(func_graph);
          CHECK_NULL_RETURN(manager);
          auto users = manager->node_users();
          if (users[cnode->input(kWeightIndex + kPrimOffset)].size() > 1) {
            MS_LOG(ERROR) << "Dont support share weight.";
            return RET_ERROR;
          }
          auto weight_input = cnode->input(kWeightIndex + 1);
          auto dst_prim = GetCNodePrimitive(cnode);
          MS_LOG(INFO) << cnode->fullname_with_scope() << " transpose_b is true.";
          dst_prim->AddAttr(mindspore::ops::kTransposeB, MakeValue(false));
          ParameterPtr param_node;
          tensor::TensorPtr tensor_info;
          GetParameterAndTensor(weight_input, &param_node, &tensor_info);
          if (tensor_info->shape_c().size() != DIMENSION_2D) {
            MS_LOG(ERROR) << weight_input->fullname_with_scope() << " shape is " << tensor_info->shape_c()
                          << " is large than 2.";
            return RET_ERROR;
          }

          if (tensor_info->data_type_c() == kNumberTypeFloat32) {
            ret = TransposeData<float>(param_node, tensor_info);
          } else if (tensor_info->data_type_c() == kNumberTypeFloat16) {
            ret = TransposeData<Float16>(param_node, tensor_info);
          } else {
            MS_LOG(ERROR) << "transpose data only support Float32 or Float16.";
            return RET_OK;
          }

          if (ret != RET_OK) {
            MS_LOG(ERROR) << weight_input->fullname_with_scope() << " transposeData failed.";
            return ret;
          }
        } else {
          MS_LOG(ERROR) << "Dont support type is " << cnode->input(kWeightIndex + kPrimOffset)->type_name();
          return RET_ERROR;
        }
      }
    }
  }
  return RET_OK;
}

int InsertQuantNodeManager::InsertTransposeNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t index) {
  auto prim_ptr = GetCNodePrimitive(cnode);
  CHECK_NULL_RETURN(prim_ptr);
  std::vector<int> perm;
  ShapeVector shape;
  auto ret = opt::FetchShapeFromAbstract(cnode->input(index)->abstract(), &shape);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fetch shape from abstract failed.";
    return RET_OK;
  }
  if (shape.size() == DIMENSION_2D) {
    perm = {1, 0};
  } else if (shape.size() == DIMENSION_3D) {
    perm = {0, 2, 1};
  } else if (shape.size() == DIMENSION_4D) {
    perm = {0, 1, 3, 2};
  } else {
    MS_LOG(ERROR) << shape.size() << " is invalid.";
    return RET_ERROR;
  }
  auto transpose = opt::GenTransposeNode(func_graph, cnode->input(index), perm,
                                         cnode->input(index)->fullname_with_scope() + "-transpose");
  auto manager = Manage(func_graph);
  MS_ASSERT(manager != nullptr);
  manager->SetEdge(cnode, kWeightIndex + kPrimOffset, transpose);
  prim_ptr->set_attr(mindspore::ops::kTransposeB, MakeValue(false));
  return RET_OK;
}
}  // namespace mindspore::lite::quant
