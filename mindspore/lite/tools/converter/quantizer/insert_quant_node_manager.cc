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
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "tools/optimizer/graph/node_infershape.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/quantizer/fse_decoder.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "ops/fse_decode.h"
#include "ops/op_name.h"
#include "ops/cast.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/array_ops.h"
#include "ir/dtype.h"

namespace mindspore::lite::quant {
namespace {
constexpr size_t kMinSize2 = 2;
constexpr size_t kMinSize3 = 3;
constexpr size_t kTableExtend = 3;
constexpr size_t kAlignOffset = 7;
constexpr size_t kInt32Mask = 31;
constexpr int kLastFisrtIndex = -1;
constexpr int kLastSecondIndex = -2;
const char *ATTR_NO_NEED_CONSTANT_FOLDING = "no_need_constant_folding";
constexpr char IN_STRATEGY[] = "in_strategy";
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

int InsertQuantNodeManager::InsertDynamicQuantWithIndex(const FuncGraphPtr &graph, const CNodePtr &cnode, size_t index,
                                                        bool activation_channel) {
  auto primitive = std::make_shared<ops::DynamicQuant>();
  CHECK_NULL_RETURN(primitive);
  auto primitive_c = primitive->GetPrim();
  primitive->set_dst_type(dst_type_);
  bool symmetric = activation_channel ? true : false;
  primitive->set_symmetric(symmetric);
  primitive->set_activation_channel(activation_channel);
  if (activation_channel && SetPreferAxis(cnode, index, primitive) != RET_OK) {
    MS_LOG(ERROR) << "Set prefer axis failed, " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto dynamic_quant_cnode = graph->NewCNode(primitive_c, {cnode->input(index)});
  CHECK_NULL_RETURN(dynamic_quant_cnode);
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
  ret = MarkDynamicQuantize(dynamic_quant_cnode);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " mark quant type failed.";
    return ret;
  }
  cnode->set_input(index, dynamic_quant_cnode);
  return RET_OK;
}

int InsertQuantNodeManager::SetPreferAxis(const CNodePtr &cnode, size_t index,
                                          const std::shared_ptr<ops::DynamicQuant> &dynamic_primitive) {
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive->name() == ops::kNameMatMulFusion || primitive->name() == ops::kNameMatMul) {
    auto matmul_prim = api::MakeShared<ops::MatMul>(primitive);
    CHECK_NULL_RETURN(matmul_prim);
    // For MatMul A
    if (index == kInputIndex + kPrimOffset) {
      if (matmul_prim->GetAttr(ops::kTransposeA) != nullptr && matmul_prim->get_transpose_a()) {
        dynamic_primitive->set_prefer_axis(kLastFisrtIndex);
        dynamic_primitive->set_transpose(true);
      } else {
        dynamic_primitive->set_prefer_axis(kLastSecondIndex);
        dynamic_primitive->set_transpose(false);
      }
    }
    // For MatMul B
    if (index == kWeightIndex + kPrimOffset) {
      if (matmul_prim->GetAttr(ops::kTransposeB) != nullptr && matmul_prim->get_transpose_b()) {
        dynamic_primitive->set_prefer_axis(kLastSecondIndex);
        dynamic_primitive->set_transpose(true);
      } else {
        dynamic_primitive->set_prefer_axis(kLastFisrtIndex);
        dynamic_primitive->set_transpose(false);
      }
    }
  } else {
    MS_LOG(WARNING) << "cnode don't need prefer axis, cnode name: " << cnode->fullname_with_scope();
  }
  return RET_OK;
}

int InsertQuantNodeManager::NewDynamicQuantNode(const FuncGraphPtr &graph, const CNodePtr &cnode,
                                                bool activation_channel) {
  auto op_name = cnode->fullname_with_scope();
  if (cnode->size() < kMinSize3) {
    MS_LOG(ERROR) << op_name << " cnode size:" << cnode->size() << " < 3.";
    return RET_ERROR;
  }
  auto input = cnode->input(kInputIndex + kPrimOffset);
  if (input->isa<mindspore::CNode>() || IsGraphInput(input)) {
    auto ret = InsertDynamicQuantWithIndex(graph, cnode, kInputIndex + kPrimOffset, activation_channel);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert dynamic quant with index failed.";
    }
  }
  auto weight = cnode->input(kWeightIndex + kPrimOffset);
  if (weight->isa<mindspore::CNode>() || IsGraphInput(weight)) {
    auto ret = InsertDynamicQuantWithIndex(graph, cnode, kWeightIndex + kPrimOffset, activation_channel);
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
                                                   const std::set<std::string> &skip_quant_node,
                                                   bool activation_channel) {
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
    ret = NewDynamicQuantNode(graph, cnode, activation_channel);
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

int InsertQuantNodeManager::InsertQuantDtypeCastNodeNew(const FuncGraphPtr &graph, const CNodePtr &cnode,
                                                        InsertDirection insert_direction, TypeId cast_dtype,
                                                        CastNodeType cast_node_type, size_t index,
                                                        const AnfNodePtr &output_node) {
  CHECK_NULL_RETURN(graph);
  CHECK_NULL_RETURN(cnode);
  if (insert_direction == FORWARD) {
    return InsertForwardQuantNodeNew(graph, cnode, cast_dtype, index, cast_node_type);
  } else if (insert_direction == BACKWARD && cast_node_type == kDeQuant) {
    return InsertBackwardDeQuantNode(graph, cnode, cast_dtype, index, output_node);
  }
  MS_LOG(ERROR) << "Invalid insert direction: " << insert_direction;
  return RET_NOT_SUPPORT;
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

int InsertQuantNodeManager::InsertForwardQuantNodeNew(const FuncGraphPtr &graph, const CNodePtr &cnode,
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
  std::vector<schema::QuantParamT> cast_input_quant_params;
  std::vector<schema::QuantParamT> cast_output_quant_params;
  if (cast_node_type == kQuant) {
    src_dtype = cast_dtype;
    dst_dtype = kNumberTypeInt8;
    cast_output_quant_params = quant::GetInputNodeQuantParam(cnode, index);
    std::copy(cast_output_quant_params.cbegin(), cast_output_quant_params.cend(),
              std::back_inserter(cast_input_quant_params));
    // Uint8toInt8
    if (src_dtype == kNumberTypeUInt8) {
      for (auto &quant_param : cast_input_quant_params) {
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
    auto quantization_param_value = input_cnode_primitive_c->GetAttr(quant::kQuantParam);
    MS_CHECK_TRUE_MSG(quantization_param_value != nullptr, RET_ERROR, "quantization_param_value is nullptr.");
    auto quantization_param_list = GetValue<std::vector<QuantizationParamPtr>>(quantization_param_value);
    if (quantization_param_list.empty()) {
      MS_LOG(ERROR) << input_node->fullname_with_scope() << " quantization param Not exist.";
      return RET_ERROR;
    }
    cast_input_quant_params = quant::ConvertQuantizationParamToQuantParamT(quantization_param_list.front());
    std::copy(cast_input_quant_params.cbegin(), cast_input_quant_params.cend(),
              std::back_inserter(cast_output_quant_params));
  }
  ValueNodePtr new_primitive =
    NewQuantCastPrimitive(src_dtype, dst_dtype, input_node, cast_output_quant_params, 0, true);
  CHECK_NULL_RETURN(new_primitive);
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
  CHECK_NULL_RETURN(new_primitive);
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
  CHECK_NULL_RETURN(new_primitive);
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
      if (GetQuantTypeNew(input_node->cast<mindspore::CNodePtr>(), &pre_quant_type) != RET_OK) {
        MS_LOG(ERROR) << "Get quant type failed, cnode name: " << cnode->fullname_with_scope();
        return RET_ERROR;
      }
    }
    if (pre_quant_type == quant::QUANT_NONE && curr_quant_type == quant::QUANT_ALL) {
      auto status = InsertQuantDtypeCastNodeNew(graph, cnode, FORWARD, cast_dtype, kQuant, index, nullptr);
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "InsertQuantDtypeCastNode kQuant failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
    } else if (pre_quant_type == quant::QUANT_ALL && curr_quant_type == quant::QUANT_NONE) {
      auto status = InsertQuantDtypeCastNodeNew(graph, cnode, FORWARD, cast_dtype, kDeQuant, index, nullptr);
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
                                                        int axis, bool is_quant_attribute) {
  MS_CHECK_LT(input_index, cnode->size(), RET_ERROR);
  auto cnode_primitive = GetValueNode<std::shared_ptr<mindspore::Primitive>>(cnode->input(kPrimIndex));
  if (cnode_primitive == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto input_node = cnode->input(input_index);
  if (!input_node->isa<mindspore::Parameter>()) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " input " << input_index << " is not parameter node.";
    return RET_ERROR;
  }
  auto input_quant_params = quant::GetInputNodeQuantParam(cnode, input_index);

  CNodePtr quant_cast_cnode = nullptr;
  if (is_quant_attribute) {
    ValueNodePtr new_primitive = NewQuantCastPrimitive(src_dtype, dst_dtype, input_quant_params, {}, axis, false);
    MS_CHECK_TRUE_MSG(new_primitive != nullptr, RET_NULL_PTR, "New quant_cast primitive failed!");
    std::vector<AnfNodePtr> op_inputs = {new_primitive, input_node};
    quant_cast_cnode = func_graph->NewCNode(op_inputs);
  } else {
    quant_cast_cnode =
      CreateQuantInputCastNode(func_graph, cnode, input_node, src_dtype, dst_dtype, input_quant_params, axis);
  }
  CHECK_NULL_RETURN(quant_cast_cnode);
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
  cnode_primitive->DelAttr(quant::kQuantParam);
  MS_LOG(INFO) << "InsertCastNode cnode name: " << quant_cast_cnode->fullname_with_scope()
               << " src_dtype: " << src_dtype << " dst_dtype: " << dst_dtype;

  return RET_OK;
}

CNodePtr InsertQuantNodeManager::CreateQuantInputCastNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                          const AnfNodePtr input_node, TypeId src_dtype,
                                                          TypeId dst_dtype,
                                                          const std::vector<schema::QuantParamT> &input_quant_params,
                                                          int axis) {
  ValueNodePtr new_primitive = NewQuantCastPrimitive(src_dtype, dst_dtype, input_node, {}, axis, false);
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
  if (quant_cast_cnode == nullptr) {
    MS_LOG(ERROR) << "New quant cast node failed.";
    return nullptr;
  }
  auto strings = SplitStringToVector(cnode->fullname_with_scope(), "-op");
  int index = 0;
  if (!ConvertIntNum(strings.at(strings.size() - 1), &index)) {
    index = 0;
  }
  const int quant_dtype_cast_offset = 10000;
  quant_cast_cnode->set_fullname_with_scope(strings.at(0) + "-QuantDtypeCast-op" +
                                            std::to_string(index + quant_dtype_cast_offset));
  return quant_cast_cnode;
}

int InsertQuantNodeManager::CalculateScaleZPNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                 size_t input_index, ParameterPtr *scales_node, ParameterPtr *zps_node,
                                                 TypeId src_dtype, TypeId dst_dtype, int axis) {
  CHECK_NULL_RETURN(scales_node);
  CHECK_NULL_RETURN(zps_node);
  MS_CHECK_LT(input_index, cnode->size(), RET_ERROR);
  auto input_node = cnode->input(input_index);
  auto input_quant_params = quant::GetInputNodeQuantParam(cnode, input_index);
  if (input_quant_params.empty()) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " index: " << input_index << " quant param is empty.";
    return RET_ERROR;
  }

  if (dst_dtype == kNumberTypeFloat16) {
    std::vector<float16> scales;
    std::vector<float16> zps;
    for (size_t i = 0; i < input_quant_params.size(); ++i) {
      scales.push_back(static_cast<float16>(input_quant_params.at(i).scale * input_quant_params.at(i).varCorr));
      zps.push_back(static_cast<float16>(-input_quant_params.at(i).zeroPoint +
                                         input_quant_params.at(i).meanCorr /
                                           (input_quant_params.at(i).scale * input_quant_params.at(i).varCorr)));
    }
    *scales_node = opt::BuildFloat16VecParameterNode(func_graph, scales, input_node->fullname_with_scope() + "-scales");
    *zps_node = opt::BuildFloat16VecParameterNode(func_graph, zps, input_node->fullname_with_scope() + "-zps");
  } else {
    std::vector<float> scales;
    std::vector<float> zps;
    for (size_t i = 0; i < input_quant_params.size(); ++i) {
      scales.push_back(static_cast<float>(input_quant_params.at(i).scale * input_quant_params.at(i).varCorr));
      zps.push_back(static_cast<float>(-input_quant_params.at(i).zeroPoint +
                                       input_quant_params.at(i).meanCorr /
                                         (input_quant_params.at(i).scale * input_quant_params.at(i).varCorr)));
    }
    *scales_node = opt::BuildFloatVecParameterNode(func_graph, scales, input_node->fullname_with_scope() + "-scales");
    *zps_node = opt::BuildFloatVecParameterNode(func_graph, zps, input_node->fullname_with_scope() + "-zps");
  }
  if (*scales_node == nullptr || *zps_node == nullptr) {
    MS_LOG(ERROR) << "Failed to build scales node, zps node ";
    return RET_ERROR;
  }
  if (input_quant_params.size() > 1) {
    ShapeVector shape;
    if (opt::FetchShapeFromAbstract(input_node->abstract(), &shape) != lite::RET_OK) {
      MS_LOG(ERROR) << "fetch shape failed." << input_node->fullname_with_scope();
      return lite::RET_ERROR;
    }

    std::vector<int64_t> shape_vector = {};
    for (size_t i = 0; i < shape.size(); i++) {
      if (i == static_cast<size_t>(axis)) {
        shape_vector.push_back((int64_t)input_quant_params.size());
      } else {
        shape_vector.push_back(1);
      }
    }
    auto scales_abstract = (*scales_node)->abstract();
    CHECK_NULL_RETURN(scales_abstract);
    scales_abstract->set_shape(std::make_shared<abstract::Shape>(shape_vector));
    auto zps_abstract = (*zps_node)->abstract();
    CHECK_NULL_RETURN(zps_abstract);
    zps_abstract->set_shape(std::make_shared<abstract::Shape>(shape_vector));
  }
  return RET_OK;
}

int InsertQuantNodeManager::SetParallelStrategy(const CNodePtr &cnode,
                                                const std::vector<std::vector<int64_t>> &in_strategy) {
  auto primitive = GetValueNode<std::shared_ptr<mindspore::Primitive>>(cnode->input(kPrimIndex));
  CHECK_NULL_RETURN(primitive);
  primitive->AddAttr(IN_STRATEGY, MakeValue(in_strategy));
  return RET_OK;
}

std::vector<std::vector<int64_t>> InsertQuantNodeManager::ExtractStrategy(const ValuePtr &stra) {
  if (stra == nullptr) {
    return {};
  }

  auto var = stra->cast<ValueTuplePtr>();
  if (var == nullptr) {
    return {};
  }
  std::vector<std::vector<int64_t>> strategy;
  MS_LOG(INFO) << "Extract information: strategy " << stra->ToString();
  if (var->size() > 0) {
    std::vector<ValuePtr> elements = var->value();
    for (uint64_t index = 0; index < elements.size(); ++index) {
      std::vector<int64_t> dim;
      if (elements[index]->isa<ValueSequence>()) {
        auto value_tuple = elements[index]->cast<ValueTuplePtr>();
        std::vector<ValuePtr> value_vector = value_tuple->value();
        (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(dim),
                             [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
        strategy.push_back(dim);
      } else {
        MS_LOG(EXCEPTION) << "Failure: Strategy's format is wrong! Need ValueSequence";
      }
    }
    if (strategy.empty()) {
      MS_LOG(EXCEPTION) << "ExtractStrategy: failed to extract strategy";
    }
  }

  return strategy;
}

std::vector<std::vector<int64_t>> InsertQuantNodeManager::GetAddMulNodeParallelStrategy(
  ShapeVector weight_shape, std::vector<int64_t> weight_strategy, int axis, bool per_channel) {
  std::vector<std::vector<int64_t>> add_mul_in_strategy;
  std::vector<int64_t> in_strategy_1 = weight_strategy;
  add_mul_in_strategy.push_back(in_strategy_1);
  std::vector<int64_t> in_strategy_2;

  // if perlayer quant, the input2 strategy is set to 1.
  // if perchannel quant, the input2 strategy is set by axis, the axis dim is set by matmul input strategy,
  // the other dim is set to 1.
  if (per_channel) {
    for (size_t i = 0; i < weight_shape.size(); i++) {
      if (i == static_cast<size_t>(axis) && i < weight_strategy.size()) {
        in_strategy_2.push_back(weight_strategy.at(i));
      } else {
        in_strategy_2.push_back(1);
      }
    }
  } else {
    in_strategy_2.push_back(1);
  }

  add_mul_in_strategy.push_back(in_strategy_2);
  return add_mul_in_strategy;
}

int InsertQuantNodeManager::InsertAscendAntiQuantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                      size_t input_index, TypeId src_dtype, TypeId dst_dtype, int axis,
                                                      const std::string &ascend_backend) {
  auto primitive = GetValueNode<std::shared_ptr<mindspore::Primitive>>(cnode->input(kPrimIndex));
  CHECK_NULL_RETURN(primitive);
  MS_CHECK_LT(input_index, cnode->size(), RET_ERROR);
  auto input_node = cnode->input(input_index);
  auto manager = func_graph->manager();
  CHECK_NULL_RETURN(manager);
  std::vector<std::vector<int64_t>> cnode_in_strategy;
  if (primitive->HasAttr(IN_STRATEGY)) {
    cnode_in_strategy = ExtractStrategy(primitive->GetAttr(IN_STRATEGY));
    CHECK_LESS_RETURN(cnode_in_strategy.size(), input_index);
    MS_LOG(INFO) << "cnode: " << cnode->fullname_with_scope() << " in strategy is " << cnode_in_strategy;
  }
  if (!input_node->isa<mindspore::Parameter>()) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " input " << input_index << " is not parameter node.";
    return RET_ERROR;
  }

  // parameter+cast+add+mul+matmul
  // parameter+gather+cast+add+mul
  auto input_quant_params = quant::GetInputNodeQuantParam(cnode, input_index);
  if (input_quant_params.empty()) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " index: " << input_index << " quant param is empty.";
    return RET_ERROR;
  }

  // Insert cast node
  CNodePtr cast_cnode = nullptr;
  if (ascend_backend == "910b") {
    MS_LOG(INFO) << "The ascend_backend is 910b, it will insert antiquant node";
    if (opt::CheckPrimitiveType(cnode, prim::kPrimGather)) {
      cast_cnode = NewAscendAntiQuantCNode(func_graph, cnode, dst_dtype);
    } else {
      cast_cnode = NewAscendAntiQuantCNode(func_graph, input_node, dst_dtype);
    }
  } else {
    if (opt::CheckPrimitiveType(cnode, prim::kPrimGather)) {
      cast_cnode = NewCastNode(func_graph, cnode, dst_dtype);
    } else {
      cast_cnode = NewCastNode(func_graph, input_node, dst_dtype);
    }
  }

  CHECK_NULL_RETURN(cast_cnode);
  // cast node do not need to set parallel strategy, antiquant node need set parallel strategy
  if (primitive->HasAttr(IN_STRATEGY) && ascend_backend == "910b") {
    std::vector<std::vector<int64_t>> cast_in_strategy;
    std::vector<int64_t> in_strategy_1 = cnode_in_strategy[input_index - kPrimOffset];
    cast_in_strategy.push_back(in_strategy_1);
    auto ret = SetParallelStrategy(cast_cnode, cast_in_strategy);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fail to set cnode parallel strategy, cnode: " << cast_cnode->fullname_with_scope();
      return RET_ERROR;
    }
  }

  ParameterPtr scales_node;
  ParameterPtr zps_node;
  auto ret = CalculateScaleZPNode(func_graph, cnode, input_index, &scales_node, &zps_node, src_dtype, dst_dtype, axis);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fail to calculate scale & zero_point node: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  auto add_cnode = NewAddNode(func_graph, cast_cnode, zps_node);
  CHECK_NULL_RETURN(add_cnode);

  auto mul_cnode = NewMulNode(func_graph, add_cnode, scales_node);
  CHECK_NULL_RETURN(mul_cnode);

  if (primitive->HasAttr(IN_STRATEGY)) {
    ShapeVector weight_shape;
    if (opt::FetchShapeFromAbstract(input_node->abstract(), &weight_shape) != lite::RET_OK) {
      MS_LOG(ERROR) << "fetch shape failed." << input_node->fullname_with_scope();
      return lite::RET_ERROR;
    }
    std::vector<int64_t> weight_strategy = cnode_in_strategy[input_index - kPrimOffset];
    bool per_channel = input_quant_params.size() > 1;
    auto add_mul_in_strategy = GetAddMulNodeParallelStrategy(weight_shape, weight_strategy, axis, per_channel);

    // add_cnode & mul_cnode set parallel strategy
    ret = SetParallelStrategy(add_cnode, add_mul_in_strategy);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fail to set add cnode parallel strategy, cnode: " << add_cnode->fullname_with_scope();
      return RET_ERROR;
    }
    ret = SetParallelStrategy(mul_cnode, add_mul_in_strategy);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fail to set mul cnode parallel strategy, cnode: " << mul_cnode->fullname_with_scope();
      return RET_ERROR;
    }
  }

  auto node_map = manager->node_users();

  // Remove QuantParam
  ret = RemoveInputNodeQuantParam(cnode, input_index);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fail to Remove node: " << input_node->fullname_with_scope() << " quant param";
    return RET_ERROR;
  }

  AnfNodeIndexSet node_user;
  if (opt::CheckPrimitiveType(cnode, prim::kPrimGather)) {
    node_user = node_map[cnode];
  } else {
    node_user = node_map[input_node];
  }
  for (const auto &user : node_user) {
    manager->SetEdge(user.first, user.second, mul_cnode);
  }
  return RET_OK;
}

int InsertQuantNodeManager::InsertFSEDecodeNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                size_t input_index, TypeId dst_dtype) {
  auto primitive = GetValueNode<std::shared_ptr<mindspore::Primitive>>(cnode->input(kPrimIndex));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  MS_CHECK_LT(input_index, cnode->size(), RET_ERROR);
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
  auto ret_bool = manager->Replace(input_node, fse_decode_cnode);
  if (!ret_bool) {
    MS_LOG(ERROR) << "Replace QuantDtypeCast failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

int InsertQuantNodeManager::CreateFSEInputs(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                                            std::vector<AnfNodePtr> *op_inputs, TypeId dst_dtype) {
  CHECK_NULL_RETURN(op_inputs);
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
  CHECK_NULL_RETURN(tensor);
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
  std::vector<uint16_t> states_table(table_size);
  std::vector<uint8_t> bit_count_table(table_size);
  std::vector<uint16_t> symbol_table(table_size);

  ret = FSEDecoder::FSECreateStatesForDecoding(fse_buffer.frequency, fse_buffer.frequency_count, fse_buffer.table_log,
                                               states_table.data(), bit_count_table.data(), symbol_table.data());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FSE create states for decoding failed.";
    return RET_ERROR;
  }
  std::vector<int64_t> shape = {static_cast<int64_t>(table_size)};

  auto states_table_tensor_info =
    lite::CreateTensorInfo(states_table.data(), sizeof(uint16_t) * table_size, shape, kNumberTypeUInt16);
  auto states_table_node = opt::BuildParameterNode(func_graph, states_table_tensor_info, "states_table");
  op_inputs->push_back(states_table_node);

  auto bit_count_table_tensor_info =
    lite::CreateTensorInfo(bit_count_table.data(), sizeof(uint8_t) * table_size, shape, kNumberTypeUInt8);
  auto bit_count_table_node = opt::BuildParameterNode(func_graph, bit_count_table_tensor_info, "bit_count_table");
  op_inputs->push_back(bit_count_table_node);

  auto symbol_table_tensor_info =
    lite::CreateTensorInfo(symbol_table.data(), sizeof(uint16_t) * table_size, shape, kNumberTypeUInt16);
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

  return RET_OK;
}

CNodePtr InsertQuantNodeManager::NewCastNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                                             int dst_type) {
  auto prim_c = std::make_shared<ops::Cast>();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr.");
  auto prim = prim_c->GetPrim();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is nullptr");
  MS_LOG(INFO) << "dst_type:" << dst_type;
  TypePtr type_ptr = TypeIdToType(TypeId(dst_type));
  prim->AddAttr(ops::kDstType, type_ptr);
  prim->AddAttr(ATTR_NO_NEED_CONSTANT_FOLDING, MakeValue(true));
  std::vector<AnfNodePtr> cast_op_inputs = {NewValueNode(prim), input_node};
  auto cast_cnode = func_graph->NewCNode(cast_op_inputs);
  cast_cnode->set_fullname_with_scope(input_node->fullname_with_scope() + "-Cast");
  cast_cnode->set_abstract(input_node->abstract()->Clone());
  auto ret = UpdateDataType(cast_cnode, TypeId(dst_type));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << cast_cnode->fullname_with_scope() << " set dst_type " << dst_type << " failed.";
    return nullptr;
  }
  return cast_cnode;
}

CNodePtr InsertQuantNodeManager::NewAscendAntiQuantCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                                                         int dst_type) {
  auto dst_prim = std::make_shared<acl::AscendAntiQuant>();
  if (dst_prim == nullptr) {
    return nullptr;
  }
  dst_prim->AddAttr("scale", MakeValue(1.0f));
  dst_prim->AddAttr("offset", MakeValue(0.0f));
  MS_LOG(INFO) << "dst_type:" << dst_type;
  TypePtr type_ptr = TypeIdToType(TypeId(dst_type));
  dst_prim->AddAttr(ops::kOutputDType, type_ptr);
  std::vector<AnfNodePtr> cast_op_inputs = {NewValueNode(dst_prim), input_node};
  auto anti_cnode = func_graph->NewCNode(cast_op_inputs);
  anti_cnode->set_fullname_with_scope(input_node->fullname_with_scope() + "-AntiQuant");
  anti_cnode->set_abstract(input_node->abstract()->Clone());
  anti_cnode->abstract()->set_type(type_ptr);
  auto ret = UpdateDataType(anti_cnode, TypeId(dst_type));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << anti_cnode->fullname_with_scope() << " set dst_type " << dst_type << " failed.";
    return nullptr;
  }
  return anti_cnode;
}

CNodePtr InsertQuantNodeManager::NewMulNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_1,
                                            const AnfNodePtr &input_2) {
  auto prim_c = std::make_shared<ops::MulFusion>();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr.");
  auto prim = prim_c->GetPrim();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is nullptr");
  prim->AddAttr(ATTR_NO_NEED_CONSTANT_FOLDING, MakeValue(true));
  std::vector<AnfNodePtr> op_inputs = {NewValueNode(prim), input_1, input_2};
  auto cnode = func_graph->NewCNode(op_inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr.";
    return nullptr;
  }
  cnode->set_fullname_with_scope(input_1->fullname_with_scope() + "-" + input_2->fullname_with_scope() + "-Mul");
  cnode->set_abstract(input_1->abstract()->Clone());
  return cnode;
}

CNodePtr InsertQuantNodeManager::NewAddNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_1,
                                            const AnfNodePtr &input_2) {
  auto prim_c = std::make_shared<ops::AddFusion>();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr.");
  auto prim = prim_c->GetPrim();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is nullptr");
  prim->AddAttr(ATTR_NO_NEED_CONSTANT_FOLDING, MakeValue(true));
  std::vector<AnfNodePtr> op_inputs = {NewValueNode(prim), input_1, input_2};
  auto cnode = func_graph->NewCNode(op_inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr.";
    return nullptr;
  }
  cnode->set_fullname_with_scope(input_1->fullname_with_scope() + "-" + input_2->fullname_with_scope() + "-Add");
  cnode->set_abstract(input_1->abstract()->Clone());
  return cnode;
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

ValueNodePtr InsertQuantNodeManager::NewQuantCastPrimitive(int src_type, int dst_type, const AnfNodePtr &input_node,
                                                           const std::vector<schema::QuantParamT> &output_quant_params,
                                                           int axis, bool set_quant_flag) {
  auto prim_c = std::make_shared<ops::QuantDTypeCast>();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr.");
  prim_c->Init(src_type, dst_type);
  prim_c->set_axis(axis);
  auto prim = prim_c->GetPrim();
  if (set_quant_flag) {
    prim->AddAttr(quant::kQuantType, MakeValue(static_cast<int>(quant::QUANT_ALL)));
  }
  // Set quant param to quant_cast_cnode
  if (!output_quant_params.empty()) {
    auto quantization_ptr = quant::ConvertQuantParamTToQuantizationParam(output_quant_params);
    std::vector<ValuePtr> quantization_list = {quantization_ptr};
    auto quant_ptr = std::make_shared<ValueList>(quantization_list);
    MS_CHECK_TRUE_MSG(quant_ptr != nullptr, nullptr, "quant_ptr is nullptr.");
    prim->AddAttr(quant::kQuantParam, quant_ptr);
  } else {
    MS_LOG(WARNING) << "New quant cast node's output quant param is empty, input node: "
                    << input_node->fullname_with_scope();
  }
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
  auto x_q_param_origin = quant::GetInputNodeQuantParam(cnode, input_index);
  if (x_q_param_origin.empty()) {
    auto curr_quant_param_holder = GetCNodeQuantHolder(cnode);
    CHECK_NULL_RETURN(curr_quant_param_holder);
    auto input_quant_param = curr_quant_param_holder->get_input_quant_params();
    x_q_param_origin = input_quant_param.at(input_index - kPrimOffset);
  }
  if (x_q_param_origin.size() != kPerTensor) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " x quant param size " << x_q_param_origin.size() << " != 1";
    return RET_ERROR;
  }
  auto x_q_param = quant::CloneQuantParam(x_q_param_origin);
  x_q_param.at(0).scale = 1 / x_q_param.at(0).scale;
  auto input_node = cnode->input(input_index);
  CHECK_NULL_RETURN(input_node);
  ValueNodePtr new_primitive = NewQuantCastPrimitive(kNumberTypeFloat32, kNumberTypeInt8, input_node, x_q_param);
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
  MS_LOG(INFO) << cnode->fullname_with_scope() << " Insert Ascend QuantNode, scale: " << x_q_param.at(0).scale;
  return RET_OK;
}

int InsertQuantNodeManager::InsertAscendDeQuantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  CHECK_NULL_RETURN(func_graph);
  CHECK_NULL_RETURN(cnode);
  auto cnode_primitive = GetValueNode<PrimitivePtr>(cnode->input(kPrimIndex));
  if (cnode_primitive == nullptr) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " primitive is nullptr.";
    return RET_ERROR;
  }
  auto curr_quant_param_holder = GetCNodeQuantHolder(cnode);
  CHECK_NULL_RETURN(curr_quant_param_holder);
  auto input_quant_param = curr_quant_param_holder->get_input_quant_params();
  auto x_q_param = quant::GetInputNodeQuantParam(cnode, Index0 + kPrimOffset);
  if (x_q_param.empty()) {
    x_q_param = input_quant_param.at(Index0);
  }
  if (x_q_param.size() != kPerTensor) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " x quant param size " << x_q_param.size() << " != 1";
    return RET_ERROR;
  }
  auto w_q_params = quant::GetInputNodeQuantParam(cnode, Index1 + kPrimOffset);
  if (w_q_params.empty()) {
    w_q_params = input_quant_param.at(Index1);
  }
  if (w_q_params.empty()) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " w quant param is empty.";
    return RET_ERROR;
  }
  MS_LOG(INFO) << cnode->fullname_with_scope() << " x scale:" << x_q_param.at(0).scale
               << " w scale size:" << w_q_params.size();
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
  // copy cnode quant param to dequant
  if (cnode_primitive->HasAttr(quant::kQuantParam)) {
    prim->AddAttr(quant::kQuantParam, cnode_primitive->GetAttr(quant::kQuantParam));
  }
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

int InsertQuantNodeManager::AdjustTransposeNodeForSingleMatMulNode(const FuncGraphPtr &func_graph,
                                                                   const CNodePtr &cnode) {
  const std::set<PrimitivePtr> support_transpose_types = {prim::kPrimMatMulFusion, prim::kPrimMatMul,
                                                          prim::kPrimBatchMatMul};
  if (!CheckNodeInSet(cnode, support_transpose_types)) {
    return RET_OK;
  }
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
  return RET_OK;
}

int InsertQuantNodeManager::AdjustTransposeNodeForMatMul(const FuncGraphPtr &func_graph) {
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto ret = AdjustTransposeNodeForSingleMatMulNode(func_graph, cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " Adjust Transpose Node failed.";
      return ret;
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
