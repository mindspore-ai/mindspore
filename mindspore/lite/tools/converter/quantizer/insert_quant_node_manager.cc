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

#include "mindspore/lite/tools/converter/quantizer/insert_quant_node_manager.h"
#include <memory>
#include <set>
#include <vector>
#include <string>
#include "ops/quant_dtype_cast.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/common/node_util.h"

namespace mindspore::lite::quant {
namespace {
constexpr size_t kMinSize3 = 3;
constexpr size_t kPrimitiveCOffset = 1;
}  // namespace
ValueNodePtr InsertQuantNodeManager::NewQuantCastValueNode(int src_type, int dst_type,
                                                           const std::vector<schema::QuantParamT> &quant_params) {
  auto prim_c = std::make_shared<ops::QuantDTypeCast>();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr.");
  prim_c->Init(src_type, dst_type);
  auto quant_params_holder = std::make_shared<QuantParamHolder>(quant_params.size(), quant_params.size());
  MS_CHECK_TRUE_MSG(quant_params_holder != nullptr, nullptr, "quant_params_holder is nullptr.");
  quant_params_holder->set_quant_type(schema::QuantType_QUANT_ALL);
  for (size_t i = 0; i < quant_params.size(); i++) {
    auto quant_param = quant_params[i];
    std::vector<schema::QuantParamT> quant_params_in = {quant_param};
    quant_params_holder->set_input_quant_param(i, quant_params_in);
    quant_params_holder->set_output_quant_param(i, quant_params_in);
  }
  prim_c->AddAttr("quant_params", quant_params_holder);
  return NewValueNode(prim_c);
}

int InsertQuantNodeManager::InsertCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode, size_t input_index,
                                           bool is_graph_input) {
  auto primitive = GetValueNode<std::shared_ptr<mindspore::Primitive>>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto input_node = cnode->input(input_index);
  auto input_cnode_quant_type = schema::QuantType_QUANT_NONE;
  std::shared_ptr<mindspore::Primitive> input_cnode_primitive_c = nullptr;
  if (!is_graph_input) {
    auto input_cnode = std::dynamic_pointer_cast<mindspore::CNode>(input_node);
    input_cnode_primitive_c = GetValueNode<std::shared_ptr<mindspore::Primitive>>(input_cnode->input(0));
    if (input_cnode_primitive_c == nullptr) {
      MS_LOG(DEBUG) << "input: " << input_index << " " << input_cnode->fullname_with_scope() << ": "
                    << " PrimitiveC is null";
      return RET_NO_CHANGE;
    }
    auto input_primitive_quant_holder = GetCNodeQuantHolder(input_cnode_primitive_c);
    MS_CHECK_TRUE_MSG(input_primitive_quant_holder != nullptr, RET_NULL_PTR,
                      "input_primitive_quant_holder is nullptr.");
    input_cnode_quant_type = input_primitive_quant_holder->quant_type();
  }
  auto primitive_quant_param_holder = GetCNodeQuantHolder(primitive);
  MS_CHECK_TRUE_MSG(primitive_quant_param_holder != nullptr, RET_NULL_PTR, "primitive_quant_param_holder is nullptr.");
  auto curnode_quant_type = primitive_quant_param_holder->quant_type();
  if (curnode_quant_type == input_cnode_quant_type) {
    return RET_NO_CHANGE;
  }

  bool insert_dequant_node =
    curnode_quant_type == schema::QuantType_QUANT_ALL && input_cnode_quant_type == schema::QuantType_QUANT_NONE;
  bool insert_quant_node =
    curnode_quant_type == schema::QuantType_QUANT_NONE && input_cnode_quant_type == schema::QuantType_QUANT_ALL;
  if (!insert_dequant_node && !insert_quant_node) {
    MS_LOG(WARNING) << "value_node is null! "
                    << "cur_node: " << cnode->fullname_with_scope() << " quant_type: "
                    << " input_" << input_index << ": "
                    << " quant_type:" << input_cnode_quant_type;
    return RET_NO_CHANGE;
  }
  ValueNodePtr value_node;
  if (insert_dequant_node) {
    auto curr_primitive_quant_param_holder = GetCNodeQuantHolder(primitive);
    if (curr_primitive_quant_param_holder->get_input_quant_params().size() < input_index) {
      MS_LOG(ERROR) << "quant param is invalid.";
      return RET_ERROR;
    }
    value_node = NewQuantCastValueNode(kNumberTypeFloat32, kNumberTypeInt8,
                                       curr_primitive_quant_param_holder->get_input_quant_params()[input_index - 1]);
  } else {  // insert_quant_node
    auto input_primitive_quant_param_holder = GetCNodeQuantHolder(input_cnode_primitive_c);
    if (input_primitive_quant_param_holder->get_output_quant_params().empty()) {
      MS_LOG(ERROR) << "output quant param is empty.";
      return RET_ERROR;
    }
    value_node = NewQuantCastValueNode(kNumberTypeInt8, kNumberTypeFloat32,
                                       input_primitive_quant_param_holder->get_output_quant_params().front());
  }
  std::vector<AnfNodePtr> op_inputs = {value_node, input_node};
  auto quant_cast_cnode = graph->NewCNode(op_inputs);
  MS_CHECK_TRUE_MSG(quant_cast_cnode != nullptr, RET_NULL_PTR, "quant_cast_cnode is nullptr.");
  quant_cast_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_quant_cast_" +
                                            std::to_string(input_index));
  cnode->set_input(input_index, quant_cast_cnode);
  return RET_OK;
}

int InsertQuantNodeManager::CheckDataType(const AnfNodePtr &input_node, TypeId check_type_id) {
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
      MS_LOG(ERROR) << "Fetch DataType from cnode failed.";
      return ret;
    }
    if (type_id != check_type_id) {
      return RET_NO_CHANGE;
    }
  }
  return RET_OK;
}

int InsertQuantNodeManager::InsertQuantDtypeCastNode(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto cnodes = graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      auto input_node = cnode->input(i);
      auto ret = CheckDataType(input_node, kNumberTypeFloat32);
      if (ret == RET_NO_CHANGE) {
        continue;
      } else if (ret != RET_OK) {
        MS_LOG(ERROR) << "Check data type failed.";
        return ret;
      }
      bool is_graph_input = IsGraphInput(input_node);
      ret = InsertCastNode(graph, cnode, i, is_graph_input);
      if (ret == RET_NO_CHANGE) {
        continue;
      } else if (ret != RET_OK) {
        MS_LOG(ERROR) << "Insert cast node failed.";
        return ret;
      }
    }
  }
  return RET_OK;
}

int InsertQuantNodeManager::InsertDynamicQuantWithIndex(const FuncGraphPtr &graph, const CNodePtr &cnode,
                                                        size_t index) {
  auto primitive_c = std::make_shared<ops::DynamicQuant>();
  primitive_c->set_dst_type(dst_type_);
  primitive_c->set_symmetric(symmetric_);
  auto dynamic_quant_cnode = graph->NewCNode(primitive_c, {cnode->input(index)});
  auto name = cnode->fullname_with_scope() + "_dynamic_cast_node_" + to_string(index);
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
    InsertDynamicQuantWithIndex(graph, cnode, kInputIndex + kPrimitiveCOffset);
  }
  auto weight = cnode->input(kWeightIndex + kPrimitiveCOffset);
  if (weight->isa<mindspore::CNode>() || IsGraphInput(weight)) {
    InsertDynamicQuantWithIndex(graph, cnode, kWeightIndex + kPrimitiveCOffset);
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
}  // namespace mindspore::lite::quant
