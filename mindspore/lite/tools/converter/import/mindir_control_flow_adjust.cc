/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "tools/converter/import/mindir_control_flow_adjust.h"
#include <vector>
#include <memory>
#include <set>
#include <algorithm>
#include "tools/converter/ops/ops_def.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/quant_param_holder.h"
#include "src/common/log_adapter.h"
#include "tools/common/node_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace lite {
constexpr const int kSwitchTruePartialIndex = 2;
constexpr const int kSwitchFalsePartialIndex = 3;
constexpr const int kPartialFgVnodeIndex = 1;

FuncGraphPtr MindIRControlFlowAdjust::GetPartialFg(const CNodePtr &partial_node) {
  auto fg_vnode = partial_node->input(kPartialFgVnodeIndex)->cast<ValueNodePtr>();
  if (fg_vnode == nullptr) {
    MS_LOG(ERROR) << "fg is not right.";
    status_ = RET_ERROR;
    return nullptr;
  }
  auto partial_fg = GetValueNode<FuncGraphPtr>(fg_vnode);
  if (partial_fg == nullptr) {
    MS_LOG(ERROR) << "partial_fg is nullptr.";
    status_ = RET_NULL_PTR;
    return nullptr;
  }
  return partial_fg;
}

bool MindIRControlFlowAdjust::HasCallAfter(const CNodePtr &partial_node) {
  auto partial_fg = GetPartialFg(partial_node);
  if (partial_fg == nullptr) {
    MS_LOG(ERROR) << "GetPartialFg failed.";
    status_ = RET_NULL_PTR;
    return false;
  }
  auto output_node = partial_fg->output();
  return IsCall(output_node);
}

std::vector<AnfNodePtr> MindIRControlFlowAdjust::GetFgOutput(const FuncGraphPtr &fg) {
  std::vector<AnfNodePtr> ret{};
  auto output_node = fg->output();
  if (output_node == nullptr) {
    MS_LOG(ERROR) << "graph is not right.";
    status_ = RET_NULL_PTR;
    return {};
  }
  auto output_cnode = output_node->cast<CNodePtr>();
  if (output_cnode == nullptr) {
    MS_LOG(INFO) << "graph output is not cnode.";
    return {};
  }
  if (!IsMakeTuple(output_node)) {
    MS_LOG(INFO) << "graph is single output.";
    return {output_node};
  }
  for (size_t i = 1; i < output_cnode->inputs().size(); ++i) {
    ret.push_back(output_cnode->input(i));
  }
  return ret;
}

int MindIRControlFlowAdjust::ModifyFgToCallAfterFg(const FuncGraphPtr &fg, const FuncGraphPtr &after_fg) {
  // create after partial node
  ValueNodePtr after_partial_anf_primitive = GetPartialFusionPrim();
  if (after_partial_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetPartialFusionPrim failed.";
    return RET_FAILED;
  }
  auto after_value_node = NewValueNode(after_fg);
  std::vector<AnfNodePtr> after_partial_cnode_inputs{after_partial_anf_primitive, after_value_node};

  if (!opt::CheckPrimitiveType(fg->output(), prim::kPrimMakeTuple)) {
    after_partial_cnode_inputs.push_back(fg->output());
  } else {
    auto then_fg_output = fg->output()->cast<CNodePtr>();
    for (size_t i = 1; i < then_fg_output->inputs().size(); ++i) {
      after_partial_cnode_inputs.push_back(then_fg_output->input(i));
    }
    fg->DropNode(then_fg_output);
  }

  // insert partial node
  auto after_partial_cnode = fg->NewCNode(after_partial_cnode_inputs);
  auto after_fg_name = after_fg->get_attr("graph_name")->ToString();
  after_partial_cnode->set_fullname_with_scope("partial_" + after_fg_name);

  // insert call node
  std::vector<AnfNodePtr> call_node_inputs{after_partial_cnode};
  auto call_node = fg->NewCNode(call_node_inputs);
  call_node->set_fullname_with_scope("call_" + after_partial_cnode->fullname_with_scope());
  fg->set_output(call_node);

  return RET_OK;
}

int MindIRControlFlowAdjust::AddAfterFuncGraph(const FuncGraphPtr &fg, const CNodePtr &true_partial_node,
                                               const CNodePtr &false_partial_node) {
  auto true_partial_fg = GetPartialFg(true_partial_node);
  auto false_partial_fg = GetPartialFg(false_partial_node);
  if (true_partial_node == nullptr || false_partial_node == nullptr) {
    MS_LOG(ERROR) << "GetPartialFg failed.";
    return RET_NULL_PTR;
  }

  // check nums of two fg output size
  auto true_partial_fg_output = GetFgOutput(true_partial_fg);
  auto false_partial_fg_output = GetFgOutput(false_partial_fg);
  if (true_partial_fg_output.empty() || false_partial_fg_output.empty() ||
      true_partial_fg_output.size() != false_partial_fg_output.size()) {
    MS_LOG(ERROR) << "graph is not right.";
    return RET_ERROR;
  }

  // create after partial node to call
  auto after_fg = std::make_shared<FuncGraph>();
  auto manager = fg->manager();
  manager->AddFuncGraph(after_fg);
  after_fg->set_attr("graph_name", MakeValue(false_partial_node->fullname_with_scope() + "_after_fg"));
  after_fg->set_manager(fg->manager());

  for (auto &iter : true_partial_fg_output) {
    auto new_parameter = after_fg->add_parameter();
    new_parameter->set_name(iter->fullname_with_scope() + "_after_parameter");
    new_parameter->set_abstract(iter->abstract());
  }

  if (after_fg->get_inputs().size() > 1) {
    std::vector<AnfNodePtr> make_tuple_inputs = after_fg->get_inputs();
    auto make_tuple_prim_ptr = std::make_shared<lite::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return RET_NULL_PTR;
    }
    auto make_tuple_prim = NewValueNode(make_tuple_prim_ptr);
    make_tuple_inputs.insert(make_tuple_inputs.begin(), make_tuple_prim);
    auto make_tuple_cnode = after_fg->NewCNode(make_tuple_inputs);
    make_tuple_cnode->set_fullname_with_scope("return tuple");

    auto return_prim_ptr = std::make_shared<lite::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    std::vector<AnfNodePtr> op_inputs = {value_node, make_tuple_cnode};
    auto cnode = after_fg->NewCNode(op_inputs);
    cnode->set_fullname_with_scope("Return");
    after_fg->set_return(cnode);
  } else {
    auto return_prim_ptr = std::make_shared<lite::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    std::vector<AnfNodePtr> op_inputs{value_node, after_fg->get_inputs().front()};
    auto return_cnode = after_fg->NewCNode(op_inputs);
    return_cnode->set_fullname_with_scope("Return");
    after_fg->set_return(return_cnode);
  }

  int ret = ModifyFgToCallAfterFg(true_partial_fg, after_fg);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "true partial fg add call after fg failed.";
    return ret;
  }
  ret = ModifyFgToCallAfterFg(false_partial_fg, after_fg);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "false partial fg add call after fg failed.";
    return ret;
  }

  return RET_OK;
}

int MindIRControlFlowAdjust::AddAfterFgForInlinedFg(const std::set<FuncGraphPtr> &all_func_graphs) {
  for (auto &graph : all_func_graphs) {
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (!IsSwitch(node)) {
        continue;
      }
      auto switch_cnode = node->cast<CNodePtr>();
      auto true_partial_node = switch_cnode->input(kSwitchTruePartialIndex)->cast<CNodePtr>();
      auto false_partial_node = switch_cnode->input(kSwitchFalsePartialIndex)->cast<CNodePtr>();

      if (!IsPartialFusion(true_partial_node) || !IsPartialFusion(false_partial_node)) {
        MS_LOG(ERROR) << "graph is not right";
        return RET_ERROR;
      }

      if (HasCallAfter(true_partial_node) || HasCallAfter(false_partial_node)) {
        continue;
      }

      int ret = AddAfterFuncGraph(graph, true_partial_node, false_partial_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "AddAfterFuncGraph failed.";
        return ret;
      }
    }
  }
  return RET_OK;
}

int MindIRControlFlowAdjust::InsertPartialFusionForRawCall(const std::set<FuncGraphPtr> &all_func_graphs) {
  for (auto &graph : all_func_graphs) {
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (!IsCall(node)) {
        continue;
      }
      auto call_cnode = node->cast<CNodePtr>();
      auto call_cnode_inputs = call_cnode->inputs();
      auto cnode_first_input = call_cnode->input(0);
      if (!utils::isa<ValueNodePtr>(cnode_first_input)) {
        continue;
      }
      if (GetValueNode<FuncGraphPtr>(cnode_first_input->cast<ValueNodePtr>()) == nullptr) {
        continue;
      }

      std::vector<AnfNodePtr> partial_cnode_inputs = {lite::GetPartialFusionPrim()};
      std::copy(call_cnode_inputs.begin(), call_cnode_inputs.end(), std::back_inserter(partial_cnode_inputs));
      auto partial_cnode = graph->NewCNode(partial_cnode_inputs);
      partial_cnode->set_fullname_with_scope("partial_" + call_cnode->fullname_with_scope());

      call_cnode->set_inputs({partial_cnode});
    }
  }
  return RET_OK;
}

bool MindIRControlFlowAdjust::Run(const FuncGraphPtr &func_graph) {
  if (this->fmk_type_ != FmkType::kFmkTypeMs) {
    MS_LOG(INFO) << "The framework type of model should be MindIR.";
    return lite::RET_OK;
  }
  MS_ASSERT(graph != nullptr);
  std::set<FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraph(func_graph, &all_func_graphs);
  if (all_func_graphs.size() == 1) {
    MS_LOG(INFO) << "Not is control flow model.";
    return true;
  }
  int ret = InsertPartialFusionForRawCall(all_func_graphs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InsertPartialFusionForRawCall failed.";
    return false;
  }
  ret = AddAfterFgForInlinedFg(all_func_graphs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AddAfterFgForInlinedFg failed.";
    return false;
  }
  if (status_ != RET_OK) {
    return false;
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore
