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
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
constexpr const int kSwitchTruePartialIndex = 2;
constexpr const int kSwitchFalsePartialIndex = 3;
constexpr const int kPartialFgVnodeIndex = 1;

bool MindIRControlFlowAdjust::HasCallAfter(const FuncGraphPtr &partial_fg) {
  MS_CHECK_TRUE_MSG(partial_fg != nullptr, false, "partial_fg is nullptr.");
  auto output_node = partial_fg->output();
  return IsCall(output_node);
}

std::vector<AnfNodePtr> MindIRControlFlowAdjust::GetFgOutput(const FuncGraphPtr &fg) {
  MS_CHECK_TRUE_MSG(fg != nullptr, {}, "fg is nullptr.");
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
    return {output_node};
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
  MS_CHECK_TRUE_MSG(fg != nullptr, RET_NULL_PTR, "fg is nullptr.");
  MS_CHECK_TRUE_MSG(after_fg != nullptr, RET_NULL_PTR, "after_fg is nullptr.");

  // create after partial node
  ValueNodePtr after_partial_anf_primitive = GetPartialFusionPrim();
  if (after_partial_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetPartialFusionPrim failed.";
    return RET_FAILED;
  }
  auto after_value_node = NewValueNode(after_fg);
  MS_CHECK_TRUE_MSG(after_value_node != nullptr, RET_NULL_PTR, "Failed to create value node.");
  std::vector<AnfNodePtr> after_partial_cnode_inputs{after_partial_anf_primitive, after_value_node};

  if (!opt::CheckPrimitiveType(fg->output(), prim::kPrimMakeTuple)) {
    after_partial_cnode_inputs.push_back(fg->output());
  } else {
    auto then_fg_output = fg->output()->cast<CNodePtr>();
    MS_ASSERT(then_fg_output != nullptr);
    for (size_t i = 1; i < then_fg_output->inputs().size(); ++i) {
      after_partial_cnode_inputs.push_back(then_fg_output->input(i));
    }
    fg->DropNode(then_fg_output);
  }

  // insert partial node
  auto after_partial_cnode = fg->NewCNode(after_partial_cnode_inputs);
  MS_CHECK_TRUE_MSG(after_partial_cnode != nullptr, RET_NULL_PTR, "Failed to create C node.");
  auto after_fg_name = after_fg->get_attr("graph_name")->ToString();
  after_partial_cnode->set_fullname_with_scope("partial_" + after_fg_name);

  // insert call node
  std::vector<AnfNodePtr> call_node_inputs{after_partial_cnode};
  auto call_node = fg->NewCNode(call_node_inputs);
  MS_CHECK_TRUE_MSG(call_node != nullptr, RET_NULL_PTR, "Failed to create C node.");
  call_node->set_fullname_with_scope("call_" + after_partial_cnode->fullname_with_scope());
  fg->set_output(call_node);

  return RET_OK;
}

FuncGraphPtr MindIRControlFlowAdjust::AddAfterFuncGraph(const FuncGraphPtr &fg,
                                                        const std::vector<AnfNodePtr> &one_of_inline_fg_output,
                                                        const string &switch_node_name) {
  MS_CHECK_TRUE_MSG(fg != nullptr, nullptr, "fg is nullptr.");
  // create after partial node to call
  auto after_fg = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_MSG(after_fg != nullptr, nullptr, "after_fg is nullptr.");
  auto manager = fg->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, nullptr, "manager is nullptr.");
  manager->AddFuncGraph(after_fg);
  after_fg->set_attr("graph_name", MakeValue(switch_node_name + "_after_fg"));
  after_fg->set_manager(fg->manager());

  int i = 0;
  for (auto &iter : one_of_inline_fg_output) {
    auto new_parameter = after_fg->add_parameter();
    MS_CHECK_TRUE_MSG(new_parameter != nullptr, nullptr, "new_parameter is nullptr.");
    new_parameter->set_name(switch_node_name + ":" + std::to_string(i++));
    new_parameter->set_abstract(iter->abstract());
  }

  if (after_fg->get_inputs().size() > 1) {
    std::vector<AnfNodePtr> make_tuple_inputs = after_fg->get_inputs();
    auto make_tuple_prim_ptr = std::make_shared<lite::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return nullptr;
    }
    auto make_tuple_prim = NewValueNode(make_tuple_prim_ptr);
    MS_CHECK_TRUE_MSG(make_tuple_prim != nullptr, nullptr, "Failed to create value node.");
    make_tuple_inputs.insert(make_tuple_inputs.begin(), make_tuple_prim);
    auto make_tuple_cnode = after_fg->NewCNode(make_tuple_inputs);
    MS_CHECK_TRUE_MSG(make_tuple_cnode != nullptr, nullptr, "Failed to create C node.");
    make_tuple_cnode->set_fullname_with_scope("return tuple");
    auto return_prim_ptr = std::make_shared<lite::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return nullptr;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    MS_CHECK_TRUE_MSG(value_node != nullptr, nullptr, "Failed to create value node.");
    std::vector<AnfNodePtr> op_inputs = {value_node, make_tuple_cnode};
    auto cnode = after_fg->NewCNode(op_inputs);
    MS_CHECK_TRUE_MSG(cnode != nullptr, nullptr, "Failed to create C node.");
    cnode->set_fullname_with_scope("Return");
    after_fg->set_return(cnode);
  } else {
    auto return_prim_ptr = std::make_shared<lite::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return nullptr;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    MS_CHECK_TRUE_MSG(value_node != nullptr, nullptr, "Failed to create value node.");
    std::vector<AnfNodePtr> op_inputs{value_node, after_fg->get_inputs().front()};
    auto return_cnode = after_fg->NewCNode(op_inputs);
    MS_CHECK_TRUE_MSG(return_cnode != nullptr, nullptr, "Failed to create C node.");
    return_cnode->set_fullname_with_scope("Return");
    after_fg->set_return(return_cnode);
  }
  return after_fg;
}

CNodePtr MindIRControlFlowAdjust::GetMainFgSwitchNode(const FuncGraphPtr &fg) {
  MS_CHECK_TRUE_MSG(fg != nullptr, nullptr, "fg is nullptr.");
  auto node_list = TopoSort(fg->get_return());
  for (auto &node : node_list) {
    if (IsSwitch(node)) {
      return node->cast<CNodePtr>();
    }
  }
  return nullptr;
}

int MindIRControlFlowAdjust::AddAfterFgForInlinedFg(const std::set<FuncGraphPtr> &all_func_graphs,
                                                    const FuncGraphPtr &main_fg) {
  auto switch_cnode = GetMainFgSwitchNode(main_fg);
  if (switch_cnode == nullptr) {
    MS_LOG(DEBUG) << "not a control flow model.";
    return RET_OK;
  }

  // get all inline fg
  std::vector<FuncGraphPtr> all_inline_fgs{};
  for (auto &graph : all_func_graphs) {
    if (HasCallAfter(graph)) {
      continue;
    }
    all_inline_fgs.push_back(graph);
  }

  // checkout all inline fg
  if (all_inline_fgs.empty()) {
    MS_LOG(ERROR) << "graph is not right.";
    return RET_ERROR;
  }

  auto first_fg_output = GetFgOutput(all_inline_fgs.front());
  auto inline_fg_output_size = first_fg_output.size();
  for (auto &graph : all_inline_fgs) {
    if (graph == nullptr) {
      MS_LOG(ERROR) << "GetPartialFg failed.";
      return RET_NULL_PTR;
    }
    if (GetFgOutput(graph).size() != inline_fg_output_size) {
      MS_LOG(ERROR) << "graph is not right, inline fg output size is not same.";
      return RET_ERROR;
    }
  }

  auto after_fg = AddAfterFuncGraph(main_fg, first_fg_output, switch_cnode->fullname_with_scope());
  if (after_fg == nullptr) {
    MS_LOG(ERROR) << "AddAfterFuncGraph failed.";
    return RET_ERROR;
  }

  for (auto &graph : all_inline_fgs) {
    if (ModifyFgToCallAfterFg(graph, after_fg) != RET_OK) {
      MS_LOG(ERROR) << "inline fg add call after fg failed.";
      return RET_ERROR;
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
      MS_ASSERT(call_node != nullptr);
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
      MS_CHECK_TRUE_MSG(partial_cnode != nullptr, RET_NULL_PTR, "Failed to create C node.");
      partial_cnode->set_fullname_with_scope("partial_" + call_cnode->fullname_with_scope());
      call_cnode->set_inputs({partial_cnode});
    }
  }
  return RET_OK;
}

int MindIRControlFlowAdjust::ResetFuncGraph(const FuncGraphPtr &fg, std::set<FuncGraphPtr> all_func_graphs) {
  MS_CHECK_TRUE_MSG(fg != nullptr, RET_NULL_PTR, "fg is nullptr.");
  auto manager = fg->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_NULL_PTR, "manager is nullptr.");
  manager->Clear();
  manager->AddFuncGraph(fg, true);
  for (auto &item : all_func_graphs) {
    if (item == fg) {
      continue;
    }
    manager->AddFuncGraph(item);
  }
  return RET_OK;
}

bool MindIRControlFlowAdjust::Run(const FuncGraphPtr &func_graph) {
  if (this->fmk_type_ != FmkType::kFmkTypeMs) {
    MS_LOG(INFO) << "The framework type of model should be MindIR.";
    return lite::RET_OK;
  }
  MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "func_graph is nullptr.");
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
  ret = AddAfterFgForInlinedFg(all_func_graphs, func_graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AddAfterFgForInlinedFg failed.";
    return false;
  }
  ret = ResetFuncGraph(func_graph, all_func_graphs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ResetFuncGraph failed.";
    return false;
  }

  if (status_ != RET_OK) {
    return false;
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore
