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
#include "ops/make_tuple.h"
#include "ops/return.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "src/common/log_adapter.h"
#include "tools/common/node_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/core_ops.h"
namespace {
constexpr const int kSwitchTruePartialIndex = 2;
constexpr const int kSwitchFalsePartialIndex = 3;
constexpr const int kSwitchInputSize = 4;
constexpr const int kSwitchLayerInputSize = 3;
constexpr const int kSwitchLayerMakeTupleIndex = 2;
}  // namespace

namespace mindspore {
namespace lite {
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
                                                        const std::vector<AnfNodePtr> &one_of_inline_fg_output) {
  MS_CHECK_TRUE_MSG(fg != nullptr, nullptr, "fg is nullptr.");
  // create after partial node to call
  auto after_fg = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_MSG(after_fg != nullptr, nullptr, "after_fg is nullptr.");
  auto manager = fg->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, nullptr, "manager is nullptr.");
  manager->AddFuncGraph(after_fg);
  std::string after_fg_name = "after_fg";
  after_fg->set_attr("graph_name", MakeValue(after_fg_name));
  after_fg->set_manager(fg->manager());

  int i = 0;
  for (auto &iter : one_of_inline_fg_output) {
    auto new_parameter = after_fg->add_parameter();
    MS_CHECK_TRUE_MSG(new_parameter != nullptr, nullptr, "new_parameter is nullptr.");
    new_parameter->set_name(after_fg_name + ":" + std::to_string(i++));
    new_parameter->set_abstract(iter->abstract());
  }

  if (after_fg->get_inputs().size() > 1) {
    std::vector<AnfNodePtr> make_tuple_inputs = after_fg->get_inputs();
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return nullptr;
    }
    auto make_tuple_prim_c = make_tuple_prim_ptr->GetPrim();
    MS_CHECK_TRUE_MSG(make_tuple_prim_c != nullptr, nullptr, "Failed to create make_tuple_prim_c.");
    auto make_tuple_prim = NewValueNode(make_tuple_prim_c);
    MS_CHECK_TRUE_MSG(make_tuple_prim != nullptr, nullptr, "Failed to create value node.");
    make_tuple_inputs.insert(make_tuple_inputs.begin(), make_tuple_prim);
    auto make_tuple_cnode = after_fg->NewCNode(make_tuple_inputs);
    MS_CHECK_TRUE_MSG(make_tuple_cnode != nullptr, nullptr, "Failed to create C node.");
    make_tuple_cnode->set_fullname_with_scope("return tuple");
    auto return_prim_ptr = std::make_shared<ops::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return nullptr;
    }
    auto return_prim_c = return_prim_ptr->GetPrim();
    MS_CHECK_TRUE_MSG(return_prim_c != nullptr, nullptr, "Failed to create return_prim_c.");
    auto value_node = NewValueNode(return_prim_c);
    MS_CHECK_TRUE_MSG(value_node != nullptr, nullptr, "Failed to create value node.");
    std::vector<AnfNodePtr> op_inputs = {value_node, make_tuple_cnode};
    auto cnode = after_fg->NewCNode(op_inputs);
    MS_CHECK_TRUE_MSG(cnode != nullptr, nullptr, "Failed to create C node.");
    cnode->set_fullname_with_scope("Return");
    after_fg->set_return(cnode);
  } else {
    auto return_prim_ptr = std::make_shared<ops::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return nullptr;
    }
    auto return_prim_c = return_prim_ptr->GetPrim();
    MS_CHECK_TRUE_MSG(return_prim_c != nullptr, nullptr, "Failed to create return_prim_c.");
    auto value_node = NewValueNode(return_prim_c);
    MS_CHECK_TRUE_MSG(value_node != nullptr, nullptr, "Failed to create value node.");
    std::vector<AnfNodePtr> op_inputs{value_node, after_fg->get_inputs().front()};
    auto return_cnode = after_fg->NewCNode(op_inputs);
    MS_CHECK_TRUE_MSG(return_cnode != nullptr, nullptr, "Failed to create C node.");
    return_cnode->set_fullname_with_scope("Return");
    after_fg->set_return(return_cnode);
  }
  return after_fg;
}

int MindIRControlFlowAdjust::MoveCallInputsToPartialFusionInputs(const std::set<FuncGraphPtr> &all_func_graphs) {
  for (auto &graph : all_func_graphs) {
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (!IsCall(node)) {
        continue;
      }
      auto call_cnode = node->cast<CNodePtr>();
      MS_ASSERT(call_node != nullptr);
      auto call_cnode_inputs = call_cnode->inputs();
      if (call_cnode_inputs.size() == 1) {
        MS_LOG(DEBUG) << "no need move call inputs.";
        continue;
      }
      auto call_first_input = call_cnode->input(0);
      if (!utils::isa<CNodePtr>(call_first_input)) {
        // This situation will be handled in the InsertPartialFusionForRawCall function
        continue;
      }
      auto call_first_input_cnode = call_first_input->cast<CNodePtr>();
      MS_ASSERT(call_first_input_cnode != nullptr);
      if (IsPartialFusion(call_first_input_cnode)) {
        auto partial_cnode_inputs = call_first_input_cnode->inputs();
        (void)std::copy(call_cnode_inputs.begin() + 1, call_cnode_inputs.end(),
                        std::back_inserter(partial_cnode_inputs));
        call_first_input_cnode->set_inputs(partial_cnode_inputs);
      }

      if (IsSwitch(call_first_input_cnode)) {
        auto switch_cnode_inputs = call_first_input_cnode->inputs();
        if (switch_cnode_inputs.size() == kSwitchInputSize) {
          MS_LOG(ERROR) << "switch op inputs size not right.";
          return RET_ERROR;
        }
        if (!IsPartialFusion(switch_cnode_inputs[kSwitchTruePartialIndex]) ||
            !IsPartialFusion(switch_cnode_inputs[kSwitchFalsePartialIndex])) {
          MS_LOG(ERROR) << "switch inputs not are partial ops, not support now.";
          return RET_NOT_SUPPORT;
        }

        auto true_partial_cnode = switch_cnode_inputs.at(kSwitchTruePartialIndex)->cast<CNodePtr>();
        auto true_partial_cnode_inputs = true_partial_cnode->inputs();
        (void)std::copy(call_cnode_inputs.begin() + 1, call_cnode_inputs.end(),
                        std::back_inserter(true_partial_cnode_inputs));
        true_partial_cnode->set_inputs(true_partial_cnode_inputs);

        auto false_partial_cnode = switch_cnode_inputs.at(kSwitchFalsePartialIndex)->cast<CNodePtr>();
        auto false_partial_cnode_inputs = false_partial_cnode->inputs();
        (void)std::copy(call_cnode_inputs.begin() + 1, call_cnode_inputs.end(),
                        std::back_inserter(false_partial_cnode_inputs));
        false_partial_cnode->set_inputs(false_partial_cnode_inputs);
      }

      if (IsSwitchLayer(call_first_input_cnode)) {
        auto switch_layer_cnode_inputs = call_first_input_cnode->inputs();
        if (switch_layer_cnode_inputs.size() != kSwitchLayerInputSize) {
          MS_LOG(ERROR) << "switch layer op inputs size not right.";
          return RET_ERROR;
        }
        if (!IsMakeTuple(switch_layer_cnode_inputs[kSwitchLayerMakeTupleIndex])) {
          MS_LOG(ERROR) << "SwitchLayer op last input not is MakeTuple ops, not support now.";
          return RET_NOT_SUPPORT;
        }
        auto make_tuple_op = switch_layer_cnode_inputs[kSwitchLayerMakeTupleIndex]->cast<CNodePtr>();
        auto make_tuple_op_intpus = make_tuple_op->inputs();
        for (size_t i = 1; i < make_tuple_op_intpus.size(); i++) {
          if (IsPartialFusion(make_tuple_op_intpus[i])) {
            auto partial_node = make_tuple_op_intpus[i]->cast<CNodePtr>();
            auto partial_node_inputs = partial_node->inputs();
            std::copy(call_cnode_inputs.begin() + 1, call_cnode_inputs.end(), std::back_inserter(partial_node_inputs));
            partial_node->set_inputs(partial_node_inputs);
            continue;
          }
          if (!utils::isa<ValueNodePtr>(make_tuple_op_intpus[i])) {
            MS_LOG(ERROR)
              << "switch layer op make tuple inputs not is partial fusion op or function graph, not support now.";
            return RET_NOT_SUPPORT;
          }
          auto make_tuple_op_value_input = make_tuple_op_intpus[i]->cast<ValueNodePtr>();
          if (GetValueNode<FuncGraphPtr>(make_tuple_op_value_input) == nullptr) {
            MS_LOG(ERROR)
              << "switch layer op make tuple inputs not is partial fusion op or function graph, not support now.";
            return RET_NOT_SUPPORT;
          }
          std::vector<AnfNodePtr> partial_cnode_inputs = {lite::GetPartialFusionPrim(), make_tuple_op_value_input};
          (void)std::copy(call_cnode_inputs.begin() + 1, call_cnode_inputs.end(),
                          std::back_inserter(partial_cnode_inputs));
          auto partial_cnode = graph->NewCNode(partial_cnode_inputs);
          MS_CHECK_TRUE_MSG(partial_cnode != nullptr, RET_NULL_PTR, "Failed to create C node.");
          partial_cnode->set_fullname_with_scope("partial_" + make_tuple_op->fullname_with_scope() + "_" +
                                                 std::to_string(i));
          make_tuple_op->set_input(i, partial_cnode);
        }
      }
      call_cnode->set_inputs({call_first_input_cnode});
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
      auto call_first_input = call_cnode->input(0);
      if (!utils::isa<ValueNodePtr>(call_first_input)) {
        continue;
      }
      if (GetValueNode<FuncGraphPtr>(call_first_input->cast<ValueNodePtr>()) == nullptr) {
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
    return true;
  }
  MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "func_graph is nullptr.");
  std::set<FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraph(func_graph, &all_func_graphs);
  if (all_func_graphs.size() == 1) {
    MS_LOG(INFO) << "Not is control flow model.";
    return true;
  }
  int ret = MoveCallInputsToPartialFusionInputs(all_func_graphs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MoveCallInputsToPartialFusionInputs failed.";
    return false;
  }
  ret = InsertPartialFusionForRawCall(all_func_graphs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InsertPartialFusionForRawCall failed.";
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
