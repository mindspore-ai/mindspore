/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/free_variables_eliminate.h"

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include "ir/func_graph_cloner.h"
#include "frontend/jit/ps/action.h"
#include "frontend/optimizer/irpass/view_inplace_utils.h"
#include "include/frontend/jit/ps/action_interface.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace {
// %0 = partial(func, arg1, arg2, ...)
// %1 = J(%0)
// %2 = %1(arg_a, arg_b, ...)
// -->
// %0 = func
// %1 = J(%0)
// %2 = %1(arg1, arg2, ...arg_a, arg_b, ...)
void PartialJCallOptPass(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  for (const auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsPrimitiveCNode(cnode->input(0), prim::kPrimJ)) {
      continue;
    }
    auto j_node = cnode->input(0)->cast<CNodePtr>();
    auto j_partial_input = j_node->input(1);
    std::vector<AnfNodePtr> args;
    AnfNodePtr func_node = nullptr;
    if (IsPrimitiveCNode(j_partial_input, prim::kPrimPartial)) {
      const auto &partial_inputs = j_partial_input->cast<CNodePtr>()->inputs();
      func_node = partial_inputs[1];
      // partial(func, arg1, arg2, ...)
      for (size_t index = 2; index < partial_inputs.size(); ++index) {
        args.push_back(partial_inputs[index]);
      }
    }
    const auto &j_caller_inputs = cnode->inputs();
    for (size_t index = 1; index < j_caller_inputs.size(); ++index) {
      args.push_back(j_caller_inputs[index]);
    }
    if (func_node == nullptr) {
      continue;
    }
    j_node->set_input(1, func_node);
    std::vector<AnfNodePtr> new_j_caller_inputs{j_node};
    (void)new_j_caller_inputs.insert(new_j_caller_inputs.end(), args.begin(), args.end());
    auto new_j_caller = func_graph->NewCNode(new_j_caller_inputs);
    new_j_caller->set_abstract(cnode->abstract());
    MS_LOG(DEBUG) << "new_j_caller:" << new_j_caller->DebugString();
    manager->Replace(cnode, new_j_caller);
  }
}

FuncGraphPtr PartialEliminateOptPass(const pipeline::ResourcePtr &resource, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(resource);

  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig partial_eliminate_opt_ =
    opt::OptPassConfig({irpass.partial_eliminate_, irpass.switch_partial_eliminater_,
                        irpass.switch_conditional_partial_eliminater_, irpass.switch_layer_partial_eliminater_});
  opt::OptPassGroupMap map({{"partial_eliminate_", partial_eliminate_opt_}});

  auto after_lift_opt = opt::Optimizer::MakeOptimizer("partial_eliminate", resource, map);

  FuncGraphPtr opt_fg = nullptr;
  ProfileExecute(MsProfile::GetProfile()->Step("partial_eliminate_before_grad"),
                 [&after_lift_opt, func_graph, &opt_fg]() { opt_fg = after_lift_opt->step(func_graph, true); });
  PartialJCallOptPass(opt_fg);
  return opt_fg;
}

FuncGraphVector PartialEliminateMulti(const pipeline::ResourceBasePtr &resource, const FuncGraphVector &func_graphs) {
  auto new_res = std::dynamic_pointer_cast<pipeline::Resource>(resource);
  if (new_res == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Parameter resources is not a pipeline::Resource";
  }
  FuncGraphVector opt_fgs;
  for (const auto &func_graph : func_graphs) {
    auto opt_fg = PartialEliminateOptPass(new_res, func_graph);
#ifdef ENABLE_DUMP_IR
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->CanDump(kIntroductory)) {
      DumpIR("after_opt_" + opt_fg->ToString() + ".ir", opt_fg);
    }
#endif
    opt_fgs.push_back(opt_fg);
  }
  return opt_fgs;
}

std::map<std::string, AnfNodePtr> GetParameterMap(const std::vector<AnfNodePtr> &params) {
  std::map<std::string, AnfNodePtr> params_map;
  for (const auto &param : params) {
    auto ref_key_str = GetRefKey(param);
    if (ref_key_str.empty()) {
      continue;
    }
    params_map[ref_key_str] = param;
  }
  return params_map;
}

std::pair<bool, std::vector<bool>> RemoveRedundantParams(const FuncGraphPtr &func_graph,
                                                         const std::map<std::string, AnfNodePtr> &ref_key_nodes) {
  const auto &inner_params = func_graph->parameters();
  AnfNodePtrList new_params;
  std::vector<bool> need_reserved;
  for (auto inner_param : inner_params) {
    auto ref_key_str = GetRefKey(inner_param);
    if (ref_key_str.empty()) {
      new_params.push_back(inner_param);
      need_reserved.push_back(true);
      continue;
    }
    const auto &iter = ref_key_nodes.find(ref_key_str);
    if (inner_param != iter->second ||
        std::find(new_params.begin(), new_params.end(), inner_param) != new_params.end()) {
      // Skip redundant parameter
      need_reserved.push_back(false);
      continue;
    }
    need_reserved.push_back(true);
    new_params.push_back(inner_param);
  }
  if (inner_params.size() != new_params.size()) {
    func_graph->set_parameters(new_params);
    return {true, need_reserved};
  }
  return {false, need_reserved};
}

CNodePtr NewCaller(const CNodePtr &cnode, const std::vector<bool> &need_reserved_flags) {
  std::vector<AnfNodePtr> new_caller_inputs{cnode->input(0)};
  for (size_t index = 0; index < need_reserved_flags.size(); ++index) {
    if (need_reserved_flags[index]) {
      new_caller_inputs.push_back(cnode->input(index + 1));
    }
  }
  auto cur_func = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(cur_func);
  auto new_caller = cur_func->NewCNodeInOrder(new_caller_inputs);
  new_caller->set_abstract(cnode->abstract());
  return new_caller;
}

void RemoveCallerRedundantArgs(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager,
                               mindspore::HashMap<FuncGraphPtr, std::vector<bool>> *func_need_reserved_flags) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &all_nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  for (const auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (IsPrimitiveCNode(cnode->input(0), prim::kPrimSwitch)) {
      auto cnode_caller = cnode->input(0)->cast<CNodePtr>();
      constexpr size_t true_index = 2;
      auto true_func = GetValueNode<FuncGraphPtr>(cnode_caller->input(true_index));
      auto true_iter = func_need_reserved_flags->find(true_func);
      if (true_iter != func_need_reserved_flags->end()) {
        auto need_reserved_flags = true_iter->second;
        if (cnode->inputs().size() - 1 != need_reserved_flags.size()) {
          MS_LOG(EXCEPTION) << "The call switch node is wrong: " << cnode_caller->DebugString();
        }
        auto new_caller = NewCaller(cnode, need_reserved_flags);
        (void)manager->Replace(node, new_caller);
        func_need_reserved_flags->erase(true_iter);
        constexpr size_t false_index = 3;
        auto false_func = GetValueNode<FuncGraphPtr>(cnode_caller->input(false_index));
        auto false_iter = func_need_reserved_flags->find(false_func);
        if (false_iter != func_need_reserved_flags->end()) {
          func_need_reserved_flags->erase(false_iter);
        }
      }
      continue;
    }
    if (!IsValueNode<FuncGraph>(cnode->input(0))) {
      continue;
    }
    auto func = GetValueNode<FuncGraphPtr>(cnode->input(0));
    auto iter = func_need_reserved_flags->find(func);
    if (iter != func_need_reserved_flags->end()) {
      auto need_reserved_flags = iter->second;
      if (cnode->inputs().size() - 1 != need_reserved_flags.size()) {
        continue;
      }
      auto new_caller = NewCaller(cnode, need_reserved_flags);
      (void)manager->Replace(node, new_caller);
    }
  }
}

void MergeParameters(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(optimizer);
  auto manager = optimizer->manager();
  mindspore::HashMap<FuncGraphPtr, std::vector<bool>> func_need_reserved_flags;
  const auto fg_used_total = func_graph->func_graphs_used_total();
  for (const auto &fg : fg_used_total) {
    const auto &inner_params = fg->parameters();
    std::map<std::string, AnfNodePtr> ref_key_nodes = GetParameterMap(inner_params);
    auto [remove, need_reserved] = RemoveRedundantParams(fg, ref_key_nodes);
    if (remove) {
      func_need_reserved_flags[fg] = need_reserved;
    }
    const auto &nodes = TopoSort(fg->get_return());
    for (const auto &node : nodes) {
      if (!node->isa<Parameter>()) {
        continue;
      }
      const auto &ref_key_str = GetRefKey(node);
      if (ref_key_str.empty()) {
        continue;
      }
      const auto &iter = ref_key_nodes.find(ref_key_str);
      if (iter == ref_key_nodes.end()) {
        continue;
      }
      const auto &real_param = iter->second;
      if (real_param != node) {
        (void)manager->Replace(node, real_param);
      }
    }
  }
  if (!func_need_reserved_flags.empty()) {
    RemoveCallerRedundantArgs(func_graph, manager, &func_need_reserved_flags);
  }
}
}  // namespace

FuncGraphPtr LiftFv(const pipeline::ResourceBasePtr &resource, const FuncGraphPtr &func_graph) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_save_graphs = context->CanDump(kIntroductory);
  if (enable_save_graphs) {
    DumpIR("before_lift_" + func_graph->ToString() + ".ir", func_graph);
  }
#endif
  FuncGraphPtr new_fg = LiftingClone(func_graph);
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("after_lift_" + new_fg->ToString() + ".ir", new_fg);
  }
#endif
  auto new_res = std::dynamic_pointer_cast<pipeline::Resource>(resource);
  if (new_res == nullptr) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, func_graph->return_node())
      << "Parameter resources is not a pipeline::Resource";
  }
  auto opt_fg = PartialEliminateOptPass(new_res, new_fg);
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("after_opt_" + opt_fg->ToString() + ".ir", opt_fg);
  }
#endif
  return opt_fg;
}

FuncGraphVector LiftFvMulti(const pipeline::ResourceBasePtr &resource, const FuncGraphVector &func_graphs) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    for (const auto &func_graph : func_graphs) {
      DumpIR("before_lift_" + func_graph->ToString() + ".ir", func_graph);
    }
  }
#endif
  bool has_used_fg = std::any_of(func_graphs.cbegin(), func_graphs.cend(), [](const FuncGraphPtr &func_graph) {
    return func_graph->func_graphs_used().size() != 0;
  });
  // All func_graphs being graded don't have used funcgraphs, no need to do lifting clone.
  if (!has_used_fg) {
    return func_graphs;
  }
  FuncGraphVector new_fgs = LiftingCloneMulti(func_graphs);
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    for (const auto &new_fg : new_fgs) {
      DumpIR("after_lift_" + new_fg->ToString() + ".ir", new_fg);
    }
  }
#endif
  return PartialEliminateMulti(resource, new_fgs);
}

FuncGraphPtr FreeVariablesEliminate(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  parse::ClearCNodeAbstract(func_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  FuncGraphPtr need_renormalize_func = func_graph;
  const auto &resources = optimizer->resource();
  if (func_graph->parent() != nullptr) {
    res = std::dynamic_pointer_cast<pipeline::Resource>(resources);
    need_renormalize_func = res->func_graph();
    func_graph->set_flag("J_INNER_FUNC", true);
  }
  abstract::AbstractBasePtrList new_args_spec;
  (void)std::transform(need_renormalize_func->parameters().begin(), need_renormalize_func->parameters().end(),
                       std::back_inserter(new_args_spec),
                       [](const AnfNodePtr &param) -> AbstractBasePtr { return param->abstract(); });
  FuncGraphPtr new_func_graph = pipeline::Renormalize(res, need_renormalize_func, new_args_spec);
  MS_LOG(DEBUG) << "LiftingClone for func_graph: " << need_renormalize_func->ToString();
  new_func_graph = LiftFv(resources, new_func_graph);
  if (func_graph->parent() != nullptr) {
    res->set_func_graph(new_func_graph);
    res->set_args_abs(new_args_spec);
    new_func_graph->set_manager(manager);
    for (auto sub_func : new_func_graph->func_graphs_used_total()) {
      if (sub_func->has_flag("J_INNER_FUNC")) {
        new_func_graph = sub_func;
        new_func_graph->set_manager(manager);
        sub_func->erase_flag("J_INNER_FUNC");
        break;
      }
    }
  }
  MergeParameters(new_func_graph, optimizer);

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("opt_free_variable_eliminate.ir", new_func_graph);
  }
#endif
  return new_func_graph;
}

bool CheckExistFv(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto exist_fv = [](const FuncGraphPtr &func_graph) { return !func_graph->free_variables_nodes().empty(); };
  if (exist_fv(func_graph)) {
    return true;
  }
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &total_fgs = manager->func_graphs_used_total(func_graph);
  return std::any_of(total_fgs.begin(), total_fgs.end(), exist_fv);
}

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
