/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/ad/grad.h"
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/jit/ps/pass.h"
#include "frontend/jit/ps/action.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/optimizer/irpass/inplace_input_replace.h"
#include "frontend/optimizer/irpass/virtualview_op.h"
#include "frontend/optimizer/irpass/virtualviewgrad_op.h"
#include "frontend/optimizer/irpass/view_inplace_utils.h"
#include "frontend/optimizer/irpass/free_variables_eliminate.h"
#include "include/frontend/optimizer/ad/grad_interface.h"
#include "ir/func_graph_cloner.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "include/utils/parallel_context.h"
#include "include/frontend/jit/ps/action_interface.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace ad {
namespace {
constexpr auto kAlreadyCheck = "already_check";
constexpr auto kNeedGradFlag = "need_grad";
constexpr auto kHasViewOutputFlag = "has_view_output";
constexpr auto kCheckViewInplaceGradFlag = "view_inplace_grad_validate";
constexpr auto kSetNeedGradFlag = "set_need_grad_flag";

bool ForwardInputsEqual(const AnfNodeWeakPtrList &first_inputs, const AnfNodeWeakPtrList &second_inputs) {
  if (first_inputs.size() != second_inputs.size()) {
    return false;
  }
  for (size_t i = 1; i < first_inputs.size(); ++i) {
    if (HasAbstractMonad(first_inputs[i].lock()) && HasAbstractMonad(second_inputs[i].lock())) {
      continue;
    }
    if (first_inputs[i].lock() != second_inputs[i].lock()) {
      return false;
    }
  }
  return true;
}

AnfNodePtr GetJUser(const FuncGraphManagerPtr &manager, const AnfNodePtr &j_node) {
  auto iter = manager->node_users().find(j_node);
  if (iter == manager->node_users().end()) {
    return nullptr;
  }
  auto users = iter->second;
  if (users.size() != 1) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, j_node) << "The size of J users should be 1, but got " << users.size();
  }
  return users.begin()->first;
}

void AddToManage(const pipeline::ResourceBasePtr &resources, const FuncGraphPtr &func_graph) {
  auto manager_ptr = resources->manager();
  MS_EXCEPTION_IF_NULL(manager_ptr);
  manager_ptr->AddFuncGraph(func_graph);
}

bool ViewInplacePrepare(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer, bool is_view_inplace) {
  const auto &resources = optimizer->resource();
  if (!is_view_inplace) {
    mindspore::pipeline::ViewInplaceBeforeGradProcessPass(resources, func_graph,
                                                          opt::irpass::ViewInplacePassType::OnlyDoInplace);
    return false;
  }

  // Do inline upfront to ensure the correct method is selected
  mindspore::pipeline::ViewInplaceBeforeGradProcessPass(resources, func_graph,
                                                        opt::irpass::ViewInplacePassType::CommonInline);
  //   Choose new view inplace grad scheme.
  (void)mindspore::opt::irpass::PreprocessForVirtualViewGradInsert(func_graph, optimizer);
  return true;
}

FuncGraphPtr InsertVirtualOpsProcess(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer) {
  const auto &resources = optimizer->resource();
  MS_LOG(INFO) << "Choose new view inplace grad scheme for func_graph:" << func_graph->ToString();
  mindspore::pipeline::ViewInplaceBeforeGradProcessPass(resources, func_graph,
                                                        opt::irpass::ViewInplacePassType::VirtualOpsInsert);

  if (!mindspore::opt::irpass::CheckExistFv(func_graph)) {
    // Convert View op name -> Primitive
    mindspore::opt::irpass::ConvertViewOpNameInVirtualViewGrad(func_graph, optimizer);
    mindspore::pipeline::ViewInplaceBeforeGradProcessPass(
      resources, func_graph, opt::irpass::ViewInplacePassType::DoInplaceAndVirtualOpsRemove);
    return func_graph;
  }
  MS_LOG(INFO) << "Exist free variable, handle supported control flow func graph";
  // If exist fv, do Renormalize and LiftFv.
  auto new_func_graph = mindspore::opt::irpass::FreeVariablesEliminate(func_graph, optimizer);
  AddToManage(resources, new_func_graph);
  // Convert View op name -> Primitive
  mindspore::opt::irpass::ConvertViewOpNameInVirtualViewGrad(new_func_graph, optimizer);
  mindspore::pipeline::ViewInplaceBeforeGradProcessPass(resources, new_func_graph,
                                                        opt::irpass::ViewInplacePassType::DoInplaceAndVirtualOpsRemove);
  return new_func_graph;
}
}  // namespace

FuncGraphPtr GradOneFuncGraph(const FuncGraphPtr &ori_func_graph, const opt::OptimizerPtr &optimizer, bool is_top,
                              BpropAutoMonadLevel level, bool is_view_inplace, bool is_grad_by_j = false) {
  MS_EXCEPTION_IF_NULL(ori_func_graph);
  auto gradkv = ori_func_graph->transforms().find("grad");
  if (gradkv != ori_func_graph->transforms().end()) {
    return gradkv->second.func_graph();
  }
  const auto &resources = optimizer->resource();
  AddToManage(resources, ori_func_graph);

  FuncGraphPtr new_func_graph = ori_func_graph;

  if (is_view_inplace) {
    parse::ClearCNodeAbstract(ori_func_graph);
    pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
    FuncGraphPtr need_renormalize_func = ori_func_graph;

    if (ori_func_graph->parent() != nullptr) {
      res = std::dynamic_pointer_cast<pipeline::Resource>(resources);
      need_renormalize_func = res->func_graph();
      ori_func_graph->set_flag("J_INNER_FUNC", true);
    }
    abstract::AbstractBasePtrList new_args_spec;
    (void)std::transform(need_renormalize_func->parameters().begin(), need_renormalize_func->parameters().end(),
                         std::back_inserter(new_args_spec),
                         [](const AnfNodePtr &param) -> AbstractBasePtr { return param->abstract(); });
    new_func_graph = pipeline::Renormalize(res, need_renormalize_func, new_args_spec);

    auto manager = optimizer->manager();
    MS_EXCEPTION_IF_NULL(manager);
    need_renormalize_func->set_manager(manager);

    if (ori_func_graph->parent() != nullptr) {
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
  }

  // Preprocessing for view inplace
  bool use_view_inplace_new_method = ViewInplacePrepare(new_func_graph, optimizer, is_view_inplace);
  FuncGraphPtr func_graph = new_func_graph;
  if (use_view_inplace_new_method) {
    func_graph = InsertVirtualOpsProcess(new_func_graph, optimizer);
  }

  auto multi_graph_sink = [&func_graph](const FuncGraphPtr &f) {
    if (MsContext::GetInstance()->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK)) {
      if (func_graph->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE)) {
        f->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE, true);
      }
    }
  };

  SavedTensorsHooks saved_tensors_hooks_scope(func_graph);
  auto f = std::make_shared<DFunctor>(func_graph, resources, is_top, is_grad_by_j);
  auto user_defined = f->KUserDefined(func_graph);
  if (user_defined != nullptr) {
    multi_graph_sink(user_defined);
    if (is_top) {
      DFunctor::Clear();
    }
    return user_defined;
  }
  f->Init(is_top);
  f->MapObject();
  f->MapMorphism();
  f->Finish();
  auto res = f->k_graph();
  res->set_attr(kAttrBpropAutoMonadLevel, MakeValue<int>(level));
  auto tape = f->tape();
  tape->set_flag(mindspore::kFuncGraphFlagBackPropEntry, true);
  if (is_top) {
    DFunctor::Clear();
  }

  // Postprocessing for view inplace
  if (use_view_inplace_new_method) {
    mindspore::pipeline::ViewInplaceBeforeGradProcessPass(resources, func_graph,
                                                          opt::irpass::ViewInplacePassType::EliminateVirtualView);
  }

  multi_graph_sink(res);
  (void)func_graph->transforms().emplace("grad", FuncGraphTransform(res));
  return res;
}

FuncGraphPtr Grad(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer, bool is_top,
                  BpropAutoMonadLevel level, bool is_view_inplace, bool is_grad_by_j) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto gradkv = func_graph->transforms().find("grad");
  if (gradkv != func_graph->transforms().end()) {
    return gradkv->second.func_graph();
  }
  if (!optimizer->is_first_order_j() && func_graph->has_attr(FUNC_GRAPH_ATTR_UNSUPPORT_HIGHER_GRAD_REASON)) {
    auto reason = func_graph->get_attr(FUNC_GRAPH_ATTR_UNSUPPORT_HIGHER_GRAD_REASON);
    MS_EXCEPTION_IF_NULL(reason);
    MS_EXCEPTION(NotSupportError) << "Higher-order differentiation is not supported for the current scenario, reason: "
                                  << GetValue<string>(reason);
  }

  const auto &resources = optimizer->resource();
  AddToManage(resources, func_graph);

  FuncGraphPtr grad_fg = func_graph;
  if (func_graph->func_graphs_used().size() != 0 && optimizer->is_first_order_j()) {
    lift_fv_before_grad = true;
    grad_fg = mindspore::opt::irpass::LiftFv(resources, func_graph);
  } else {
    lift_fv_before_grad = false;
  }
  if (is_view_inplace && mindspore::opt::irpass::CheckExistFv(func_graph)) {
    auto res = std::dynamic_pointer_cast<pipeline::Resource>(resources);
    auto res_func = res->func_graph();
    func_graph->set_flag("J_INNER_FUNC", true);
    grad_fg = mindspore::opt::irpass::LiftFv(resources, res_func);
    auto manager = optimizer->manager();
    MS_EXCEPTION_IF_NULL(manager);
    grad_fg->set_manager(manager);
    for (auto sub_func : grad_fg->func_graphs_used_total()) {
      bool has_flag = sub_func->has_flag("J_INNER_FUNC");
      if (has_flag) {
        grad_fg = sub_func;
        sub_func->erase_flag("J_INNER_FUNC");
        break;
      }
    }
  }
  auto output_graph = GradOneFuncGraph(grad_fg, optimizer, is_top, level, is_view_inplace, is_grad_by_j);
  auto primal_fg_iter = output_graph->transforms().find("primal");
  if (primal_fg_iter != output_graph->transforms().end()) {
    auto actual_primal_graph = primal_fg_iter->second.func_graph();
    auto jit_config = PhaseManager::GetInstance().jit_config();
    actual_primal_graph->set_user_data<std::map<std::string, std::string>>(
      "jit_config", std::make_shared<std::map<std::string, std::string>>(jit_config));
  }
  return output_graph;
}

FuncGraphVector GradMultiFuncGraph(const FuncGraphVector &func_graphs, const opt::OptimizerPtr &optimizer,
                                   const std::vector<bool> &is_view_inplace, bool is_top, bool is_grad_by_j) {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  const bool is_parallel_mode =
    parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel;
  BpropAutoMonadLevel bprop_auto_monad_level = is_parallel_mode ? kLevelTop : kLevelWhole;
  FuncGraphVector grad_fgs;
  if (func_graphs.size() == 1) {
    auto grad_fg = Grad(func_graphs[0], optimizer, is_top, bprop_auto_monad_level, is_view_inplace[0], is_grad_by_j);
    grad_fgs.push_back(grad_fg);
    return grad_fgs;
  }
  const auto &resources = optimizer->resource();
  auto manager_ptr = resources->manager();
  MS_EXCEPTION_IF_NULL(manager_ptr);
  for (const auto &func_graph : func_graphs) {
    manager_ptr->AddFuncGraph(func_graph);
  }
  FuncGraphVector before_grad_fgs;
  if (optimizer->is_first_order_j()) {
    lift_fv_before_grad = true;
    before_grad_fgs = mindspore::opt::irpass::LiftFvMulti(resources, func_graphs);
  } else {
    before_grad_fgs = func_graphs;
    lift_fv_before_grad = false;
  }
  for (size_t i = 0; i < before_grad_fgs.size(); ++i) {
    auto func_graph = before_grad_fgs[i];
    auto grad_fg =
      GradOneFuncGraph(func_graph, optimizer, is_top, bprop_auto_monad_level, is_view_inplace[i], is_grad_by_j);
    grad_fgs.push_back(grad_fg);
  }
  return grad_fgs;
}

FuncGraphPtr Kprim(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) {
  auto fg = g_k_prims.KPrimitive(nullptr, value_node, resources);
  if (fg == nullptr) {
    return nullptr;
  }
  return BasicClone(fg);
}

MetaFuncGraphPtr Kmeta(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &, const AnfNodePtr &node) {
  MetaFuncGraphPtr fg = g_k_prims.KMetaFuncGraph(prim, node);
  return fg;
}

bool MergeForward(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::unordered_map<FuncGraphPtr, std::vector<AnfNodePtr>> forward_fg_to_j_nodes;
  auto all_nodes = TopoSort(root->get_return(), SuccDeeperSimple, AlwaysInclude);
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimJ)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto merge_forward = cnode->user_data<bool>("merge_forward");
    if (merge_forward == nullptr || !(*merge_forward)) {
      continue;
    }
    auto forward_fg = GetValueNode<FuncGraphPtr>(cnode->input(1));
    if (forward_fg == nullptr) {
      continue;
    }
    (void)forward_fg_to_j_nodes[forward_fg].emplace_back(node);
  }
  bool change = false;
  for (const auto &iter : forward_fg_to_j_nodes) {
    auto &j_nodes = iter.second;
    MS_LOG(DEBUG) << "J nodes size is " << j_nodes.size();
    if (j_nodes.size() <= 1) {
      continue;
    }
    auto first_j_user = GetJUser(manager, j_nodes[0]);
    if (first_j_user == nullptr) {
      continue;
    }
    const auto &first_forward_inputs = first_j_user->cast<CNodePtr>()->weak_inputs();
    for (size_t i = 1; i < j_nodes.size(); ++i) {
      auto j_user = GetJUser(manager, j_nodes[i]);
      const auto &forward_inputs = j_user->cast<CNodePtr>()->weak_inputs();
      if (!ForwardInputsEqual(first_forward_inputs, forward_inputs)) {
        continue;
      }
      manager->Replace(j_user, first_j_user);
      MS_LOG(DEBUG) << "Replace J user " << j_user->DebugString() << " with the first J user "
                    << first_j_user->DebugString();
      change = true;
    }
  }
  return change;
}
}  // namespace ad
}  // namespace mindspore
