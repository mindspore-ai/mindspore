/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/pass.h"

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "ir/func_graph_cloner.h"
#include "pipeline/jit/parse/parse_base.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/validator.h"
#include "pipeline/jit/remove_value_node_dup.h"
#include "frontend/optimizer/opt.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/cse_pass.h"
#include "frontend/optimizer/clean.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/graph_transform.h"
#include "frontend/optimizer/auto_monad_eliminate.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "frontend/parallel/cache_embedding/cache_embedding.h"
#include "frontend/parallel/allreduce_fusion/step_allreduce_fusion.h"
#include "frontend/optimizer/recompute.h"
#include "utils/log_adapter.h"
#include "pipeline/jit/pipeline_split.h"
#include "pipeline/pynative/pynative_execute.h"
#include "pipeline/jit/static_analysis/auto_monad.h"
#include "frontend/optimizer/irpass/branch_culling.h"
#include "frontend/optimizer/irpass/gradient_eliminate.h"
#include "frontend/optimizer/irpass/parameter_eliminate.h"
#include "frontend/optimizer/irpass/updatestate_eliminate.h"
#if ((defined ENABLE_CPU) && (!defined _WIN32))
#include "ps/util.h"
#include "ps/ps_context.h"
#endif

namespace mindspore {
namespace pipeline {
using OptPassGroupMap = opt::OptPassGroupMap;
using Optimizer = opt::Optimizer;
using CompileGraphs = compile::CompileGraphs;
using abstract::AnalysisResult;
using mindspore::abstract::AnalysisContextPtr;
using mindspore::validator::Validate;
namespace {
void DoRenormalize(const bool &changed, const FuncGraphPtr &func_graph, const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(res);
  abstract::AbstractBasePtrList args_spec;
  auto parameters = func_graph->parameters();
  (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_spec),
                       [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
  if (changed) {
    FuncGraphPtr new_fg = Renormalize(res, func_graph, args_spec);
    res->set_func_graph(new_fg);
  }
  res->set_args_spec(args_spec);
}
}  // namespace

bool SimplifyDataStructuresPass(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  FuncGraphPtr func_graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = opt::SimplifyDataStructures(func_graph, res->manager());
  DoRenormalize(changed, func_graph, res);
  return true;
}

bool TransformTopGraphPass(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  if (res->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "Transform top graph error.";
  }
  FuncGraphPtr func_graph = res->func_graph();
  if (opt::FuncGraphHasTupleInput(func_graph)) {
    opt::GraphTupleParamTransform graph_trans;
    func_graph = graph_trans(func_graph, res->manager());
    res->set_func_graph(func_graph);
    AbstractBasePtrList abs_spec_list;
    auto &params = func_graph->parameters();
    std::transform(params.begin(), params.end(), std::back_inserter(abs_spec_list),
                   [](const AnfNodePtr &node) { return node->abstract(); });
    res->set_args_spec(abs_spec_list);
  }
  return true;
}

bool CleanAfterOptAPass(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  FuncGraphPtr func_graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = opt::CleanAfterOptA(func_graph, res->manager());
  DoRenormalize(changed, func_graph, res);
  return true;
}

FuncGraphPtr PrimBpOptPassStep1(const opt::irpass::OptimizeIRPassLib &irpass, const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  MS_EXCEPTION_IF_NULL(res->func_graph());
  opt::OptPassConfig pynative_eliminate = opt::OptPassConfig({
    irpass.pynative_eliminate_,
  });

  opt::OptPassConfig switch_simplify = opt::OptPassConfig({
    irpass.switch_simplify_,
  });

  opt::OptPassConfig inline_opt = opt::OptPassConfig({
    irpass.inline_,
  });

  OptPassGroupMap map(
    {{"ad_eliminate", pynative_eliminate}, {"ad_inline", inline_opt}, {"ad_switch_simplify", switch_simplify}});

  auto prim_bprop_opt_step_1 = opt::Optimizer::MakeOptimizer("prim_bprop_opt_step_1", res, map);
  FuncGraphPtr func_graph = res->func_graph();
  WITH(MsProfile::GetProfile()->Step("prim_bprop_opt_step_1"))[&prim_bprop_opt_step_1, &func_graph]() {
    func_graph = prim_bprop_opt_step_1->step(func_graph, true);
  };
  return func_graph;
}

FuncGraphPtr PrimBpOptPassStep2(const opt::irpass::OptimizeIRPassLib &irpass, const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  MS_EXCEPTION_IF_NULL(res->func_graph());
  opt::OptPassConfig special_op_simplify = opt::OptPassConfig({
    irpass.switch_simplify_,
    irpass.reduce_eliminate_,
    irpass.tile_eliminate_,
    irpass.arithmetic_simplify_,
  });

  opt::OptPassConfig inline_opt = opt::OptPassConfig({
    irpass.inline_,
  });

  auto re_auto_monadwrapper = [](const FuncGraphPtr &root, const opt::OptimizerPtr &) -> bool {
    return ReAutoMonad(root);
  };
  OptPassGroupMap map({{"ad_renormalize", opt::OptPassConfig::Renormalize()},
                       {"ad_inline", inline_opt},
                       {"ad_special_op_simplify", special_op_simplify},
                       {"auto_monad_grad", opt::OptPassConfig(re_auto_monadwrapper)}});

  auto prim_bprop_opt_step_2 = opt::Optimizer::MakeOptimizer("prim_bprop_opt_step_2", res, map);
  FuncGraphPtr func_graph = res->func_graph();
  WITH(MsProfile::GetProfile()->Step("prim_bprop_opt_step_2"))[&prim_bprop_opt_step_2, &func_graph]() {
    func_graph = prim_bprop_opt_step_2->step(func_graph, true);
  };
  return func_graph;
}

FuncGraphPtr BpropGraphFinalOptPass(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  MS_EXCEPTION_IF_NULL(res->func_graph());
  (void)TransformTopGraphPass(res);

  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig bg_final_opt = opt::OptPassConfig({
    irpass.inline_,
    irpass.tuple_list_get_set_item_eliminator_,
    irpass.tuple_list_get_item_eliminator_,
    irpass.tuple_list_set_item_eliminator_,
    irpass.depend_value_elim_,
    irpass.reshape_eliminate_,
    irpass.switch_simplify_,
    irpass.addn_zero_filter_,
  });
  opt::OptPassConfig fill_zeros_like = opt::OptPassConfig{irpass.zero_like_fill_zero_};
  OptPassGroupMap map({
    {"ad_final_opt", bg_final_opt},
    {"zeros_like", fill_zeros_like},
  });

  if (pynative::PynativeExecutor::GetInstance()->grad_executor()->need_renormalize()) {
    (void)map.emplace_back(std::make_pair("renormalize", opt::OptPassConfig::Renormalize()));
    opt::OptPassConfig env_eliminate = opt::OptPassConfig({
      irpass.incorporate_call_,
      irpass.incorporate_call_switch_,
      irpass.incorporate_getitem_set_,
    });
    (void)map.emplace_back(std::make_pair("env_eliminate", env_eliminate));
  }

  auto bprop_graph_final_opt = opt::Optimizer::MakeOptimizer("bprop_graph_final_opt", res, map);
  FuncGraphPtr func_graph = res->func_graph();
  WITH(MsProfile::GetProfile()->Step("bprop_graph_final_opt"))[&bprop_graph_final_opt, &func_graph]() {
    func_graph = bprop_graph_final_opt->step(func_graph, true);
  };

  return func_graph;
}

namespace {
bool ReAutoMonadWrapper(const FuncGraphPtr &root, const opt::OptimizerPtr &) { return ReAutoMonad(root); }

bool parallel_mode() {
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  if (ps::PSContext::instance()->is_server() || ps::PSContext::instance()->is_scheduler()) {
    return false;
  }
#endif
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  return (parallel_mode == parallel::AUTO_PARALLEL) || (parallel_mode == parallel::SEMI_AUTO_PARALLEL);
}

void AddParallelRenormalize(OptPassGroupMap *map_a) {
  if (parallel_mode()) {
    auto parallel_end_opt =
      find_if(map_a->begin(), map_a->end(), [](auto opt_pair) { return opt_pair.first == "grad"; });
    if (parallel_end_opt != map_a->end()) {
      (void)map_a->insert(parallel_end_opt, {"parallel_renormalize", opt::OptPassConfig::Renormalize()});
    }
  }
}

opt::OptPassConfig GetOptPassA1(const opt::irpass::OptimizeIRPassLib &irpass) {
  return opt::OptPassConfig({
    irpass.switch_defer_inline_,
    irpass.switch_layer_defer_inline_,
    irpass.switch_simplify_,
    irpass.exchange_switch_depend_value_,
    irpass.float_depend_g_call_,

    // Safe inlining
    irpass.inline_,
    irpass.updatestate_useless_node_eliminater_,
    irpass.updatestate_pure_node_eliminater_,
    irpass.load_eliminater_,
    irpass.stopgrad_eliminater_,
    irpass.partial_eliminate_,
    irpass.replace_applicator_,

    // Miscellaneous
    irpass.tuple_list_get_item_eliminator_,
    irpass.tuple_list_get_item_const_eliminator_,
    irpass.tuple_list_set_item_eliminator_,
    irpass.tuple_list_get_set_item_eliminator_,
    irpass.tuple_list_get_item_depend_reorder_,
    irpass.tuple_list_convert_item_index_to_positive_,

    irpass.env_get_item_eliminate_,
    irpass.env_get_item_add_eliminate_,
    irpass.env_get_set_item_eliminate_,
    irpass.env_get_item_depend_swap_,

    irpass.cast_eliminate_,
    irpass.reshape_eliminate_,
    irpass.reduce_eliminate_,
    irpass.tile_eliminate_,
    irpass.transpose_eliminate_,
    irpass.minmaximum_grad_,
    irpass.get_make_ref_eliminate_,

    // Arithmetic simplifications
    irpass.arithmetic_simplify_,
    irpass.addn_zero_filter_,
    irpass.adjust_all_reduce_mul_add_,
    irpass.accumulaten_eliminater_,

    // Safe inlining
    irpass.inline_,
    irpass.updatestate_useless_node_eliminater_,
    irpass.updatestate_pure_node_eliminater_,
    irpass.load_eliminater_,
    irpass.stopgrad_eliminater_,
    irpass.sparse_tensor_eliminate_,
  });
}

OptPassGroupMap GetOptPassesA(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig a_1 = GetOptPassA1(irpass);
  opt::OptPassConfig a_2 = opt::OptPassConfig(
    {
      irpass.switch_simplify_,
      irpass.specialize_transform_,
      irpass.merge_addn_,
      irpass.float_tuple_getitem_switch_,
      irpass.float_env_getitem_switch_,
      irpass.inline_,
      irpass.incorporate_getitem_set_,
      irpass.incorporate_call_,
      irpass.incorporate_call_switch_,
      irpass.incorporate_env_getitem_bypass_recursive_,
      irpass.incorporate_env_getitem_switch_,
      irpass.env_get_item_eliminate_,
      irpass.depend_value_elim_,
      irpass.all_reduce_const_elim_,
    },
    false, true);

  opt::OptPassConfig a_after_grad = opt::OptPassConfig({irpass.inline_without_move_});

  opt::OptPassConfig a_3 = opt::OptPassConfig(
    {
      irpass.arithmetic_simplify2_,
      irpass.same_eliminate_,
      irpass.check_bprop_eliminate_,
      irpass.switch_layer_defer_inline_,
      irpass.replace_applicator_,
      irpass.mirror_mini_step_elim_,
      irpass.virtual_add_elim_,
      irpass.row_tensor_add_zeros_like_,
      irpass.mini_step_allgather_replace_,
      irpass.micro_step_allgather_replace_,
    },
    false, true);
  opt::OptPassConfig accelerated_algorithm = opt::OptPassConfig({irpass.less_batch_normalization_});
  opt::OptPassConfig virtual_dataset = opt::OptPassConfig({irpass.virtual_dataset_eliminate_});
  opt::OptPassConfig after_resolve_pass =
    opt::OptPassConfig({irpass.get_make_ref_eliminate_, irpass.replace_old_param_});
  opt::OptPassConfig updatestate_depend_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateDependEliminater());
  opt::OptPassConfig updatestate_assign_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateAssignEliminater());
  opt::OptPassConfig updatestate_loads_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateLoadsEliminater());

  // Before adjusting map_a, check GetA1A2() and GetOptPynativeGradEpiloguePhases().
  OptPassGroupMap map_a({{"switch_simplify", opt::OptPassConfig({irpass.switch_simplify_})},
                         {"a_1", a_1},
                         {"updatestate_depend_eliminate", updatestate_depend_eliminate},
                         {"updatestate_assign_eliminate", updatestate_assign_eliminate},
                         {"updatestate_loads_eliminate", updatestate_loads_eliminate},
                         {"parameter_eliminate", opt::OptPassConfig(opt::irpass::ParameterEliminator())},
                         {"a_2", a_2},
                         {"accelerated_algorithm", accelerated_algorithm},
                         {"auto_parallel", opt::OptPassConfig(parallel::StepAutoParallel)},
                         {"parallel", opt::OptPassConfig(parallel::StepParallel)},
                         {"allreduce_fusion", opt::OptPassConfig(parallel::StepAllreduceFusion)},
                         {"virtual_dataset", virtual_dataset},
                         {"virtual_output", opt::OptPassConfig({irpass.virtual_output_eliminate_})},
                         {"grad", opt::OptPassConfig(opt::irpass::ExpandJPrim())},
                         {"after_resolve", after_resolve_pass},
                         {"a_after_grad", a_after_grad},
                         {"renormalize", opt::OptPassConfig::Renormalize()},
                         {"auto_monad_grad", opt::OptPassConfig(ReAutoMonadWrapper)},
                         {"auto_monad_eliminator", opt::OptPassConfig(opt::AutoMonadEliminator())},
                         {"cse", opt::OptPassConfig(opt::CSEPass(false))},
                         {"a_3", a_3}});
  AddParallelRenormalize(&map_a);
  return map_a;
}

OptPassGroupMap GetA1A2(const opt::irpass::OptimizeIRPassLib &irpass) {
  auto opt_a = GetOptPassesA(irpass);
  constexpr auto a1_a2_len = 7;
  OptPassGroupMap a1_a2(opt_a.begin(), opt_a.begin() + a1_a2_len);
  return a1_a2;
}

OptPassGroupMap GetOptPassesAfterCconv(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig c_1 = opt::OptPassConfig({
    // Safe inlining,
    irpass.inline_,
    irpass.updatestate_useless_node_eliminater_,
    irpass.updatestate_pure_node_eliminater_,
    irpass.load_eliminater_,
    irpass.switch_call_monad_eliminater_,
    irpass.stopgrad_eliminater_,
    irpass.partial_eliminate_,
  });
  opt::OptPassConfig updatestate_depend_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateDependEliminater());
  opt::OptPassConfig updatestate_assign_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateAssignEliminater());
  opt::OptPassConfig updatestate_loads_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateLoadsEliminater());

  OptPassGroupMap map_a({{"c_1", c_1},
                         {"updatestate_depend_eliminate", updatestate_depend_eliminate},
                         {"updatestate_assign_eliminate", updatestate_assign_eliminate},
                         {"updatestate_loads_eliminate", updatestate_loads_eliminate},
                         {"cse", opt::OptPassConfig(opt::CSEPass(false))},
                         {"renormalize", opt::OptPassConfig::Renormalize()}});

  return map_a;
}

OptPassGroupMap GetOptPassesTransformGraph(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig d_1 =
    opt::OptPassConfig({irpass.call_graph_tuple_transform_, irpass.tuple_list_get_item_eliminator_,
                        irpass.tuple_list_get_item_const_eliminator_, irpass.tuple_list_set_item_eliminator_,
                        irpass.tuple_list_get_set_item_eliminator_, irpass.tuple_list_get_item_depend_reorder_,
                        irpass.tuple_list_convert_item_index_to_positive_});

  OptPassGroupMap map_a({{"d_1", d_1}, {"renormalize", opt::OptPassConfig::Renormalize()}});

  return map_a;
}

OptPassGroupMap GetOptPassesB(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig b_1 = opt::OptPassConfig({irpass.zero_like_fill_zero_,
                                               irpass.tuple_list_get_item_eliminator_,
                                               irpass.tuple_list_get_item_const_eliminator_,
                                               irpass.tuple_list_set_item_eliminator_,
                                               irpass.tuple_list_get_set_item_eliminator_,
                                               irpass.tuple_list_get_item_depend_reorder_,
                                               irpass.tuple_list_convert_item_index_to_positive_,
                                               irpass.float_tuple_getitem_switch_,
                                               irpass.reset_defer_inline_,
                                               irpass.inline_,
                                               irpass.updatestate_useless_node_eliminater_,
                                               irpass.updatestate_pure_node_eliminater_,
                                               irpass.load_eliminater_,
                                               irpass.stopgrad_eliminater_,
                                               irpass.special_op_eliminate_,
                                               irpass.get_make_ref_eliminate_,
                                               irpass.incorporate_env_getitem_,
                                               irpass.incorporate_env_getitem_switch_,
                                               irpass.env_get_item_eliminate_,
                                               irpass.env_get_item_add_eliminate_,
                                               irpass.env_get_set_item_eliminate_,
                                               irpass.env_get_item_depend_swap_,
                                               irpass.incorporate_env_getitem_switch_layer_,
                                               irpass.value_based_eliminate_,
                                               irpass.virtual_accu_grad_,
                                               irpass.virtual_assign_add_,
                                               irpass.mirror_micro_step_},
                                              false, true);
  opt::OptPassConfig b_2 = opt::OptPassConfig({
    irpass.replace_refkey_by_param_,
    irpass.make_ref_eliminate_,
    irpass.get_ref_param_eliminate_,
    irpass.row_tensor_eliminate_,
  });
  opt::OptPassConfig updatestate_depend_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateDependEliminater());
  opt::OptPassConfig updatestate_assign_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateAssignEliminater());
  opt::OptPassConfig updatestate_loads_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateLoadsEliminater());
  OptPassGroupMap map({
    {"b_1", b_1},
    {"b_2", b_2},
    {"updatestate_depend_eliminate", updatestate_depend_eliminate},
    {"updatestate_assign_eliminate", updatestate_assign_eliminate},
    {"updatestate_loads_eliminate", updatestate_loads_eliminate},
    {"renormalize", opt::OptPassConfig::Renormalize()},
    {"cse", opt::OptPassConfig(opt::CSEPass(false))},
  });
  return map;
}

OptPassGroupMap GetOptPassesPynativeElim(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig pynative_eliminate = opt::OptPassConfig({
    irpass.pynative_eliminate_,
  });

  OptPassGroupMap map({
    {"pynative_eliminate", pynative_eliminate},
  });
  return map;
}

OptPassGroupMap GetOptPassesC(const opt::irpass::OptimizeIRPassLib &) {
  return OptPassGroupMap({{"renormalize", opt::OptPassConfig::Renormalize()}});
}

OptPassGroupMap GetControlPhases(const opt::irpass::OptimizeIRPassLib &) {
  opt::OptPassConfig control_group = opt::OptPassConfig(opt::irpass::ConvertSwitchReplacement());
  OptPassGroupMap map({
    {"control_group", control_group},
    {"renormalize", opt::OptPassConfig::Renormalize()},
  });
  return map;
}

OptPassGroupMap GetOptPynativeGradEpiloguePhases(const opt::irpass::OptimizeIRPassLib &irpass) {
  auto opt_a = GetOptPassesA(irpass);
  auto a3 = opt_a[opt_a.size() - 1];
  OptPassGroupMap map({
    {"renormalize", opt::OptPassConfig::Renormalize()},
    {"cse", opt::OptPassConfig(opt::CSEPass(false))},
    {a3},
  });
  return map;
}

OptPassGroupMap GetInferenceOptPreparePhases() {
  opt::irpass::InferenceOptPrepareLib irpass;
  auto grad_var_prepare = opt::OptPassConfig({irpass.grad_var_prepare_});
  opt::OptPassGroupMap prepare_map({{"inference_opt_prep", grad_var_prepare}});
  return prepare_map;
}

OptPassGroupMap GetPreparePhases(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig prepare_group = opt::OptPassConfig({irpass.print_tuple_wrapper_});
  OptPassGroupMap map({{"prepare_group", prepare_group}});
  return map;
}

OptPassGroupMap GetBeforeRecomputePass(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig set_cell_output_no_recompute = opt::OptPassConfig({irpass.set_cell_output_no_recompute_});
  OptPassGroupMap map({{"set_cell_output_no_recompute", set_cell_output_no_recompute}});
  return map;
}

OptPassGroupMap GetAfterRecomputePass(const opt::irpass::OptimizeIRPassLib &) {
  OptPassGroupMap map({{"cse", opt::OptPassConfig(opt::CSEPass(false))}});
  return map;
}

static std::unordered_map<std::string, std::shared_ptr<Optimizer>> g_pass_opts = {};

void InitOpt(const ResourcePtr &res) {
  if (g_pass_opts.size() == 0) {
    opt::irpass::OptimizeIRPassLib irpass;
    g_pass_opts["a1a2"] = Optimizer::MakeOptimizer("a1a2", res, GetA1A2(irpass));
    g_pass_opts["opt_a"] = Optimizer::MakeOptimizer("opt_a", res, GetOptPassesA(irpass));
    g_pass_opts["opt_b"] = Optimizer::MakeOptimizer("opt_b", res, GetOptPassesB(irpass), false, true);
    g_pass_opts["opt_after_cconv"] =
      Optimizer::MakeOptimizer("opt_after_cconv", res, GetOptPassesAfterCconv(irpass), false, true);
    g_pass_opts["opt_trans_graph"] =
      Optimizer::MakeOptimizer("opt_trans_graph", res, GetOptPassesTransformGraph(irpass), true, true);
    g_pass_opts["renormal"] = Optimizer::MakeOptimizer("renormal", res, GetOptPassesC(irpass));
    g_pass_opts["opt_control"] = Optimizer::MakeOptimizer("opt_control", res, GetControlPhases(irpass), true, true);
    g_pass_opts["opt_grad_epilogue"] =
      Optimizer::MakeOptimizer("opt_grad_epilogue", res, GetOptPynativeGradEpiloguePhases(irpass), true, false);
    g_pass_opts["opt_prepare"] = Optimizer::MakeOptimizer("opt_prepare", res, GetPreparePhases(irpass));
    g_pass_opts["opt_before_recompute"] =
      Optimizer::MakeOptimizer("opt_before_recompute", res, GetBeforeRecomputePass(irpass));
    g_pass_opts["opt_after_recompute"] =
      Optimizer::MakeOptimizer("opt_after_recompute", res, GetAfterRecomputePass(irpass));
  }
}
}  // namespace

void ReclaimOptimizer() {
  for (auto &opt : g_pass_opts) {
    opt.second = nullptr;
  }
  g_pass_opts.clear();
}

bool OptPassGroup(const ResourcePtr &res, const std::string &name) {
  MS_EXCEPTION_IF_NULL(res);
  if (res->func_graph() == nullptr) {
    MS_LOG(ERROR) << "Opt passes int64_t error";
    return false;
  }

  FuncGraphPtr func_graph = res->func_graph();
  MS_LOG(DEBUG) << "Start " << name << " func graph:" << func_graph->ToString() << ", "
                << func_graph->get_return()->DebugString(true);
  InitOpt(res);
  if (g_pass_opts.find(name) != g_pass_opts.end()) {
    res->set_func_graph(g_pass_opts[name]->step(func_graph));
  }
  // Note: StepParallel may modify the AbstractValue of the parameters of func_graph, but they are not updated to
  // res->args_spec_ yet. So if any later pass or action want to use that variable, it should be set here.
  return true;
}

bool OptPassA1A2(const ResourcePtr &res) { return OptPassGroup(res, "a1a2"); }
bool OptPassAGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_a"); }
bool OptPassBGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_b"); }
bool OptPassAfterCconvGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_after_cconv"); }
bool OptPassTransformGraphGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_trans_graph"); }
bool ControlGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_control"); }
bool PrepareGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_prepare"); }
bool OptBeforeRecomputeGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_before_recompute"); }
bool OptAfterRecomputeGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_after_recompute"); }

bool OptPassRNGroup(const ResourcePtr &res) { return OptPassGroup(res, "renormal"); }

bool OptPassGradEpilogueGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_grad_epilogue"); }

bool AddRecomputationPass(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  opt::InsertRecomputedNodes(res->func_graph());
  return true;
}

bool AddCacheEmbeddingPass(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  if (ps::PSContext::instance()->is_ps_mode()) {
    return true;
  }
#endif
  FuncGraphPtr func_graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);

  parallel::AddCacheEmbedding(func_graph);
  if (func_graph->has_flag(GRAPH_FLAG_CACHE_ENABLE)) {
    auto params = func_graph->parameters();
    AbstractBasePtrList args_spec_list;
    std::for_each(params.begin(), params.end(),
                  [&args_spec_list](const AnfNodePtr &node) { args_spec_list.push_back(node->abstract()); });
    func_graph = pipeline::Renormalize(res, func_graph, args_spec_list);
  }
  return true;
}

bool RemoveValueNodeDuplicationsPass(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  if (res->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "Remove value node duplications error.";
  }
  auto manager = res->manager();
  HashCache hash_cache;
  HashValue hashes;
  // Remove duplicated value nodes across all graphs in manager
  auto node_user_map = manager->node_users();
  for (auto &fg : manager->func_graphs()) {
    auto value_nodes = fg->value_nodes();
    for (const auto &value_pair : value_nodes) {
      auto users = node_user_map[value_pair.first];
      // For data parallel with some parameters redundant, the allreduce will share the same value node
      // which will raise an error when do allreduce fusion, so the solution is to make the allreduce's value node
      // not be removed, if we found the fusion tag.
      if (users.size() == 1) {
        auto cnode = users.front().first->cast<CNodePtr>();
        if (IsPrimitiveCNode(cnode, prim::kPrimAllReduce) && cnode->inputs().size() > 1 &&
            cnode->input(1)->isa<ValueNode>()) {
          auto allreduce_prim = GetCNodePrimitive(users.front().first);
          auto attrs = allreduce_prim->attrs();
          auto fusion_id = attrs.find(mindspore::parallel::FUSION);
          if (fusion_id != attrs.end() && GetValue<int64_t>(fusion_id->second) > 0) {
            continue;
          }
        }
      }
      TryToDoReplace(manager.get(), value_pair.first, &hash_cache, &hashes);
    }
  }
  return true;
}

bool CconvPass(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  MS_EXCEPTION_IF_NULL(res->func_graph());
  FuncGraphPtr func_graph = res->func_graph();
  FuncGraphPtr new_fg = LiftingClone(func_graph);
  res->set_func_graph(new_fg);
  return true;
}

bool PipelineSplitPass(const ResourcePtr &res) { return PipelineSplit(res); }

bool ValidatePass(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  MS_EXCEPTION_IF_NULL(res->func_graph());
  FuncGraphPtr func_graph = res->func_graph();
  Validate(func_graph);
  return true;
}

bool InferenceOptPreparePass(const ResourcePtr &res) {
  FuncGraphPtr func_graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto prepare_map = GetInferenceOptPreparePhases();
  auto infer_opt_prepare = opt::Optimizer::MakeOptimizer("inference_prepare", res, prepare_map);
  (void)infer_opt_prepare->step(func_graph, false);
  return true;
}

bool PynativeOptPass(const ResourcePtr &res) {
  FuncGraphPtr func_graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  opt::irpass::OptimizeIRPassLib irpass;
  auto pynative_opt = GetOptPassesPynativeElim(irpass);
  auto pynative_opt_opt = opt::Optimizer::MakeOptimizer("pynative_opt", res, pynative_opt);
  (void)pynative_opt_opt->step(func_graph, false);
  return true;
}

bool AutoMonadElimOptPass(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->manager());
  auto res = std::make_shared<pipeline::Resource>();
  res->set_func_graph(func_graph);
  res->set_manager(func_graph->manager());

  // opt::irpass::OptimizeIRPassLib is not used here to avoid double free problems in external calls.
  opt::SubstitutionPtr updatestate_useless_node_eliminater =
    opt::MakeSubstitution(std::make_shared<opt::irpass::UpdatestateUselessNodeEliminater>(),
                          "updatestate_useless_node_eliminater", prim::kPrimUpdateState);
  opt::SubstitutionPtr updatestate_pure_node_eliminater =
    opt::MakeSubstitution(std::make_shared<opt::irpass::UpdatestatePureNodeEliminater>(),
                          "updatestate_pure_node_eliminater", prim::kPrimUpdateState);

  opt::OptPassConfig updatestate_eliminater = opt::OptPassConfig({
    updatestate_useless_node_eliminater,
    updatestate_pure_node_eliminater,
  });
  opt::OptPassConfig updatestate_depend_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateDependEliminater());
  opt::OptPassConfig updatestate_assign_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateAssignEliminater());
  opt::OptPassConfig updatestate_loads_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateLoadsEliminater());
  opt::OptPassGroupMap elim_map({
    {"updatestate_eliminater", updatestate_eliminater},
    {"updatestate_depend_eliminate", updatestate_depend_eliminate},
    {"updatestate_assign_eliminate", updatestate_assign_eliminate},
    {"updatestate_loads_eliminate", updatestate_loads_eliminate},
    {"auto_monad_eliminator", opt::OptPassConfig(opt::AutoMonadEliminator())},
  });

  auto auto_monad_elim_opt = opt::Optimizer::MakeOptimizer("auto_monad_elim", res, elim_map);
  (void)auto_monad_elim_opt->step(func_graph, false);
  return true;
}

std::vector<PassItem> kVmPasses = {{"simplify_data_structures", SimplifyDataStructuresPass},
                                   {"opt_before_recompute", OptBeforeRecomputeGroup},
                                   {"opt_a", OptPassAGroup},
                                   {"clean_after_opta", CleanAfterOptAPass},
                                   {"opt_b", OptPassBGroup},
                                   {"cconv", CconvPass},
                                   {"opt_after_cconv", OptPassAfterCconvGroup},
                                   {"remove_dup_value", RemoveValueNodeDuplicationsPass},
                                   {"tuple_transform", OptPassTransformGraphGroup},
                                   {"add_cache_embedding", AddCacheEmbeddingPass},
                                   {"add_recomputation", AddRecomputationPass},
                                   {"cse_after_recomputation", OptAfterRecomputeGroup}};

std::vector<PassItem> kGePasses = {{"simplify_data_structures", SimplifyDataStructuresPass},
                                   {"opt_a", OptPassAGroup},
                                   {"clean_after_opta", CleanAfterOptAPass},
                                   {"opt_b", OptPassBGroup},
                                   {"opt_control", ControlGroup},
                                   {"opt_prepare", PrepareGroup},
                                   {"cconv", CconvPass}};

std::vector<PassItem> kPynativePasses = {{"opt_a", OptPassAGroup},
                                         {"opt_b", OptPassBGroup},
                                         {"cconv", CconvPass},
                                         {"transform_top", TransformTopGraphPass},
                                         {"transform_graph", OptPassTransformGraphGroup}};

std::vector<PassItem> kInlinePasses = {{"simplify_data_structures", SimplifyDataStructuresPass}, {"a1a2", OptPassA1A2}};
}  // namespace pipeline
}  // namespace mindspore
