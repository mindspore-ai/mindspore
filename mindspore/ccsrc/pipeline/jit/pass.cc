/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "utils/hash_map.h"
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
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "frontend/parallel/cache_embedding/cache_embedding.h"
#include "frontend/parallel/cache_embedding/ps_embedding_cache_inserter.h"
#include "frontend/parallel/allreduce_fusion/step_allreduce_fusion.h"
#include "frontend/parallel/pynative_shard/pynative_shard.h"
#include "frontend/optimizer/recompute.h"
#include "frontend/optimizer/slice_activation_in_recompute.h"
#include "frontend/optimizer/micro_interleaved_order_control.h"
#include "frontend/optimizer/comm_op_attrs.h"
#include "frontend/optimizer/environ_conversion.h"
#include "frontend/optimizer/comm_op_reuse_tag.h"
#include "utils/log_adapter.h"
#include "pipeline/jit/pipeline_split.h"
#include "pipeline/pynative/pynative_execute.h"
#include "pipeline/jit/static_analysis/auto_monad.h"
#include "frontend/optimizer/irpass/branch_culling.h"
#include "frontend/optimizer/irpass/meta_fg_eliminate.h"
#include "frontend/optimizer/irpass/ge/ge_specialized_prepare.h"
#include "frontend/optimizer/irpass/gradient_eliminate.h"
#include "frontend/optimizer/irpass/shard_eliminate.h"
#include "frontend/optimizer/irpass/taylor_eliminate.h"
#include "frontend/optimizer/irpass/parameter_eliminate.h"
#include "frontend/optimizer/irpass/updatestate_eliminate.h"
#include "frontend/optimizer/irpass/expand_dump_flag.h"
#include "frontend/optimizer/irpass/ge/batchnorm_transform.h"
#ifdef WITH_BACKEND
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
void UpdateArgsSpec(const FuncGraphPtr &func_graph, const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(resource);
  abstract::AbstractBasePtrList args_abs;
  const auto &parameters = func_graph->parameters();
  args_abs.reserve(parameters.size());
  (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_abs),
                       [](const AnfNodePtr &p) { return p->abstract(); });
  resource->set_args_abs(args_abs);
}
}  // namespace

bool BatchNormTransformPass(const ResourcePtr &resource) {
  // Transform only work in train process;
  auto env_ge = common::GetEnv("MS_GE_TRAIN");
  if (env_ge != "1") {
    MS_LOG(INFO) << "no need transfer batch norm in inference process";
    return true;
  }
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  opt::irpass::BatchNormTransform(func_graph, resource->manager());
  return true;
}

bool SimplifyDataStructuresPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  (void)opt::SimplifyDataStructures(func_graph, resource->manager());
  UpdateArgsSpec(func_graph, resource);
  return true;
}

bool TransformTopGraphPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "Transform top graph error.";
  }
  FuncGraphPtr func_graph = resource->func_graph();
  if (opt::FuncGraphHasTupleInput(func_graph)) {
    opt::GraphTupleParamTransform graph_trans;
    func_graph = graph_trans(func_graph, resource->manager());
    resource->set_func_graph(func_graph);
    AbstractBasePtrList abs_spec_list;
    auto &params = func_graph->parameters();
    std::transform(params.begin(), params.end(), std::back_inserter(abs_spec_list),
                   [](const AnfNodePtr &node) { return node->abstract(); });
    resource->set_args_abs(abs_spec_list);
  }
  return true;
}

bool CleanAfterOptAPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  (void)opt::CleanAfterOptA(func_graph, resource->manager());
  UpdateArgsSpec(func_graph, resource);
  return true;
}

FuncGraphPtr PrimBpOptPassStep1(const opt::irpass::OptimizeIRPassLib &irpass, const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_EXCEPTION_IF_NULL(resource->func_graph());
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

  auto prim_bprop_opt_step_1 = opt::Optimizer::MakeOptimizer("prim_bprop_opt_step_1", resource, map);
  FuncGraphPtr func_graph = resource->func_graph();
  WITH(MsProfile::GetProfile()->Step("prim_bprop_opt_step_1"))[&prim_bprop_opt_step_1, &func_graph]() {
    func_graph = prim_bprop_opt_step_1->step(func_graph, true);
  };
  return func_graph;
}

FuncGraphPtr PrimBpOptPassStep2(const opt::irpass::OptimizeIRPassLib &irpass, const ResourcePtr &resource,
                                const std::vector<bool> &need_grad_flags) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_EXCEPTION_IF_NULL(resource->func_graph());
  OptPassGroupMap map;

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

  map.push_back({"ad_renormalize", opt::OptPassConfig::Renormalize()});
  map.push_back({"ad_inline", inline_opt});
  map.push_back({"ad_special_op_simplify", special_op_simplify});
  map.push_back({"auto_monad_grad", opt::OptPassConfig(re_auto_monadwrapper)});
  if (!need_grad_flags.empty()) {
    // If func graph has not need_grad_flag_of_inputs attr, this graph has no need do this pass.
    opt::OptPassConfig pynative_no_grad_eliminate = opt::OptPassConfig({
      irpass.pynative_no_grad_eliminate_,
    });

    map.push_back({"pynative_no_grad_eliminate", pynative_no_grad_eliminate});
  }

  auto prim_bprop_opt_step_2 = opt::Optimizer::MakeOptimizer("prim_bprop_opt_step_2", resource, map);
  FuncGraphPtr func_graph = resource->func_graph();
  WITH(MsProfile::GetProfile()->Step("prim_bprop_opt_step_2"))[&prim_bprop_opt_step_2, &func_graph]() {
    func_graph = prim_bprop_opt_step_2->step(func_graph, true);
  };
  return func_graph;
}

FuncGraphPtr BpropGraphFinalOptPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_EXCEPTION_IF_NULL(resource->func_graph());
  (void)TransformTopGraphPass(resource);

  auto func_graph = resource->func_graph();
  // PyNative dynamic shape need add those pass, like convert make_list to make_tuple.
  // Cannot execute those pass due to performance reasons if the graph is a dynamic structure graph.
  MS_EXCEPTION_IF_NULL(func_graph);
  if (func_graph->has_flag(FUNC_GRAPH_FLAG_DYNAMIC_SHAPE) || !func_graph->has_flag(kFlagIsDynamicStructure)) {
    (void)OptPassAGroup(resource);
    (void)CleanAfterOptAPass(resource);
  }
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
    irpass.ad_related_special_op_eliminate_,
  });
  opt::OptPassConfig fill_zeros_like = opt::OptPassConfig{irpass.zero_like_fill_zero_};
  OptPassGroupMap map({
    {"ad_final_opt", bg_final_opt},
    {"zeros_like", fill_zeros_like},
  });

  if (pynative::PyNativeExecutor::GetInstance()->grad_executor()->need_renormalize()) {
    (void)map.emplace_back(std::make_pair("renormalize", opt::OptPassConfig::Renormalize()));
    opt::OptPassConfig real_op_eliminate = opt::OptPassConfig{irpass.real_op_eliminate_};
    (void)map.emplace_back(std::make_pair("real_op_eliminate", real_op_eliminate));
    opt::OptPassConfig environ_eliminate = opt::OptPassConfig({
      irpass.incorporate_call_,
      irpass.incorporate_call_switch_,
    });
    (void)map.emplace_back(std::make_pair("environ_eliminate", environ_eliminate));
  }

  auto bprop_graph_final_opt = opt::Optimizer::MakeOptimizer("bprop_graph_final_opt", resource, map);
  func_graph = resource->func_graph();
  WITH(MsProfile::GetProfile()->Step("bprop_graph_final_opt"))[&bprop_graph_final_opt, &func_graph]() {
    func_graph = bprop_graph_final_opt->step(func_graph, true);
  };
  func_graph = LiftingClone(func_graph);
  Validate(func_graph);
  return func_graph;
}

namespace {
bool ReAutoMonadWrapper(const FuncGraphPtr &root, const opt::OptimizerPtr &) { return ReAutoMonad(root); }

bool parallel_mode() {
#ifdef WITH_BACKEND
  if (ps::PSContext::instance()->is_server() || ps::PSContext::instance()->is_scheduler()) {
    return false;
  }
#endif
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  return (parallel_mode == parallel::kAutoParallel) || (parallel_mode == parallel::kSemiAutoParallel);
}

void AddParallelRenormalize(OptPassGroupMap *map_a) {
  if (parallel_mode()) {
    auto parallel_end_opt =
      find_if(map_a->begin(), map_a->end(), [](auto opt_pair) { return opt_pair.first == "meta_fg_expand"; });
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
    irpass.make_slice_get_slice_eliminator_,
    irpass.tuple_list_get_item_const_eliminator_,
    irpass.tuple_list_set_item_eliminator_,
    irpass.tuple_list_get_set_item_eliminator_,
    irpass.tuple_list_get_item_depend_reorder_,
    irpass.tuple_list_convert_item_index_to_positive_,

    irpass.environ_get_eliminate_,
    irpass.environ_get_add_eliminate_,
    irpass.environ_get_set_eliminate_,
    irpass.environ_get_depend_swap_,
    irpass.environ_add_const_eliminate_,

    irpass.cast_eliminate_,
    irpass.reshape_eliminate_,
    irpass.reduce_eliminate_,
    irpass.tile_eliminate_,
    irpass.transpose_eliminate_,
    irpass.minmaximum_grad_,

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
  });
}

opt::OptPassConfig GetGeTensorArrayPass(const opt::irpass::OptimizeIRPassLib &irpass) {
  return opt::OptPassConfig({
    irpass.ge_tensor_array_add_flow_,
    irpass.ge_tensor_array_cast_index_,
  });
}

OptPassGroupMap GetOptPassesA(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig a_1 = GetOptPassA1(irpass);
  opt::OptPassConfig a_2 = opt::OptPassConfig(
    {
      irpass.switch_simplify_,
      irpass.specialize_transform_,
      irpass.merge_addn_,
      irpass.compare_switch_simplify_,
      irpass.addn_check_dump_,
      irpass.float_tuple_getitem_switch_,
      irpass.float_environ_get_switch_,
      irpass.inline_,
      irpass.updatestate_useless_node_eliminater_,
      irpass.tuple_list_set_item_eliminator_,
      irpass.tuple_list_get_item_eliminator_,
      irpass.incorporate_call_,
      irpass.incorporate_call_switch_,
      irpass.environ_get_eliminate_,
      irpass.depend_value_elim_,
      irpass.all_reduce_const_elim_,
    },
    false, true);

  opt::OptPassConfig a_after_grad = opt::OptPassConfig({irpass.inline_without_move_, irpass.stack_unstack_eliminate_});

  opt::OptPassConfig a_3 = opt::OptPassConfig(
    {
      irpass.arithmetic_simplify2_,
      irpass.same_eliminate_,
      irpass.check_bprop_eliminate_,
      irpass.switch_layer_defer_inline_,
      irpass.replace_applicator_,
      irpass.row_tensor_add_zeros_like_,
      irpass.mini_step_allgather_replace_,
      irpass.micro_step_allgather_replace_,
      irpass.split_environ_get_set_with_tuple_value_,
    },
    false, true);
  opt::OptPassConfig accelerated_algorithm = opt::OptPassConfig({irpass.less_batch_normalization_});
  opt::OptPassConfig virtual_dataset = opt::OptPassConfig({irpass.virtual_dataset_eliminate_});
  opt::OptPassConfig after_resolve_pass = opt::OptPassConfig({irpass.replace_old_param_});
  opt::OptPassConfig updatestate_depend_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateDependEliminater());
  opt::OptPassConfig updatestate_assign_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateAssignEliminater());
  opt::OptPassConfig updatestate_loads_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateLoadsEliminater());
  opt::OptPassConfig recompute_prepare = opt::OptPassConfig({irpass.set_cell_output_no_recompute_});

  // Before adjusting map_a, check GetA1A2() and GetOptPynativeGradEpiloguePhases().
  OptPassGroupMap map_a({{"expand_dump_flag", opt::OptPassConfig(opt::irpass::ExpandDumpFlag())},
                         {"switch_simplify", opt::OptPassConfig({irpass.switch_simplify_})},
                         {"a_1", a_1},
                         {"recompute_prepare", recompute_prepare},
                         {"updatestate_depend_eliminate", updatestate_depend_eliminate},
                         {"updatestate_assign_eliminate", updatestate_assign_eliminate},
                         {"updatestate_loads_eliminate", updatestate_loads_eliminate},
                         {"parameter_eliminate", opt::OptPassConfig(opt::irpass::ParameterEliminator())},
                         {"a_2", a_2},
                         {"accelerated_algorithm", accelerated_algorithm},
                         {"pynative_shard", opt::OptPassConfig(parallel::PynativeShard)},
                         {"auto_parallel", opt::OptPassConfig(parallel::StepAutoParallel)},
                         {"parallel", opt::OptPassConfig(parallel::StepParallel)},
                         {"allreduce_fusion", opt::OptPassConfig(parallel::StepAllreduceFusion)},
                         {"virtual_dataset", virtual_dataset},
                         {"virtual_output", opt::OptPassConfig({irpass.virtual_output_eliminate_})},
                         {"meta_fg_expand", opt::OptPassConfig(opt::irpass::ExpandMetaFg())},
                         {"after_resolve", after_resolve_pass},
                         {"a_after_grad", a_after_grad},
                         {"renormalize", opt::OptPassConfig::Renormalize()},
                         {"real_op_eliminate", opt::OptPassConfig({irpass.real_op_eliminate_})},
                         {"auto_monad_grad", opt::OptPassConfig(ReAutoMonadWrapper)},
                         {"auto_monad_eliminator", opt::OptPassConfig(opt::AutoMonadEliminator())},
                         {"cse", opt::OptPassConfig(opt::CSEPass(false))},
                         {"a_3", a_3}});
  AddParallelRenormalize(&map_a);
  return map_a;
}

OptPassGroupMap GetA1A2(const opt::irpass::OptimizeIRPassLib &irpass) {
  auto opt_a = GetOptPassesA(irpass);
  constexpr auto a1_a2_len = 9;
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
  opt::OptPassConfig d_1 = opt::OptPassConfig({
    irpass.call_graph_tuple_transform_,
    irpass.tuple_list_get_item_eliminator_,
    irpass.tuple_list_get_item_const_eliminator_,
    irpass.tuple_list_set_item_eliminator_,
    irpass.tuple_list_get_set_item_eliminator_,
    irpass.tuple_list_get_item_depend_reorder_,
    irpass.tuple_list_convert_item_index_to_positive_,
  });

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
                                               irpass.make_slice_get_slice_eliminator_,
                                               irpass.float_tuple_getitem_switch_,
                                               irpass.reset_defer_inline_,
                                               irpass.inline_,
                                               irpass.updatestate_useless_node_eliminater_,
                                               irpass.updatestate_pure_node_eliminater_,
                                               irpass.load_eliminater_,
                                               irpass.stopgrad_eliminater_,
                                               irpass.special_op_eliminate_,
                                               irpass.environ_get_eliminate_,
                                               irpass.environ_get_add_eliminate_,
                                               irpass.environ_get_set_eliminate_,
                                               irpass.environ_get_depend_swap_,
                                               irpass.environ_add_const_eliminate_,
                                               irpass.value_based_eliminate_,
                                               irpass.parallel_virtual_node_},
                                              false, true);
  opt::OptPassConfig b_2 = opt::OptPassConfig({
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
    // After CleanAfterOptA, it may need renormalize to eliminate unused elements in Tuple.
    {"renormalize", opt::OptPassConfig::Renormalize()},
    {"control_group", control_group},
    {"renormalize", opt::OptPassConfig::Renormalize()},
  });
  return map;
}

OptPassGroupMap GetGeSpecializedPhases() {
  opt::OptPassConfig ge_ta_size_group = opt::OptPassConfig(opt::irpass::GeTensorArrayPrepare());
  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig ge_tensor_array_passes = GetGeTensorArrayPass(irpass);
  OptPassGroupMap map({
    {"ge_ta_size_group", ge_ta_size_group},
    {"ge_ta_passes", ge_tensor_array_passes},
  });
  return map;
}

OptPassGroupMap GetAvgPoolGradForGEPass(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig avg_pool_grad = opt::OptPassConfig({irpass.avg_pool_grad_for_ge_});
  OptPassGroupMap map({{"avg_pool_grad_for_ge", avg_pool_grad}});
  return map;
}

OptPassGroupMap GetDropoutForGEPass(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig dropout = opt::OptPassConfig({
    irpass.dropout_for_ge_,
    irpass.dropout_grad_for_ge_,
  });
  OptPassGroupMap map({{"dropout_for_ge", dropout}});
  return map;
}

OptPassGroupMap GetLambForGEPass(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig lamb = opt::OptPassConfig({irpass.lamb_for_ge_});
  OptPassGroupMap map({{"lamb_for_ge", lamb}});
  return map;
}

OptPassGroupMap GetClipByNormForGEPass(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig clip_by_norm = opt::OptPassConfig({irpass.clip_by_norm_for_ge_});
  OptPassGroupMap map({{"clip_by_norm_for_ge", clip_by_norm}});
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

OptPassGroupMap GetMetaUnpackPreparePhases() {
  opt::irpass::MetaUnpackPrepareLib irpass;
  auto meta_unpack_prepare = opt::OptPassConfig({irpass.meta_unpack_prepare_});
  opt::OptPassGroupMap prepare_map({{"meta_unpack_prepare", meta_unpack_prepare}});
  return prepare_map;
}

OptPassGroupMap GetPreparePhases(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig prepare_group = opt::OptPassConfig({irpass.print_tuple_wrapper_});
  OptPassGroupMap map({{"prepare_group", prepare_group}});
  return map;
}

OptPassGroupMap GetAfterRecomputePass(const opt::irpass::OptimizeIRPassLib &) {
  OptPassGroupMap map({{"cse", opt::OptPassConfig(opt::CSEPass(false))}});
  return map;
}

OptPassGroupMap GetSparseSoftmaxCrossEntropyWithLogitsSplitPass(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig sparse_split = opt::OptPassConfig({irpass.sparse_softmax_cross_entropy_with_logits_});
  OptPassGroupMap map({{"sparse_split", sparse_split}});
  return map;
}

static mindspore::HashMap<std::string, std::shared_ptr<Optimizer>> g_pass_opts = {};

void InitOpt(const ResourcePtr &resource) {
  if (g_pass_opts.size() == 0) {
    opt::irpass::OptimizeIRPassLib irpass;
    g_pass_opts["a1a2"] = Optimizer::MakeOptimizer("a1a2", resource, GetA1A2(irpass));
    g_pass_opts["opt_a"] = Optimizer::MakeOptimizer("opt_a", resource, GetOptPassesA(irpass));
    g_pass_opts["opt_b"] = Optimizer::MakeOptimizer("opt_b", resource, GetOptPassesB(irpass), false, true);
    g_pass_opts["opt_after_cconv"] =
      Optimizer::MakeOptimizer("opt_after_cconv", resource, GetOptPassesAfterCconv(irpass), false, true);
    g_pass_opts["opt_trans_graph"] =
      Optimizer::MakeOptimizer("opt_trans_graph", resource, GetOptPassesTransformGraph(irpass), true, true);
    g_pass_opts["renormal"] = Optimizer::MakeOptimizer("renormal", resource, GetOptPassesC(irpass));
    g_pass_opts["opt_control"] =
      Optimizer::MakeOptimizer("opt_control", resource, GetControlPhases(irpass), true, false);
    g_pass_opts["opt_grad_epilogue"] =
      Optimizer::MakeOptimizer("opt_grad_epilogue", resource, GetOptPynativeGradEpiloguePhases(irpass), true, false);
    g_pass_opts["opt_prepare"] = Optimizer::MakeOptimizer("opt_prepare", resource, GetPreparePhases(irpass));
    g_pass_opts["opt_after_recompute"] =
      Optimizer::MakeOptimizer("opt_after_recompute", resource, GetAfterRecomputePass(irpass));
    g_pass_opts["sparse_split"] =
      Optimizer::MakeOptimizer("sparse_spilt", resource, GetSparseSoftmaxCrossEntropyWithLogitsSplitPass(irpass));
    g_pass_opts["avg_pool_grad_for_ge"] =
      Optimizer::MakeOptimizer("avg_pool_grad_for_ge", resource, GetAvgPoolGradForGEPass(irpass));
    g_pass_opts["dropout_for_ge"] = Optimizer::MakeOptimizer("dropout_for_ge", resource, GetDropoutForGEPass(irpass));
    g_pass_opts["lamb_for_ge"] = Optimizer::MakeOptimizer("lamb_for_ge", resource, GetLambForGEPass(irpass));
    g_pass_opts["clip_by_norm_for_ge"] =
      Optimizer::MakeOptimizer("clip_by_norm_for_ge", resource, GetClipByNormForGEPass(irpass));
  }
}
}  // namespace

void ReclaimOptimizer() {
  for (auto &opt : g_pass_opts) {
    opt.second = nullptr;
  }
  g_pass_opts.clear();
}

bool OptPassGroup(const ResourcePtr &resource, const std::string &name) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->func_graph() == nullptr) {
    MS_LOG(ERROR) << "Opt passes int64_t error";
    return false;
  }

  FuncGraphPtr func_graph = resource->func_graph();
  MS_LOG(DEBUG) << "Start " << name << " func graph:" << func_graph->ToString() << ", "
                << func_graph->get_return()->DebugString(true);
  InitOpt(resource);
  if (g_pass_opts.find(name) != g_pass_opts.end()) {
    resource->set_func_graph(g_pass_opts[name]->step(func_graph));
  }
  // Note: StepParallel may modify the AbstractValue of the parameters of func_graph, but they are not updated to
  // resource->args_abs_ yet. So if any later pass or action want to use that variable, it should be set here.
  return true;
}

bool OptPassA1A2(const ResourcePtr &resource) { return OptPassGroup(resource, "a1a2"); }
bool OptPassAGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_a"); }
bool OptPassBGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_b"); }
bool OptPassAfterCconvGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_after_cconv"); }
bool OptPassTransformGraphGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_trans_graph"); }
bool ControlGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_control"); }
bool PrepareGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_prepare"); }
bool OptAfterRecomputeGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_after_recompute"); }
bool SparseSplitPass(const ResourcePtr &resource) { return OptPassGroup(resource, "sparse_split"); }
bool AvgPoolGradForGEPass(const ResourcePtr &resource) { return OptPassGroup(resource, "avg_pool_grad_for_ge"); }
bool DropoutGradForGEPass(const ResourcePtr &resource) { return OptPassGroup(resource, "dropout_for_ge"); }
bool LambForGEPass(const ResourcePtr &resource) { return OptPassGroup(resource, "lamb_for_ge"); }
bool ClipByNormForGEPass(const ResourcePtr &resource) { return OptPassGroup(resource, "clip_by_norm_for_ge"); }

bool OptPassRNGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "renormal"); }

bool OptPassGradEpilogueGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_grad_epilogue"); }

bool AddRecomputationPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  opt::InsertRecomputedNodes(resource->func_graph());
  return true;
}

bool SliceRecomputeActivationPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  opt::SliceRecomputedActivationNodes(resource->func_graph());
  return true;
}

bool MicroInterLeavedOrderControlPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  opt::MicroInterleavedOrderControl(resource->func_graph());
  return true;
}

bool CommOpAddAttrs(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  opt::CommOpAttrs(resource->func_graph());
  return true;
}

bool AddCommOpReusePass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  opt::AddCommOpReuseTag(resource->func_graph());
  return true;
}

bool AddCacheEmbeddingPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
#ifdef WITH_BACKEND
  if (ps::PSContext::instance()->is_ps_mode()) {
    return true;
  }
#endif
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);

  parallel::AddCacheEmbedding(func_graph);
  if (func_graph->has_flag(GRAPH_FLAG_CACHE_ENABLE)) {
    auto params = func_graph->parameters();
    AbstractBasePtrList args_spec_list;
    std::for_each(params.begin(), params.end(),
                  [&args_spec_list](const AnfNodePtr &node) { args_spec_list.push_back(node->abstract()); });
    func_graph = pipeline::Renormalize(resource, func_graph, args_spec_list);
  }
  return true;
}

bool RemoveValueNodeDuplicationsPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "Remove value node duplications error.";
  }
  auto manager = resource->manager();
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

bool CconvPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_EXCEPTION_IF_NULL(resource->func_graph());
  FuncGraphPtr func_graph = resource->func_graph();
  FuncGraphPtr new_fg = LiftingClone(func_graph);
  resource->set_func_graph(new_fg);
  return true;
}

bool PipelineSplitPass(const ResourcePtr &resource) { return PipelineSplit(resource); }

bool GeSpecializedPass(const ResourcePtr &resource) {
  // valid null ptr
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  // get phases
  auto ge_specialized_map = GetGeSpecializedPhases();
  auto ge_specialized_opt = opt::Optimizer::MakeOptimizer("ge_specialized", resource, ge_specialized_map, true);
  (void)ge_specialized_opt->step(func_graph, false);
  return true;
}

bool ValidatePass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_EXCEPTION_IF_NULL(resource->func_graph());
  FuncGraphPtr func_graph = resource->func_graph();
  Validate(func_graph);
  return true;
}

bool MetaUnpackPreparePass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto prepare_map = GetMetaUnpackPreparePhases();
  auto infer_opt_prepare = opt::Optimizer::MakeOptimizer("meta_unpack_prepare", resource, prepare_map);
  (void)infer_opt_prepare->step(func_graph, false);
  return true;
}

bool PynativeOptPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  opt::irpass::OptimizeIRPassLib irpass;
  auto pynative_opt = GetOptPassesPynativeElim(irpass);
  auto pynative_opt_opt = opt::Optimizer::MakeOptimizer("pynative_opt", resource, pynative_opt);
  (void)pynative_opt_opt->step(func_graph, false);
  return true;
}

bool EliminateAdRelatedSpecialOpOptPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig ad_related_special_op_eliminate = opt::OptPassConfig({
    irpass.ad_related_special_op_eliminate_,
  });
  OptPassGroupMap map({
    {"ad_related_special_op_eliminate", ad_related_special_op_eliminate},
  });
  auto ad_related_special_op_eliminate_opt =
    opt::Optimizer::MakeOptimizer("ad_related_special_op_eliminate", resource, map);
  (void)ad_related_special_op_eliminate_opt->step(func_graph, false);
  return true;
}

bool AutoMonadElimOptPass(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->manager());
  auto resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(func_graph);
  resource->set_manager(func_graph->manager());

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

  auto auto_monad_elim_opt = opt::Optimizer::MakeOptimizer("auto_monad_elim", resource, elim_map);
  (void)auto_monad_elim_opt->step(func_graph, false);
  return true;
}

bool EnvironConversionPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  (void)opt::EnvironConversion(resource);
  return true;
}

// Build service-side graph for embedding distributed cache based on Parameter Server.
bool AddEmbeddingCachePass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
  if (!ps::PSContext::instance()->cache_enable() || !distributed::cluster::ClusterContext::instance()->initialized() ||
      !ps::PSContext::instance()->is_server()) {
    return true;
  }

  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto node = distributed::cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);

  // 1. Build service-size graph.
  auto node_role = distributed::cluster::ClusterContext::instance()->node_role();
  uint32_t worker_num = ps::PSContext::instance()->worker_num();
  std::shared_ptr<parallel::PsEmbeddingCacheInserter> embedding_cache_inserter =
    std::make_shared<parallel::PsEmbeddingCacheInserter>(func_graph, static_cast<int64_t>(node->rank_id()), node_role,
                                                         worker_num);
  if (!embedding_cache_inserter->Run()) {
    MS_LOG(ERROR) << "Insert ps embedding cache failed.";
    return false;
  }

  // 2. Renomalize: Infer shape and Set abstract for all nodes in graph.
  abstract::AbstractBasePtrList args_abs;
  auto parameters = func_graph->parameters();
  (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_abs),
                       [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
  FuncGraphPtr new_fg = Renormalize(resource, func_graph, args_abs);
  resource->set_func_graph(new_fg);
  resource->set_args_abs(args_abs);
#endif

  return true;
}

std::vector<PassItem> kVmPasses = {
  {"simplify_data_structures", SimplifyDataStructuresPass},
  {"opt_a", OptPassAGroup},
  {"clean_after_opta", CleanAfterOptAPass},
  {"opt_b", OptPassBGroup},
  {"cconv", CconvPass},
  {"opt_after_cconv", OptPassAfterCconvGroup},
  {"remove_dup_value", RemoveValueNodeDuplicationsPass},
  {"tuple_transform", OptPassTransformGraphGroup},
  {"add_cache_embedding", AddCacheEmbeddingPass},
  {"add_recomputation", AddRecomputationPass},
  {"cse_after_recomputation", OptAfterRecomputeGroup},
  {"environ_conv", EnvironConversionPass},
  {"slice_recompute_activation", SliceRecomputeActivationPass},
  {"micro_interleaved_order_control", MicroInterLeavedOrderControlPass},
  {"comm_op_add_attrs", CommOpAddAttrs},
  {"add_comm_op_reuse_tag", AddCommOpReusePass},
};

std::vector<PassItem> kGePasses = {{"simplify_data_structures", SimplifyDataStructuresPass},
                                   {"batchnorm_transform", BatchNormTransformPass},
                                   {"opt_a", OptPassAGroup},
                                   {"clean_after_opta", CleanAfterOptAPass},
                                   {"opt_b", OptPassBGroup},
                                   {"opt_control", ControlGroup},
                                   {"opt_prepare", PrepareGroup},
                                   {"sparse_split", SparseSplitPass},
                                   {"avg_pool_grad_for_ge", AvgPoolGradForGEPass},
                                   {"dropout_for_ge", DropoutGradForGEPass},
                                   {"lamb_for_ge", LambForGEPass},
                                   {"clip_by_norm_for_ge", ClipByNormForGEPass},
                                   {"cconv", CconvPass}};

std::vector<PassItem> kPynativePasses = {{"opt_a", OptPassAGroup},
                                         {"opt_b", OptPassBGroup},
                                         {"cconv", CconvPass},
                                         {"transform_top", TransformTopGraphPass},
                                         {"transform_graph", OptPassTransformGraphGroup}};

std::vector<PassItem> kInlinePasses = {{"simplify_data_structures", SimplifyDataStructuresPass}, {"a1a2", OptPassA1A2}};
}  // namespace pipeline
}  // namespace mindspore
