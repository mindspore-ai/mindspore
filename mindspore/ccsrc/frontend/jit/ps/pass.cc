/**
 * Copyright 2019-2026 Huawei Technologies Co., Ltd
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

#include "frontend/jit/ps/pass.h"

#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include "primitive/other_ops.h"
#include "primitive/framework_ops.h"
#include "utils/hash_map.h"
#include "ir/func_graph_cloner.h"
#include "include/frontend/jit/ps/pass_interface.h"
#include "include/frontend/jit/ps/action_interface.h"
#include "frontend/jit/ps/parse/parse_base.h"
#include "frontend/jit/ps/resource.h"
#include "frontend/jit/ps/validator.h"
#include "frontend/jit/ps/remove_value_node_dup.h"
#include "frontend/optimizer/opt.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/cse_pass.h"
#include "frontend/optimizer/fallback_rewriter.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/graph_transform.h"
#include "frontend/optimizer/auto_monad_eliminate.h"
#include "frontend/optimizer/utils.h"
#include "include/utils/pipeline/fallback.h"
#include "include/utils/parallel_context.h"
#include "include/utils/pynative/grad_state.h"
#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/pipeline_transformer/pipeline_scheduler.h"
#include "frontend/parallel/pipeline_transformer/detach_backward.h"
#include "frontend/parallel/pipeline_transformer/pipeline_interleave.h"
#include "frontend/parallel/pipeline_transformer/gpipe_interleave_scheduler.h"
#include "frontend/parallel/pass/merge_comm.h"
#include "frontend/parallel/pass/merge_send_recv.h"
#include "frontend/parallel/pass/merge_recompute_call_nodes.h"
#include "frontend/parallel/pass/set_forward_comm_id_for_comm_node.h"
#include "frontend/parallel/allreduce_fusion/step_allreduce_fusion.h"
#include "frontend/parallel/shard/shard.h"
#include "frontend/parallel/pass/optimize_parallel_allgather_comm.h"
#include "frontend/parallel/pass/label_micro_interleaved_index.h"
#include "frontend/parallel/pass/dataset_reader_optimizer.h"
#include "frontend/parallel/pass/label_fine_grained_interleaved_index.h"
#include "frontend/parallel/pass/reorder_send_recv_between_fp_bp.h"
#include "frontend/parallel/pass/micro_interleaved_order_control.h"
#include "frontend/parallel/pass/full_micro_interleaved_order_control.h"
#include "frontend/parallel/pass/overlap_recompute_allgather_and_flashattention_grad.h"
#include "frontend/parallel/pass/assign_add_opt.h"
#include "frontend/parallel/pass/float32_redistribution.h"
#include "frontend/parallel/pass/swap_dp_allreduce_reducescatter.h"
#include "frontend/parallel/pass/merge_cast_opt.h"
#include "frontend/parallel/pass/remove_cast_before_assign_add.h"
#include "frontend/parallel/pass/bias_add_comm_swap.h"
#include "frontend/parallel/pass/matmul_add_comm_reduction.h"
#include "frontend/parallel/pass/allreduce_slice_to_reducescatter.h"
#include "frontend/parallel/pass/overlap_opt_shard_in_pipeline.h"
#include "frontend/parallel/pass/slice_activation_in_cell_share_recompute.h"
#include "frontend/parallel/pass/handle_group_info.h"
#include "frontend/parallel/pass/overlap_recompute_and_grad_model_parallel.h"
#include "frontend/parallel/pass/overlap_gradmatmul_and_gradallreduce.h"
#include "frontend/parallel/pass/overlap_grad_ring_attention.h"
#include "frontend/parallel/pass/overlap_grad_flash_sp.h"
#include "frontend/parallel/pass/interleave_split_concat_branches.h"
#include "frontend/parallel/pass/interleave_parallel_branches.h"
#include "frontend/parallel/pass/begin_end_overlap_inline.h"
#include "frontend/parallel/pass/offload_activation.h"
#include "frontend/parallel/pass/split_matmul_comm_elementwise_fp.h"
#include "frontend/parallel/pass/split_layernorm_comm_fp.h"
#include "frontend/parallel/pipeline_transformer/pipeline_transformer.h"
#include "frontend/parallel/pass/overlap_grad_comm.h"
#include "frontend/parallel/pass/overlap_param_gather.h"
#include "frontend/parallel/pass/overlap_recompute_comm.h"
#include "frontend/optimizer/recompute.h"
#include "frontend/optimizer/irpass/recompute.h"
#include "frontend/parallel/pass/slice_activation_in_recompute.h"
#include "frontend/parallel/pass/grouped_pairwise_exchange_alltoall.h"
#include "frontend/parallel/pass/offloading_packed_expert.h"
#include "frontend/parallel/pass/comm_op_attrs.h"
#include "frontend/parallel/pass/process_send_recv_for_ge.h"
#include "include/frontend/optimizer/environ_conversion.h"
#include "frontend/parallel/pass/comm_op_reuse_tag.h"
#include "frontend/optimizer/py_interpret_to_execute.h"
#include "frontend/parallel/pass/flash_sp.h"
#include "frontend/parallel/pass/fias_sp.h"
#include "utils/log_adapter.h"
#include "utils/compile_config.h"
#include "frontend/jit/ps/pipeline_split.h"
#include "frontend/jit/ps/static_analysis/auto_monad.h"
#include "frontend/jit/ps/static_analysis/event_method.h"
#include "frontend/optimizer/irpass/branch_culling.h"
#include "frontend/optimizer/irpass/meta_fg_eliminate.h"
#include "frontend/optimizer/irpass/gradient_eliminate.h"
#include "frontend/optimizer/irpass/shard_eliminate.h"
#include "frontend/optimizer/irpass/taylor_eliminate.h"
#include "frontend/optimizer/irpass/parameter_eliminate.h"
#include "frontend/optimizer/irpass/updatestate_eliminate.h"
#include "frontend/optimizer/irpass/expand_dump_flag.h"
#include "frontend/optimizer/irpass/symbol_engine_optimizer.h"
#include "frontend/optimizer/irpass/add_forward_monad_depend.h"
#include "frontend/optimizer/irpass/virtualviewgrad_op.h"
#include "frontend/optimizer/irpass/virtualview_op.h"
#include "frontend/optimizer/irpass/inplace_input_replace.h"
#include "frontend/optimizer/irpass/isolate_inplace_func_replace.h"
#include "frontend/optimizer/irpass/with_stream_mark.h"
#include "frontend/jit/ps/pass_config.h"
#include "frontend/jit/ps/graph_circle_handler.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

#ifndef REGISTER_PASS_FUNC_IMPL
#define REGISTER_PASS_FUNC_IMPL(name)                                                                        \
  namespace {                                                                                                \
  static auto helper_pass_func_##name = opt::RegisterPassFunc(#name, opt::OptPassConfigLib::PassFunc(name)); \
  }
#endif

namespace mindspore {
namespace pipeline {
using OptPassGroupMap = opt::OptPassGroupMap;
using Optimizer = opt::Optimizer;
using abstract::AnalysisResult;
using mindspore::abstract::AnalysisContextPtr;
using mindspore::validator::Validate;
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

bool PyInterpretToExecutePass(const ResourcePtr &resource) {
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() == kLax);
  if (!allow_fallback_runtime) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  (void)opt::PyInterpretToExecute(resource);
  UpdateArgsSpec(func_graph, resource);
  return true;
}
REGISTER_PASS_FUNC_IMPL(PyInterpretToExecutePass)

bool RewriterBeforeOptAPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  (void)opt::RewriterBeforeOptA(func_graph, resource->manager());
  UpdateArgsSpec(func_graph, resource);
  return true;
}
REGISTER_PASS_FUNC_IMPL(RewriterBeforeOptAPass)

bool TransformTopGraphPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->func_graph() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Transform top graph error.";
  }
  FuncGraphPtr func_graph = resource->func_graph();
  if (opt::FuncGraphHasSequenceInput(func_graph)) {
    opt::GraphSequenceParamTransform graph_trans;
    func_graph = graph_trans(func_graph, resource->manager());
    resource->set_func_graph(func_graph);
    AbstractBasePtrList abs_spec_list;
    auto &params = func_graph->parameters();
    (void)std::transform(params.begin(), params.end(), std::back_inserter(abs_spec_list),
                         [](const AnfNodePtr &node) { return node->abstract(); });
    resource->set_args_abs(abs_spec_list);
  }
  return true;
}
REGISTER_PASS_FUNC_IMPL(TransformTopGraphPass)

bool RewriterAfterOptAPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  (void)opt::RewriterAfterOptA(func_graph, resource);
  UpdateArgsSpec(func_graph, resource);
  return true;
}
REGISTER_PASS_FUNC_IMPL(RewriterAfterOptAPass)

bool ConvertAfterRewriterPass(const ResourcePtr &resource) {
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  if (!allow_fallback_runtime) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  (void)opt::ConvertAfterRewriter(func_graph, resource);
  UpdateArgsSpec(func_graph, resource);
  return true;
}
REGISTER_PASS_FUNC_IMPL(ConvertAfterRewriterPass)

bool OrderPyExecuteAfterRewriterPass(const ResourcePtr &resource) {
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  if (!allow_fallback_runtime) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  (void)opt::OrderPyExecuteAfterRewriter(func_graph, resource);
  UpdateArgsSpec(func_graph, resource);
  return true;
}
REGISTER_PASS_FUNC_IMPL(OrderPyExecuteAfterRewriterPass)

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
  ProfileExecute(MsProfile::GetProfile()->Step("prim_bprop_opt_step_1"), [&prim_bprop_opt_step_1, &func_graph]() {
    func_graph = prim_bprop_opt_step_1->step(func_graph, true);
  });
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
  ProfileExecute(MsProfile::GetProfile()->Step("prim_bprop_opt_step_2"), [&prim_bprop_opt_step_2, &func_graph]() {
    func_graph = prim_bprop_opt_step_2->step(func_graph, true);
  });
  return func_graph;
}

FuncGraphPtr JitBpropGraphPass(const ResourcePtr &resource, bool need_renormalize) {
  opt::irpass::OptimizeIRPassLib irpass;
  OptPassGroupMap map;

  if (need_renormalize) {
    opt::OptPassConfig after_resolve_pass = opt::OptPassConfig({irpass.replace_old_param_});
    // Disable after_resolve_pass if Pre-Lift is enabled.
    static const bool enable_pre_lift = (common::GetCompileConfig("PRE_LIFT") == "1");
    if (enable_pre_lift) {
      after_resolve_pass.set_disabled(true);
    }
    opt::OptPassConfig a_after_grad =
      opt::OptPassConfig({irpass.inline_without_move_, irpass.stack_unstack_eliminate_});

    auto re_auto_monadwrapper = [](const FuncGraphPtr &root, const opt::OptimizerPtr &) -> bool {
      return ReAutoMonad(root);
    };

    (void)map.emplace_back("after_resolve", after_resolve_pass);
    (void)map.emplace_back("a_after_grad", a_after_grad);
    (void)map.emplace_back("renormalize", opt::OptPassConfig::Renormalize());
    (void)map.emplace_back("auto_monad_grad", opt::OptPassConfig(re_auto_monadwrapper));
    (void)map.emplace_back("auto_monad_eliminator", opt::OptPassConfig(opt::AutoMonadEliminator()));
  }

  opt::OptPassConfig grad_graph_opt = opt::OptPassConfig({
    irpass.reset_defer_inline_,
    irpass.inline_,
    irpass.list_to_tuple_eliminator_,
    irpass.tuple_to_list_eliminator_,
    irpass.tuple_list_get_set_item_eliminator_,
    irpass.tuple_list_get_item_eliminator_,
    irpass.tuple_list_set_item_eliminator_,
    irpass.depend_value_elim_,
    irpass.reshape_eliminate_,
    irpass.switch_simplify_,
    irpass.merge_addn_,
    irpass.addn_zero_filter_,
    irpass.ad_related_special_op_eliminate_,
    irpass.special_op_eliminate_,
    irpass.virtual_view_grad_op_eliminate_,
  });
  opt::OptPassConfig fill_zeros_like = opt::OptPassConfig{irpass.zero_like_fill_zero_};
  // In case custom bprop has meta fg need to expand, such as J.
  opt::OptPassConfig expand_meta_fg = opt::OptPassConfig{opt::irpass::ExpandMetaFg()};

  (void)map.emplace_back("grad_graph_opt", grad_graph_opt);
  (void)map.emplace_back("updatestate_depend_eliminate",
                         opt::OptPassConfig(opt::irpass::UpdatestateDependEliminater()));
  (void)map.emplace_back("zeros_like", fill_zeros_like);
  (void)map.emplace_back("expand_meta_fg", expand_meta_fg);

  MS_EXCEPTION_IF_NULL(resource);
  auto func_graph = resource->func_graph();
  auto graph_opt = opt::Optimizer::MakeOptimizer("jit_bprop_graph_opt", resource, map, false, false, false);
  auto optimized_fg = graph_opt->step(func_graph, false);
  auto lifted_fg = LiftingClone(optimized_fg);

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_save_graphs = context->CanDump(kIntroductory);
  if (enable_save_graphs) {
    DumpIR("jit_bprop_graph_lift_" + lifted_fg->ToString() + ".ir", lifted_fg);
  }
#endif

  return lifted_fg;
}

FuncGraphPtr HighGradBpropGraphPass(const ResourcePtr &resource) {
  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig grad_graph_opt = opt::OptPassConfig({
    irpass.pynative_gradjit_primitivepy_eliminate_,
  });
  OptPassGroupMap map({
    {"grad_graph_opt", grad_graph_opt},
  });
  MS_EXCEPTION_IF_NULL(resource);
  auto func_graph = resource->func_graph();
  auto graph_opt = opt::Optimizer::MakeOptimizer("high_grad_bprop_graph_opt", resource, map);
  return graph_opt->step(func_graph, false);
}

FuncGraphPtr FinalBpropGraphPass(const ResourcePtr &resource, bool has_control_flow) {
  MS_EXCEPTION_IF_NULL(resource);
  auto func_graph = resource->func_graph();

  opt::irpass::OptimizeIRPassLib irpass;
  OptPassGroupMap map;
  opt::OptPassConfig inline_opt = opt::OptPassConfig({
    irpass.inline_,
  });
  (void)map.emplace_back("ad_inline", inline_opt);

  opt::OptPassConfig grad_graph_opt = opt::OptPassConfig({
    irpass.tuple_list_get_item_eliminator_,
    irpass.zero_like_fill_zero_,
  });
  (void)map.emplace_back("grad_graph_opt", grad_graph_opt);

  if (has_control_flow) {
    opt::OptPassConfig env_eliminate = opt::OptPassConfig({
      irpass.environ_get_eliminate_,
      irpass.environ_get_add_eliminate_,
      irpass.environ_get_set_eliminate_,
      irpass.environ_get_depend_swap_,
      irpass.environ_add_const_eliminate_,
    });
    (void)map.emplace_back("env_eliminate", env_eliminate);
  }
  auto graph_opt = opt::Optimizer::MakeOptimizer("final_bprop_graph_opt", resource, map);
  return graph_opt->step(func_graph, false);
}

namespace {
bool ReAutoMonadWrapper(const FuncGraphPtr &root, const opt::OptimizerPtr &) { return ReAutoMonad(root); }
REGISTER_OPT_PASS_FUNC(ReAutoMonadWrapper)

bool OffloadActivationWrapper(const FuncGraphPtr &root, const opt::OptimizerPtr &) {
  return parallel::OffloadActivation(root);
}
REGISTER_OPT_PASS_FUNC(OffloadActivationWrapper)

bool EventMethodActionWrapper(const FuncGraphPtr &root, const opt::OptimizerPtr &) { return EventMethod(root); }
REGISTER_OPT_PASS_FUNC(EventMethodActionWrapper)

bool parallel_mode() {
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  return (parallel_mode == parallel::kAutoParallel) || (parallel_mode == parallel::kSemiAutoParallel);
}

bool HasMetaMorphosisCNode(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = resource->manager();
  MS_EXCEPTION_IF_NULL(mng);

  const auto &all_nodes = mng->all_nodes();
  return std::any_of(all_nodes.begin(), all_nodes.end(),
                     [](const auto &node) { return opt::irpass::IsMetamorphosisCNode(node); });
}

void AddMetaMorphosis(const ResourcePtr &resource, const std::string &add_before, const std::string &jump_to,
                      opt::OptPassGroupMap *map_a) {
  if (HasMetaMorphosisCNode(resource)) {
    auto it =
      find_if(map_a->begin(), map_a->end(), [&add_before](const auto &item) { return item.name == add_before; });
    if (it != map_a->end()) {
      opt::irpass::OptimizeIRPassLib irpass;
      opt::OptPassGroupMap map_meta_morph(
        {{"meta_morphosis", opt::OptPassConfig({irpass.meta_morphosis_}, true)},
         {"meta_morphosis_renormalize", opt::OptPassConfig::Renormalize(true)},
         {"meta_morphosis_event_method", opt::OptPassConfig(EventMethodActionWrapper, true)},
         {"meta_morphosis_auto_monad_grad", opt::OptPassConfig(ReAutoMonadWrapper, true), jump_to}});
      (void)map_a->insert(it, map_meta_morph.begin(), map_meta_morph.end());
    }
  }
}

void AddParallelRenormalize(OptPassGroupMap *map_a) {
  auto update_top_fg = [](const FuncGraphPtr &root, const opt::OptimizerPtr &) {
    parse::Parser::UpdateTopFuncGraph(root);
    return false;
  };
  if (parallel_mode()) {
    auto parallel_end_opt =
      find_if(map_a->begin(), map_a->end(), [](const auto &opt_pair) { return opt_pair.name == kMetaFgExpandFlag; });
    if (parallel_end_opt != map_a->end()) {
      opt::irpass::OptimizeIRPassLib irpass;
      opt::OptPassConfig cast_eliminate_pass = opt::OptPassConfig({irpass.cast_eliminate_});
      auto iter = map_a->insert(parallel_end_opt, {"cast_eliminate", cast_eliminate_pass});
      iter = map_a->insert(iter, {"update_top_fg", opt::OptPassConfig(update_top_fg)});
      (void)map_a->insert(iter, {"parallel_renormalize", opt::OptPassConfig::Renormalize(true)});
    }
  }
}

opt::OptPassConfig GetOptPassA1(const opt::irpass::OptimizeIRPassLib &irpass) {
  return opt::OptPassConfig({
    irpass.partial_defer_inline_,
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
    irpass.redundant_stopgrad_eliminater_,
    irpass.partial_eliminate_,
    irpass.replace_applicator_,

    // Miscellaneous
    irpass.list_to_tuple_eliminator_,
    irpass.tuple_to_list_eliminator_,
    irpass.tuple_list_get_item_eliminator_,
    irpass.make_slice_get_slice_eliminator_,
    irpass.tuple_list_get_item_const_eliminator_,
    irpass.tuple_list_set_item_eliminator_,
    irpass.tuple_list_get_set_item_eliminator_,
    irpass.tuple_list_get_item_depend_reorder_,
    irpass.tuple_list_convert_item_index_to_positive_,
    irpass.dict_get_item_eliminator_,
    irpass.dict_get_item_const_eliminator_,
    irpass.dict_set_item_eliminator_,

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
    irpass.redundant_stopgrad_eliminater_,
    irpass.print_const_string_wrapper_,
  });
}

bool FlashSPFrontPass(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer) {
  if (!func_graph->has_flag(parallel::FLASH_SP_RUN_ONCE_ONLY)) {
    auto result = parallel::SetFlashSP(func_graph);
    func_graph->set_flag(parallel::FLASH_SP_RUN_ONCE_ONLY, true);
    return result;
  }
  if (!func_graph->has_flag(parallel::FIAS_SP_RUN_ONCE_ONLY)) {
    auto result = parallel::SetFiasSP(func_graph);
    func_graph->set_flag(parallel::FIAS_SP_RUN_ONCE_ONLY, true);
    return result;
  }
  return false;
}

opt::OptPassConfig GetOptPassA2(const opt::irpass::OptimizeIRPassLib &irpass) {
  return opt::OptPassConfig(
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
      irpass.arithmetic_simplify_,
      irpass.tuple_list_set_item_eliminator_,
      irpass.tuple_list_get_item_eliminator_,
      irpass.incorporate_call_,
      irpass.incorporate_call_switch_,
      irpass.environ_get_eliminate_,
      irpass.depend_value_elim_,
      irpass.all_reduce_const_elim_,
    },
    false, true);
}

opt::OptPassConfig GetOptPassA3(const opt::irpass::OptimizeIRPassLib &irpass) {
  return opt::OptPassConfig(
    {
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
}

OptPassGroupMap GetOptPassesA(const opt::irpass::OptimizeIRPassLib &irpass, const ResourcePtr &resource) {
  opt::OptPassConfig a_1 = GetOptPassA1(irpass);
  opt::OptPassConfig a_2 = GetOptPassA2(irpass);

  opt::OptPassConfig before_grad = opt::OptPassConfig({irpass.j_node_and_user_rematch_});
  opt::OptPassConfig a_after_grad = opt::OptPassConfig({irpass.inline_without_move_, irpass.stack_unstack_eliminate_});
  opt::OptPassConfig a_3 = GetOptPassA3(irpass);
  opt::OptPassConfig accelerated_algorithm = opt::OptPassConfig({irpass.less_batch_normalization_});
  opt::OptPassConfig virtual_dataset = opt::OptPassConfig({irpass.virtual_dataset_eliminate_});
  opt::OptPassConfig after_resolve_pass = opt::OptPassConfig({irpass.replace_old_param_});
  // Disable after_resolve_pass if Pre-Lift is enabled.
  static const bool enable_pre_lift = (common::GetCompileConfig("PRE_LIFT") == "1");
  if (enable_pre_lift) {
    after_resolve_pass.set_disabled(true);
  }
  opt::OptPassConfig updatestate_depend_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateDependEliminater());
  opt::OptPassConfig updatestate_assign_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateAssignEliminater());
  opt::OptPassConfig updatestate_loads_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateLoadsEliminater());
  opt::OptPassConfig recompute_prepare = opt::OptPassConfig({irpass.set_cell_output_no_recompute_});
  opt::OptPassConfig get_grad = opt::OptPassConfig({irpass.get_grad_eliminate_});
  opt::OptPassConfig cell_reuse_handle_not_recompute_node_pass =
    opt::OptPassConfig({irpass.remove_not_recompute_node_}, false, true);
  opt::OptPassConfig c_1 = opt::OptPassConfig({
    irpass.switch_call_monad_eliminater_,
    irpass.partial_eliminate_,
  });
  // Disable c_1 if Pre-Lift is not enabled.
  if (!enable_pre_lift) {
    c_1.set_disabled(true);
  }
  // Before adjusting map_a, check GetA1A2() and GetOptPynativeGradEpiloguePhases().
  OptPassGroupMap map_a(
    {{kExpandDumpFlag, opt::OptPassConfig(opt::irpass::ExpandDumpFlag())},
     {kSwitchSimplifyFlag, opt::OptPassConfig({irpass.switch_simplify_})},
     {"loop_unroll", opt::OptPassConfig({irpass.loop_unroll_before_grad_})},
     {"a_1", a_1},
     {"with_stream_mark", opt::OptPassConfig(opt::OptPassConfig(opt::irpass::WithStreamMark))},
     {"recompute_prepare", recompute_prepare},
     {"updatestate_depend_eliminate", updatestate_depend_eliminate},
     {"updatestate_assign_eliminate", updatestate_assign_eliminate},
     {"updatestate_loads_eliminate", updatestate_loads_eliminate},
     {"c_1", c_1},
     {"parameter_eliminate", opt::OptPassConfig(opt::irpass::ParameterEliminator())},
     {"a_2", a_2},
     {"accelerated_algorithm", accelerated_algorithm},
     {"shard", opt::OptPassConfig(parallel::Shard)},
     {"meta_shard_fg_expand", opt::OptPassConfig(opt::irpass::ExpandMetaShardFg())},
     {"shard_inline", opt::OptPassConfig({irpass.inline_})},
     {"merge_send_recv", opt::OptPassConfig(parallel::MergeSendReceive)},
     {"auto_parallel", opt::OptPassConfig(parallel::StepAutoParallel)},
     {"parallel", opt::OptPassConfig(parallel::StepParallel)},
     {"flash_sp", opt::OptPassConfig(FlashSPFrontPass)},
     {"merge_comm", opt::OptPassConfig(parallel::MergeComm)},
     {"allreduce_fusion", opt::OptPassConfig(parallel::StepAllreduceFusion)},
     {"matmul_add_comm_reduction", opt::OptPassConfig(parallel::MatmulAddCommReduction)},
     {"allreduce_slice_to_reducescatter", opt::OptPassConfig(parallel::AllReduceSliceToReduceScatter)},
     {"virtual_shard_identity", opt::OptPassConfig({irpass.virtual_shard_identity_})},
     {"virtual_dataset", virtual_dataset},
     {"get_grad_eliminate_", get_grad},
     {"virtual_output", opt::OptPassConfig({irpass.virtual_output_eliminate_})},
     {"merge_forward", opt::OptPassConfig(ad::MergeForward)},
     {"cell_reuse_recompute_pass", opt::OptPassConfig(opt::irpass::Recomputation())},
     {"offload_activation", opt::OptPassConfig(OffloadActivationWrapper)},
     {"cell_reuse_handle_not_recompute_node_pass", cell_reuse_handle_not_recompute_node_pass},
     {"merge_recompute_call_nodes", opt::OptPassConfig(parallel::MergeRecomputeCallNodes)},
     {"before_grad", before_grad},
     {kSetForwardCommIdForCommNodePass, opt::OptPassConfig(parallel::SetForwardCommIdForCommNode)},
     {kMetaFgExpandFlag, opt::OptPassConfig(opt::irpass::ExpandMetaFg())},
     {"flash_sp_send_recv_attached", opt::OptPassConfig(parallel::FlashSPSendRecvNodeAttach)},
     {"receive_attached", opt::OptPassConfig(parallel::IsolatedNodeAttach)},
     {"after_resolve", after_resolve_pass},
     {"a_after_grad", a_after_grad},
     {"renormalize", opt::OptPassConfig::Renormalize()},
     {"add_forward_monad_depend", opt::OptPassConfig(opt::irpass::AddForwardMonadDepend)},
     {"auto_monad_grad", opt::OptPassConfig(ReAutoMonadWrapper)},
     {"auto_monad_eliminator", opt::OptPassConfig(opt::AutoMonadEliminator())},
     {"cse", opt::OptPassConfig(opt::CSEPass(false))},
     {"a_3", a_3}});
  AddMetaMorphosis(resource, kSetForwardCommIdForCommNodePass, kExpandDumpFlag, &map_a);
  AddParallelRenormalize(&map_a);
  return map_a;
}

opt::OptPassConfig GetJitOptPassA1(const opt::irpass::OptimizeIRPassLib &irpass) {
  return opt::OptPassConfig({
    irpass.switch_defer_inline_,
    irpass.switch_layer_defer_inline_,
    irpass.switch_simplify_,

    // Safe inlining
    irpass.inline_,
    irpass.updatestate_useless_node_eliminater_,
    irpass.updatestate_pure_node_eliminater_,
    irpass.load_eliminater_,
    irpass.redundant_stopgrad_eliminater_,
    irpass.partial_eliminate_,
    irpass.replace_applicator_,

    // Miscellaneous
    irpass.list_to_tuple_eliminator_,
    irpass.tuple_to_list_eliminator_,
    irpass.tuple_list_get_item_eliminator_,
    irpass.tuple_list_set_item_eliminator_,
    irpass.make_slice_get_slice_eliminator_,
    irpass.tuple_list_get_item_depend_reorder_,
    irpass.tuple_list_convert_item_index_to_positive_,
    irpass.dict_get_item_eliminator_,
    irpass.dict_get_item_const_eliminator_,
    irpass.dict_set_item_eliminator_,

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
    irpass.accumulaten_eliminater_,

    // a2
    irpass.merge_addn_,
    irpass.compare_switch_simplify_,
    irpass.addn_check_dump_,
    irpass.depend_value_elim_,

    // a3
    irpass.same_eliminate_,
    irpass.row_tensor_add_zeros_like_,
    irpass.split_environ_get_set_with_tuple_value_,

    // other
    irpass.value_based_eliminate_,
    irpass.print_const_string_wrapper_,
    irpass.stack_unstack_eliminate_,
  });
}

OptPassGroupMap GetJitOptPassesA(const opt::irpass::OptimizeIRPassLib &irpass, const ResourcePtr &resource) {
  OptPassGroupMap map_a(
    {{kSwitchSimplifyFlag, opt::OptPassConfig({irpass.switch_simplify_})},
     {"loop_unroll", opt::OptPassConfig({irpass.loop_unroll_before_grad_})},
     {"a_1", GetJitOptPassA1(irpass)},
     {"with_stream_mark", opt::OptPassConfig(opt::OptPassConfig(opt::irpass::WithStreamMark))},
     {"recompute_prepare", opt::OptPassConfig({irpass.set_cell_output_no_recompute_})},
     {"updatestate_depend_eliminate", opt::OptPassConfig(opt::irpass::UpdatestateDependEliminater())},
     {"updatestate_assign_eliminate", opt::OptPassConfig(opt::irpass::UpdatestateAssignEliminater())},
     {"updatestate_loads_eliminate", opt::OptPassConfig(opt::irpass::UpdatestateLoadsEliminater())},
     {"parameter_eliminate", opt::OptPassConfig(opt::irpass::ParameterEliminator())},
     {"specialize_transform", opt::OptPassConfig({irpass.specialize_transform_})},
     {"updatestate_useless_node_eliminater", opt::OptPassConfig({irpass.updatestate_useless_node_eliminater_})},
     {"accelerated_algorithm", opt::OptPassConfig({irpass.less_batch_normalization_})},
     {"meta_shard_fg_expand", opt::OptPassConfig(opt::irpass::ExpandMetaShardFg())},
     {"get_grad_eliminate_", opt::OptPassConfig({irpass.get_grad_eliminate_})},
     {"merge_forward", opt::OptPassConfig(ad::MergeForward)},
     {"cell_reuse_recompute_pass", opt::OptPassConfig(opt::irpass::Recomputation())},
     {"cell_reuse_handle_not_recompute_node_pass",
      opt::OptPassConfig({irpass.remove_not_recompute_node_}, false, true)},
     {"j_node_and_user_rematch", opt::OptPassConfig({irpass.j_node_and_user_rematch_})},
     {kMetaFgExpandFlag, opt::OptPassConfig(opt::irpass::ExpandMetaFg())},
     {"replace_old_param", opt::OptPassConfig({irpass.replace_old_param_})},
     {"inline_without_move", opt::OptPassConfig({irpass.inline_without_move_})},
     {"renormalize", opt::OptPassConfig::Renormalize()},
     {"add_forward_monad_depend", opt::OptPassConfig(opt::irpass::AddForwardMonadDepend)},
     {"auto_monad_grad", opt::OptPassConfig(ReAutoMonadWrapper)},
     {"auto_monad_eliminator", opt::OptPassConfig(opt::AutoMonadEliminator())},
     {"cse", opt::OptPassConfig(opt::CSEPass(false))},
     {"replace_applicator", opt::OptPassConfig({irpass.replace_applicator_})}});
  AddMetaMorphosis(resource, kMetaFgExpandFlag, kSwitchSimplifyFlag, &map_a);
  return map_a;
}

OptPassGroupMap GetA1A2(const opt::irpass::OptimizeIRPassLib &irpass, const ResourcePtr &resource) {
  auto opt_a = GetOptPassesA(irpass, resource);
  auto iter = std::find_if(opt_a.begin(), opt_a.end(), [](const auto &item) { return item.name == "a_2"; });
  auto a1_a2_len = std::distance(opt_a.begin(), iter) + 1;
  OptPassGroupMap a1_a2(opt_a.begin(), opt_a.begin() + a1_a2_len);
  opt::irpass::OptimizeIRPassLib irpass_inline;
  opt::OptPassConfig inline_pass = opt::OptPassConfig({irpass_inline.inline_});
  (void)a1_a2.emplace_back("parallel_inline_pass", inline_pass);
  return a1_a2;
}

OptPassGroupMap GetAddAttr(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassGroupMap addattr_pass_group;
  (void)addattr_pass_group.emplace_back("tag_attr", opt::OptPassConfig(parallel::HandleAddAttr));
  (void)addattr_pass_group.emplace_back("meta_addattr_fg_expand",
                                        opt::OptPassConfig(opt::irpass::ExpandMetaAddAttrFg()));
  if (parallel::ParallelContext::GetInstance()->parallel_mode() == "semi_auto_parallel" ||
      parallel::ParallelContext::GetInstance()->parallel_mode() == "auto_parallel") {
    (void)addattr_pass_group.emplace_back("addattr_inline", opt::OptPassConfig({irpass.inline_}));
  }
  return addattr_pass_group;
}

OptPassGroupMap GetOptPassesAfterCconv(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig c_1 = opt::OptPassConfig({
    // Safe inlining,
    irpass.inline_,
    irpass.updatestate_useless_node_eliminater_,
    irpass.updatestate_pure_node_eliminater_,
    irpass.load_eliminater_,
    irpass.switch_call_monad_eliminater_,
    irpass.redundant_stopgrad_eliminater_,
    irpass.partial_eliminate_,
    irpass.slice_to_tuple_,
  });
  opt::OptPassConfig updatestate_depend_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateDependEliminater());
  opt::OptPassConfig updatestate_assign_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateAssignEliminater());
  opt::OptPassConfig updatestate_loads_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateLoadsEliminater());

  OptPassGroupMap map_a({{"c_1", c_1},
                         {"parameter_eliminate", opt::OptPassConfig(opt::irpass::ParameterEliminator())},
                         {"updatestate_depend_eliminate", updatestate_depend_eliminate},
                         {"updatestate_assign_eliminate", updatestate_assign_eliminate},
                         {"updatestate_loads_eliminate", updatestate_loads_eliminate},
                         {"cse", opt::OptPassConfig(opt::CSEPass(false))},
                         {"renormalize", opt::OptPassConfig::Renormalize()}});

  return map_a;
}

OptPassGroupMap GetJitOptPassesAfterCconv(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig c_1 = opt::OptPassConfig({
    // Safe inlining,
    irpass.inline_,
    irpass.updatestate_useless_node_eliminater_,
    irpass.updatestate_pure_node_eliminater_,
    irpass.load_eliminater_,
    irpass.switch_call_monad_eliminater_,
    irpass.partial_eliminate_,
    irpass.slice_to_tuple_,
  });
  opt::OptPassConfig updatestate_depend_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateDependEliminater());
  opt::OptPassConfig updatestate_assign_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateAssignEliminater());
  opt::OptPassConfig updatestate_loads_eliminate = opt::OptPassConfig(opt::irpass::UpdatestateLoadsEliminater());
  opt::OptPassConfig switch_simplify = opt::OptPassConfig({
    irpass.switch_simplify_,
  });

  OptPassGroupMap map_a(
    {{"c_1", c_1},
     {"parameter_eliminate", opt::OptPassConfig(opt::irpass::ParameterEliminator())},
     {"updatestate_depend_eliminate", updatestate_depend_eliminate},
     {"updatestate_assign_eliminate", updatestate_assign_eliminate},
     {"updatestate_loads_eliminate", updatestate_loads_eliminate},
     {"cse", opt::OptPassConfig(opt::CSEPass(false))},
     {"call_graph_tuple_transform", opt::OptPassConfig({irpass.call_graph_tuple_transform_})},
     {"tuple_list_get_item_eliminator", opt::OptPassConfig({irpass.tuple_list_get_item_eliminator_})},
     {"none_parameter_eliminate", opt::OptPassConfig(opt::irpass::NoneParameterEliminator())},
     {"renormalize", opt::OptPassConfig::Renormalize()},
     {kSwitchSimplifyFlag, switch_simplify}});

  return map_a;
}

OptPassGroupMap GetOptPassesTransformGraph(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig d_1 = opt::OptPassConfig({
    irpass.call_graph_tuple_transform_,
    irpass.list_to_tuple_eliminator_,
    irpass.tuple_to_list_eliminator_,
    irpass.tuple_list_get_item_eliminator_,
    irpass.tuple_list_get_item_const_eliminator_,
    irpass.tuple_list_set_item_eliminator_,
    irpass.tuple_list_get_set_item_eliminator_,
    irpass.tuple_list_get_item_depend_reorder_,
    irpass.tuple_list_convert_item_index_to_positive_,
  });
  opt::OptPassConfig switch_simplify = opt::OptPassConfig({
    irpass.switch_simplify_,
  });

  OptPassGroupMap map_a({{"d_1", d_1},
                         {"none_parameter_eliminate", opt::OptPassConfig(opt::irpass::NoneParameterEliminator())},
                         {"renormalize", opt::OptPassConfig::Renormalize()},
                         {kSwitchSimplifyFlag, switch_simplify}});

  return map_a;
}

OptPassGroupMap GetOptPassesB(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig b_1 = opt::OptPassConfig({irpass.zero_like_fill_zero_,
                                               irpass.list_to_tuple_eliminator_,
                                               irpass.tuple_to_list_eliminator_,
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
                                               irpass.redundant_stopgrad_eliminater_,
                                               irpass.special_op_eliminate_,
                                               irpass.dump_gradient_eliminate_,
                                               irpass.environ_get_eliminate_,
                                               irpass.environ_get_add_eliminate_,
                                               irpass.environ_get_set_eliminate_,
                                               irpass.environ_get_depend_swap_,
                                               irpass.environ_add_const_eliminate_,
                                               irpass.value_based_eliminate_,
                                               irpass.parallel_virtual_node_,
                                               irpass.const_output_eliminate_},
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

OptPassGroupMap GetOptPynativeGradEpiloguePhases(const opt::irpass::OptimizeIRPassLib &irpass,
                                                 const ResourcePtr &resource) {
  auto opt_a = GetOptPassesA(irpass, resource);
  auto a3 = opt_a[opt_a.size() - 1];
  OptPassGroupMap map({
    {"renormalize", opt::OptPassConfig::Renormalize()},
    {"cse", opt::OptPassConfig(opt::CSEPass(false))},
    {a3},
  });
  return map;
}

OptPassGroupMap GetGradPartialTransformPhases() {
  opt::irpass::GradPartialPassLib irpass;
  auto grad_partial_transform = opt::OptPassConfig({irpass.grad_partial_transform_});
  opt::OptPassGroupMap grad_partial_transform_map({{"grad_partial_transform", grad_partial_transform}});
  return grad_partial_transform_map;
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

OptPassGroupMap GetSymbolEngineOptPass(const opt::irpass::OptimizeIRPassLib &irpass) {
  if (common::GetEnv("MS_SYMBOL_ENGINE_OPTIMIZE") == "off") {
    MS_LOG(INFO) << "SymbolEngineOptimizer is disabled.";
    return OptPassGroupMap();
  }
  OptPassGroupMap map({{"build", opt::OptPassConfig(opt::irpass::SymbolEngineBuilder())},
                       {"elim_shapecalc", opt::OptPassConfig({irpass.elim_shapecalc_of_broadcastargs_})},
                       {"elim_not_effective", opt::OptPassConfig({irpass.elim_not_effective_node_})},
                       {"opt_reshape", opt::OptPassConfig({irpass.opt_reshape_})},
                       {"fold_const_symbol", opt::OptPassConfig({irpass.fold_const_symbol_})},
                       {"renormalize", opt::OptPassConfig::Renormalize()}});
  return map;
}

OptPassGroupMap GetJitOptPassesB(const opt::irpass::OptimizeIRPassLib &irpass) {
  std::vector<opt::SubstitutionPtr> frontend_op_eliminate_pass_list = {
    irpass.zero_like_fill_zero_, irpass.check_bprop_eliminate_, irpass.row_tensor_eliminate_};
  if (!pynative::GradState::Get().RequiresGrad()) {
    (void)frontend_op_eliminate_pass_list.emplace_back(irpass.special_op_eliminate_);
  }
  opt::OptPassConfig frontend_op_eliminate = opt::OptPassConfig(frontend_op_eliminate_pass_list);
  std::vector<opt::SubstitutionPtr> inline_after_opt_a_pass_list = {irpass.tuple_list_get_item_eliminator_};
  if (!pynative::GradState::Get().RequiresGrad()) {
    (void)inline_after_opt_a_pass_list.emplace_back(irpass.reset_defer_inline_);
    (void)inline_after_opt_a_pass_list.emplace_back(irpass.inline_);
  }
  opt::OptPassConfig inline_after_opt_a = opt::OptPassConfig(inline_after_opt_a_pass_list, false, true);

  OptPassGroupMap opt_map(
    {{"frontend_op_eliminate", frontend_op_eliminate}, {"inline_after_opt_a", inline_after_opt_a}});
  return opt_map;
}

OptPassGroupMap GetViewInplaceProcessMap(opt::irpass::ViewInplacePassType type) {
  opt::irpass::OptimizeIRPassLib irpass;
  if (type == opt::irpass::ViewInplacePassType::CommonInline) {
    OptPassGroupMap inline_map({{"inline", opt::OptPassConfig({irpass.inline_})}});
    return inline_map;
  } else if (type == opt::irpass::ViewInplacePassType::VirtualOpsInsert) {
    OptPassGroupMap view_inplace_map(
      {{"virtual_view_grad_insert", opt::OptPassConfig(opt::irpass::VirtualViewGradInsert)}});
    return view_inplace_map;
  } else if (type == opt::irpass::ViewInplacePassType::DoInplaceAndVirtualOpsRemove) {
    OptPassGroupMap view_inplace_map(
      {{"virtual_view_insert", opt::OptPassConfig(opt::irpass::VirtualViewInsert)},
       {"do_inplace_input_replace", opt::OptPassConfig(opt::irpass::DoInplaceInputReplace)},
       {"isolate_inplace_func_replace", opt::OptPassConfig(opt::irpass::IsolateInplaceFuncReplace)},
       {"remove_redundant_virtual_ops", opt::OptPassConfig(opt::irpass::RemoveRedundantVirtualOps)},
       {"updatestate_depend_eliminate", opt::OptPassConfig(opt::irpass::UpdatestateDependEliminater())}});
    return view_inplace_map;
  } else if (type == opt::irpass::ViewInplacePassType::OnlyDoInplace) {
    OptPassGroupMap inplace_map(
      {{"isolate_inplace_func_replace", opt::OptPassConfig(opt::irpass::IsolateInplaceFuncReplace)},
       {"do_inplace_input_replace", opt::OptPassConfig(opt::irpass::DoInplaceInputReplace)}});
    return inplace_map;
  } else if (type == opt::irpass::ViewInplacePassType::EliminateVirtualView) {
    OptPassGroupMap eliminate_virtual_view_map(
      {{"eliminate_virtual_view", opt::OptPassConfig({irpass.virtual_view_op_eliminate_})}});
    return eliminate_virtual_view_map;
  }
  MS_LOG(EXCEPTION) << "Unsupported view inplace process type";
}

static mindspore::HashMap<std::string, std::shared_ptr<Optimizer>> g_pass_opts = {};

void InitOpt(const ResourcePtr &resource) {
  if (g_pass_opts.size() == 0) {
    opt::irpass::OptimizeIRPassLib irpass;
    g_pass_opts["a1a2"] = Optimizer::MakeOptimizer("a1a2", resource, GetA1A2(irpass, resource));
    g_pass_opts["opt_a"] = Optimizer::MakeOptimizer("opt_a", resource, GetOptPassesA(irpass, resource));
    g_pass_opts["opt_b"] = Optimizer::MakeOptimizer("opt_b", resource, GetOptPassesB(irpass), false, true);
    g_pass_opts["opt_after_cconv"] =
      Optimizer::MakeOptimizer("opt_after_cconv", resource, GetOptPassesAfterCconv(irpass), false, true);
    g_pass_opts["opt_trans_graph"] =
      Optimizer::MakeOptimizer("opt_trans_graph", resource, GetOptPassesTransformGraph(irpass), true, true);
    g_pass_opts["renormal"] = Optimizer::MakeOptimizer("renormal", resource, GetOptPassesC(irpass));
    g_pass_opts["opt_grad_epilogue"] = Optimizer::MakeOptimizer(
      "opt_grad_epilogue", resource, GetOptPynativeGradEpiloguePhases(irpass, resource), true, false);
    g_pass_opts["opt_prepare"] = Optimizer::MakeOptimizer("opt_prepare", resource, GetPreparePhases(irpass));
    g_pass_opts["opt_after_recompute"] =
      Optimizer::MakeOptimizer("opt_after_recompute", resource, GetAfterRecomputePass(irpass));
    g_pass_opts["symbol_engine_opt"] =
      Optimizer::MakeOptimizer("symbol_engine_opt", resource, GetSymbolEngineOptPass(irpass), true, true);
    g_pass_opts["jit_opt_a"] = Optimizer::MakeOptimizer("jit_opt_a", resource, GetJitOptPassesA(irpass, resource));
    g_pass_opts["jit_opt_b"] = Optimizer::MakeOptimizer("jit_opt_b", resource, GetJitOptPassesB(irpass));
    g_pass_opts["jit_opt_after_cconv"] =
      Optimizer::MakeOptimizer("jit_opt_after_cconv", resource, GetJitOptPassesAfterCconv(irpass), false, true);
    g_pass_opts["add_attr"] = Optimizer::MakeOptimizer("add_attr", resource, GetAddAttr(irpass));
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
    MS_LOG(ERROR) << "Opt passes error";
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
bool JitOptPassAGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "jit_opt_a"); }
bool OptPassBGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_b"); }
bool OptPassAfterCconvGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_after_cconv"); }
bool JitOptPassAfterCconvGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "jit_opt_after_cconv"); }
bool OptPassTransformGraphGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_trans_graph"); }
bool ControlGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_control"); }
bool PrepareGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_prepare"); }
bool OptAfterRecomputeGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_after_recompute"); }
bool JitOptPassBGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "jit_opt_b"); }

bool OptPassRNGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "renormal"); }
bool SymbolEngineOptGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "symbol_engine_opt"); }

bool OptPassGradEpilogueGroup(const ResourcePtr &resource) { return OptPassGroup(resource, "opt_grad_epilogue"); }
bool OptPassAddAttr(const ResourcePtr &resource) { return OptPassGroup(resource, "add_attr"); }

bool AddRecomputationPass(const ResourcePtr &resource) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (opt::RecomputeBeforeInline()) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(resource);
  opt::InsertRecomputedNodes(resource->func_graph());
  return true;
}

bool SliceRecomputeActivationPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  opt::SliceRecomputedActivationNodes(resource->func_graph());
  return true;
}

bool GroupedPairwiseExchangeAllToAllPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  opt::SetGroupedPairwiseExchangeAllToAll(resource);
  return true;
}

bool OffloadingPackedExpertFrontPass2(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  circle_handler::SetAttrToDepend(func_graph);
  bool res = parallel::SetOffloadingPackedExpert(func_graph);
  if (res) {
    abstract::AbstractBasePtrList args_abs;
    const auto parameters = func_graph->parameters();
    (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_abs),
                         [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
    FuncGraphPtr new_fg = pipeline::Renormalize(resource, func_graph, args_abs);
    resource->set_func_graph(new_fg);
    resource->set_args_abs(args_abs);
  }
  circle_handler::DetectAndRevertGraphCircle(func_graph, resource->manager(), "OffloadingPackedExpertFrontPass2");
  return true;
}

bool SliceReuseRecomputedActivationPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  parallel::SliceReuseRecomputedActivationNodes(resource->func_graph());
  return true;
}

bool LabelMicroInterleavedIndexPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    parallel::LabelMicroInterleavedIndex(resource->func_graph());
  }
  return true;
}

bool OverlapRecomputeAllGatherAndFlashAttentionGradPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    parallel::OverlapRecomputeAllGatherAndFlashAttentionGrad(resource->func_graph());
  }
  return true;
}

bool OverlapRecomputeCommPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    parallel::OverlapRecomputeComm(resource->func_graph());
  }
  return true;
}

bool OverlapGradRingAttentionPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    const auto &graph = resource->func_graph();
    circle_handler::SetAttrToDepend(graph);
    parallel::OverlapGradRingAttention(graph);
    circle_handler::DetectAndRevertGraphCircle(graph, resource->manager(), "OverlapGradRingAttentionPass");
  }
  return true;
}

bool OverlapGradFlashSP(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    const auto &graph = resource->func_graph();
    if (parallel::OverlapGradFlashSP(graph)) {
      FuncGraphPtr new_fg = LiftingClone(graph);
      resource->set_func_graph(new_fg);
    }
    circle_handler::DetectAndRevertGraphCircle(graph, resource->manager(), "OverlapGradFlashSP");
  }
  return true;
}

bool InterleaveSplitConcatBranches(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    parallel::InterleaveSplitConcatBranches(resource->func_graph());
  }
  return true;
}

bool InterleaveParallelBranches(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    parallel::InterleaveParallelBranches(resource->func_graph());
  }
  return true;
}

bool OptimizeParallelAllGatherCommPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    parallel::OptimizeParallelAllGatherComm(resource->func_graph());
  }
  return true;
}

bool OptimizeParamGatherPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    parallel::OverlapParamGather(resource->func_graph());
  }
  return true;
}

bool LabelFineGrainedInterleavedIndexPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    parallel::LabelFineGrainedInterleavedIndex(resource->func_graph());
  }
  return true;
}

bool AssignAddOpt(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  parallel::AssignAddOpt(func_graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto enable_concat_eliminate = ms_context->get_param<bool>(MS_CTX_ENABLE_CONCAT_ELIMINATE_OPT);
  if (!enable_concat_eliminate) {
    return true;
  }
  OptPassGroupMap map({{"renormalize", opt::OptPassConfig({opt::OptPassConfig::Renormalize()})}});
  auto renormalize = opt::Optimizer::MakeOptimizer("renormalize", resource, map);
  (void)renormalize->step(func_graph, false);
  return true;
}

bool PartialUnusedArgsEliminatePass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto opt = opt::irpass::PartialUnusedArgsEliminate();
  auto changed = opt(func_graph);
  if (changed) {
    OptPassGroupMap map({{"renormalize", opt::OptPassConfig({opt::OptPassConfig::Renormalize()})}});
    auto renormalize = opt::Optimizer::MakeOptimizer("renormalize", resource, map);
    (void)renormalize->step(func_graph, false);
  }
  return true;
}

bool MergeCastOpt(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  parallel::MergeCastOpt(resource->func_graph());
  return true;
}

bool ForceFp32Comm(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  parallel::Float32Redistribution(resource->func_graph());
  return true;
}

bool SwapDpAllReduceReduceScatterPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  parallel::SwapDpAllreduceReduceScatter(resource->func_graph());
  return true;
}

bool RemoveCastBeforeAssignAdd(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  parallel::RemoveCastBeforeAssignAdd(resource->func_graph());
  return true;
}

bool BiasAddCommSwap(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  parallel::BiasAddCommSwap(resource->func_graph());
  return true;
}

bool ReorderSendRecvBetweenFpBpPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    parallel::ReorderSendRecvBetweenFpBp(resource->func_graph());
  }
  return true;
}

bool MicroInterLeavedOrderControlPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    parallel::MicroInterleavedOrderControl(resource->func_graph());
  }
  return true;
}

bool OverlapGradCommPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  parallel::OverlapGradComm(resource->func_graph());
  return true;
}

bool FullMicroInterLeavedOrderControlPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    parallel::FullMicroInterleavedOrderControl(resource->func_graph());
  }
  return true;
}

bool SplitMatmulCommElementwiseOpFpPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  parallel::SplitMatmulCommElementwiseFp(resource->func_graph());
  return true;
}

bool SplitLayerNormCommFpPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  parallel::SplitLayerNormCommFp(resource->func_graph());
  return true;
}

bool CommOpAddAttrs(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  opt::CommOpAttrs(resource->func_graph());
  return true;
}

bool ProcessSendRecvForGE(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  opt::ProcessSendRecvForGE(resource->func_graph());
  return true;
}

bool AddCommOpReusePass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  opt::AddCommOpReuseTag(resource->func_graph());
  return true;
}

bool OverlapOptShardInPipelinePass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  parallel::OverlapOptShardInPipeline(resource->func_graph());
  return true;
}

bool BeginEndOverlapInlinePass(const ResourcePtr &resource) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = ms_context->get_param<bool>(MS_CTX_ENABLE_BEGIN_END_INLINE_OPT) && AnfAlgo::IsBackendGe();
  if (!is_enable) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  circle_handler::SetAttrToDepend(func_graph);
  parallel::BeginEndOverlapInlineOpt(resource->func_graph());
  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig get_item_eliminator_pass = opt::OptPassConfig({irpass.tuple_list_get_item_eliminator_});
  OptPassGroupMap map({{"get_item_eliminator", get_item_eliminator_pass}});
  auto get_item_eliminator = opt::Optimizer::MakeOptimizer("get_item_eliminator", resource, map);
  (void)get_item_eliminator->step(func_graph, false);
  circle_handler::DetectAndRevertGraphCircle(func_graph, resource->manager(), "BeginEndOverlapInlinePass",
                                             "enable_begin_end_inline_opt");
  return true;
}

bool OverlapGradMatmulAndGradAllreduce(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    parallel::OverlapGradMatmulAndGradAllreduce(resource->func_graph());
  }
  return true;
}

bool OverlapOptShardGradInPipelinePass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  parallel::OverlapOptShardGradInPipeline(resource->func_graph());
  return true;
}

bool HandleGroupInfoPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  parallel::HandleGroupInfo();
  return true;
}

bool LoopUnrollPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig loop_unroll_pass = opt::OptPassConfig({irpass.loop_unroll_after_grad_});
  OptPassGroupMap map({{"loop_unroll", loop_unroll_pass}});
  auto loop_unroll_ = opt::Optimizer::MakeOptimizer("loop_unroll_optimizer", resource, map);
  (void)loop_unroll_->step(func_graph, false);
  return true;
}

bool OverlapRecomputeAndGradModelParallel(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsFrontendPassEnabledForGPTO()) {
    const auto &graph = resource->func_graph();
    circle_handler::SetAttrToDepend(graph);
    parallel::OverlapRecomputeAndGradModelParallel(graph);
    circle_handler::DetectAndRevertGraphCircle(graph, resource->manager(), "OverlapRecomputeAndGradModelParallel");
  }
  return true;
}

bool RemoveValueNodeDuplicationsPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->func_graph() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Remove value node duplications error.";
  }
  auto manager = resource->manager();
  HashCache hash_cache;
  HashValue hashes;
  // Remove duplicated value nodes across all graphs in manager
  const auto &node_user_map = manager->node_users();
  for (auto &fg : manager->func_graphs()) {
    auto value_nodes = fg->value_nodes();
    for (const auto &value_pair : value_nodes) {
      auto &users = node_user_map.at(value_pair.first);
      auto prim = GetValueNode<PrimitivePtr>(value_pair.first);
      if (IsPrimitiveEquals(prim, prim::kPrimUpdateState)) {
        continue;
      }
      // If valuenode is used by inplace_prim.
      bool used_by_inplace_prim = std::any_of(users.begin(), users.end(), [](const auto &user) {
        auto cnode = dyn_cast<CNode>(user.first);
        if (cnode == nullptr) {
          return false;
        }
        auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
        return (prim != nullptr) && prim->inplace_prim();
      });
      if (used_by_inplace_prim) {
        continue;
      }
      // For data parallel with some parameters redundant, the allreduce will share the same value node
      // which will raise an error when do allreduce fusion, so the solution is to make the allreduce's value node
      // not be removed, if we found the fusion tag.
      if (users.size() == 1) {
        auto cnode = users.front().first->cast<CNodePtr>();
        if (IsPrimitiveCNode(cnode, prim::kPrimAllReduce) && cnode->size() > 1 && cnode->input(1)->isa<ValueNode>()) {
          const auto &allreduce_prim = GetCNodePrimitive(cnode);
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

bool RemoveValueNodeDuplicationsPassForJit(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->func_graph() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Remove value node duplications error.";
  }
  auto manager = resource->manager();
  HashCache hash_cache;
  HashValue hashes;
  // Remove duplicated value nodes across all graphs in manager
  const auto &node_user_map = manager->node_users();
  for (auto &fg : manager->func_graphs()) {
    auto value_nodes = fg->value_nodes();
    for (const auto &value_pair : value_nodes) {
      auto prim = GetValueNode<PrimitivePtr>(value_pair.first);
      if (IsPrimitiveEquals(prim, prim::kPrimUpdateState)) {
        continue;
      }
      // If valuenode is used by inplace_prim.
      auto &users = node_user_map.at(value_pair.first);
      bool used_by_inplace_prim = std::any_of(users.begin(), users.end(), [](const auto &user) {
        auto cnode = dyn_cast<CNode>(user.first);
        if (cnode == nullptr) {
          return false;
        }
        auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
        return (prim != nullptr) && prim->inplace_prim();
      });
      if (used_by_inplace_prim) {
        continue;
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

bool ExpandDumpFlagPass(const ResourcePtr &resource) {
  auto expand_dump_flag_opt = opt::irpass::ExpandDumpFlag();
  expand_dump_flag_opt(resource);
  return true;
}

bool ControlDataBroadcastOrderPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto graph = resource->func_graph();
  circle_handler::SetAttrToDepend(graph);
  parallel::FreezeParallelOptimizerCommOrder(graph);
  parallel::ReplaceGetnextWithBroadcast(graph);
  parallel::ControlOptShardCommAndDataBroadcastOrder(graph);
  parallel::ControlPipelineCommAndDataBroadcastOrder(graph);
  circle_handler::DetectAndRevertGraphCircle(graph, resource->manager(), "ControlDataBroadcastOrderPass");
  return true;
}

bool DatasetRepeatReaderOptPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto root = resource->func_graph();
  auto manager = resource->manager();
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  if (parallel_mode != parallel::kSemiAutoParallel && parallel_mode != parallel::kAutoParallel) {
    MS_LOG(INFO) << "Only auto_parallel and semi_auto_parallel support dataset repeat optimizer.";
    return true;
  }
  auto dataset_opt = std::make_shared<parallel::DatasetReaderOptimizer>(manager, root);
  if (!dataset_opt->Init()) {
    return true;
  }
  dataset_opt->BroadcastDataset();
  return true;
}

bool PipelineSplitPass(const ResourcePtr &resource) { return PipelineSplit(resource); }

bool ParallelVirtualDatasetPass(const ResourcePtr &resource) { return ParallelVirtualDataset(resource); }

bool DetachBackward(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  if (parallel_mode != parallel::kSemiAutoParallel && parallel_mode != parallel::kAutoParallel) {
    MS_LOG(INFO) << "Only auto_parallel and semi_auto_parallel support detach backward graph.";
    return true;
  }
  auto pp_scheduler = parallel_context->pipeline_scheduler();
  auto stage_num = parallel_context->pipeline_stage_split_num();
  if (pp_scheduler == parallel::ZBV && stage_num > 1) {
    auto manager = resource->manager();
    auto root = resource->func_graph();
    auto stage = parallel::InferStage();
    std::shared_ptr<parallel::DetachBackward> processor =
      std::make_shared<parallel::DetachBackward>(manager, root, stage);
    processor->Init();
    processor->Run();
    abstract::AbstractBasePtrList args_abs;
    auto parameters = root->parameters();
    (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_abs),
                         [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
    FuncGraphPtr new_fg = pipeline::Renormalize(resource, root, args_abs);
    resource->set_func_graph(new_fg);
    resource->set_args_abs(args_abs);
  }
  return true;
}

void ResetPipelineConfig() {
  // Temporary solution: PipelineInterleaved does not support predict. When refactoring predict, it should be removed.
  // Reset the user-set pp configuration, which is saved by the pipeline_spilit.cc SavePipelineConfigOrigin.
  const auto parallel_context = parallel::ParallelContext::GetInstance();
  const auto is_pp_interleave_temp = parallel_context->pipeline_interleave_temp();
  const auto pipeline_scheduler_temp = parallel_context->pipeline_scheduler_temp();
  parallel_context->set_pipeline_interleave(is_pp_interleave_temp);
  parallel_context->set_pipeline_scheduler(pipeline_scheduler_temp);
}

bool PipelineParallelScheduler(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto root = resource->func_graph();
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  if (parallel_mode != parallel::kSemiAutoParallel && parallel_mode != parallel::kAutoParallel) {
    MS_LOG(INFO) << "Only auto_parallel and semi_auto_parallel support pipeline split.";
    return true;
  }
  auto is_pp_interleave = parallel_context->pipeline_interleave();
  auto stage_num = parallel_context->pipeline_stage_split_num();
  if (is_pp_interleave && stage_num > 1) {
    auto manager = resource->manager();
    auto stage = parallel::InferStage();
    auto pp_scheduler = parallel_context->pipeline_scheduler();
    std::shared_ptr<parallel::PipelineScheduler> scheduler =
      parallel::SchedulerCreator::Instance().Create(pp_scheduler, manager, root, stage, stage_num);
    if (!scheduler) {
      MS_LOG(EXCEPTION) << "Unsupported pipeline parallel scheduler: " << pp_scheduler;
    }
    scheduler->GetBorderNode();
    scheduler->Reorder();
    ResetPipelineConfig();
  }
  opt::ProcessSendRecvForGE(root);
  return true;
}

bool AutoParallelPass(const ResourcePtr &resource) {
  auto func_graph = resource->func_graph();
  auto opt = opt::Optimizer::MakeEmptyOptimizer(resource);
  return parallel::StepAutoParallel(func_graph, opt);
}

bool SetTrainingFlagPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto root = resource->func_graph();
  MS_EXCEPTION_IF_NULL(root);
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &graphs = manager->func_graphs();
  bool is_training =
    std::any_of(graphs.cbegin(), graphs.cend(), [](auto cur_graph) -> bool { return cur_graph->has_flag(kTraining); });
  if (is_training) {
    root->set_flag(kTraining, true);
  }

  return true;
}

bool AutoParallelSymbolPassWithReNormalize(const ResourcePtr &resource) {
  // 1, auto parallel; 2, dynamic shape
  auto func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!parallel::IsParallelDynamicShape(func_graph)) {
    return true;
  }
  MS_LOG(INFO) << "symbol pass for parallel begin";
  // must be bind with renormalize
  opt::irpass::OptimizeIRPassLib irpass;
  OptPassGroupMap opt_map({{"renormalize", opt::OptPassConfig::Renormalize()},
                           {"build", opt::OptPassConfig(opt::irpass::SymbolEngineBuilder())},
                           {"fold_same_value", opt::OptPassConfig({irpass.fold_same_value_})}});
  auto opt = opt::Optimizer::MakeOptimizer("parallel-infer-symbol", resource, opt_map, true);
  (void)opt->step(func_graph, false);
  MS_LOG(INFO) << "symbol pass for parallel end";
  return true;
}

bool EliminateUnusedParamsPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_EXCEPTION_IF_NULL(resource->func_graph());
  FuncGraphPtr func_graph = resource->func_graph();
  ud_chain::Preprocess(func_graph);
  AnfNodePtrList parameters;
  size_t eliminate_cnt = 0;
  for (const auto &param : func_graph->parameters()) {
    if (!ud_chain::GetUsers(param).empty() || param->cast<ParameterPtr>()->has_default()) {
      parameters.push_back(param);

      // update kActualArgumentIndex
      auto abstract = param->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      if (abstract->has_user_data(pipeline::kActualArgumentIndex)) {
        std::shared_ptr<size_t> index_ptr = abstract->user_data<size_t>(pipeline::kActualArgumentIndex);
        MS_EXCEPTION_IF_NULL(index_ptr);
        auto new_index = *index_ptr - eliminate_cnt;
        MS_EXCEPTION_IF_CHECK_FAIL(*index_ptr >= eliminate_cnt, "argument index < eliminate cnt");
        abstract->set_user_data<size_t>(kActualArgumentIndex, std::make_shared<size_t>(new_index));
        MS_LOG(DEBUG) << "Param:" << param->DebugString() << ", original index:" << new_index + eliminate_cnt
                      << ", eliminate_cnt:" << eliminate_cnt << ", new index:" << new_index;
      } else {
        MS_LOG(INFO) << "Cannot find index of param: " << param->DebugString() << ", " << abstract->ToString();
      }
    } else {
      ++eliminate_cnt;
      MS_LOG(DEBUG) << "Eliminate param:" << param->DebugString();
    }
  }
  func_graph->set_parameters(parameters);
  return true;
}

bool ValidatePass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_EXCEPTION_IF_NULL(resource->func_graph());
  FuncGraphPtr func_graph = resource->func_graph();
  auto jit_config = PhaseManager::GetInstance().jit_config();
  resource->func_graph()->set_user_data<std::map<std::string, std::string>>(
    "jit_config", std::make_shared<std::map<std::string, std::string>>(jit_config));
  Validate(func_graph);
  return true;
}

bool GradPartialTransformPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto grad_partial_transform_map = GetGradPartialTransformPhases();
  auto grad_partial_transform =
    opt::Optimizer::MakeOptimizer("grad_partial_transform", resource, grad_partial_transform_map);
  (void)grad_partial_transform->step(func_graph, false);
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

bool OptAfterJitGradPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig ad_related_special_op_eliminate =
    opt::OptPassConfig({irpass.ad_related_special_op_eliminate_, irpass.special_op_eliminate_,
                        irpass.dump_gradient_eliminate_, irpass.virtual_view_grad_op_eliminate_});

  OptPassGroupMap map(
    {{"ad_related_special_op_eliminate", ad_related_special_op_eliminate},
     {"updatestate_depend_eliminate", opt::OptPassConfig(opt::irpass::UpdatestateDependEliminater())}});
  if (pynative::GradState::Get().RequiresGrad()) {
    opt::OptPassConfig inline_after_jit_grad =
      opt::OptPassConfig({irpass.reset_defer_inline_, irpass.inline_}, false, true);
    (void)map.emplace_back("inline_after_jit_grad", inline_after_jit_grad);
  }

  auto opt_after_jit_grad = opt::Optimizer::MakeOptimizer("opt_after_jit_grad", resource, map);
  (void)opt_after_jit_grad->step(func_graph, false);
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

bool MutableEliminatePass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  opt::irpass::OptimizeIRPassLib irpass;

  OptPassGroupMap map({{"mutable_op_eliminate", opt::OptPassConfig({irpass.mutable_op_eliminate_})}});

  auto opt_mutable_eliminate = opt::Optimizer::MakeOptimizer("mutable_eliminate", resource, map);
  (void)opt_mutable_eliminate->step(func_graph, false);
  return true;
}

bool EnvironConversionPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  (void)opt::EnvironConversion(resource);
  return true;
}

bool BackendPass(const ResourcePtr &resource) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->CellReuseLevel() != CellReuseLevel::kLazyInline) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(resource);
  auto func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  opt::irpass::AdjustGraphAfterValidatePassLib irpass;
  opt::OptPassConfig make_tuple_from_fprop_eliminate = opt::OptPassConfig({
    irpass.make_tuple_from_fprop_eliminate_,
  });
  OptPassGroupMap map({
    {"make_tuple_from_fprop_eliminate", make_tuple_from_fprop_eliminate},
    {"renormalize", opt::OptPassConfig::Renormalize()},
  });
  auto backend_pass = opt::Optimizer::MakeOptimizer("backend_pass", resource, map, false, true);
  (void)backend_pass->step(func_graph, false);
  (void)EnvironConversionPass(resource);
  return true;
}

void ViewInplaceBeforeGradProcessPass(const ResourceBasePtr &resource, const FuncGraphPtr &func_graph,
                                      opt::irpass::ViewInplacePassType type) {
  MS_EXCEPTION_IF_NULL(func_graph);

  OptPassGroupMap map = GetViewInplaceProcessMap(type);
  auto graph_opt = opt::Optimizer::MakeOptimizer("view_inplace", resource, map, true, false, false);
  (void)graph_opt->step(func_graph, false);
}

bool IsFrontendPassEnabledForGPTO() {
  std::string gpto_options_str = common::GetEnv("MS_GPTO_OPTIONS");
  if (gpto_options_str.empty()) {
    return true;
  }
  nlohmann::json gpto_options_json = nlohmann::json::parse(gpto_options_str);
  auto it = gpto_options_json.find("passes");
  if (it == gpto_options_json.end()) {
    return true;
  }
  return (it.value() == "on");
}

REGISTER_PASS_FUNC_IMPL(CconvPass)
REGISTER_PASS_FUNC_IMPL(RemoveValueNodeDuplicationsPass)
REGISTER_PASS_FUNC_IMPL(AddRecomputationPass)

REGISTER_PASS_FUNC_IMPL(MutableEliminatePass)
REGISTER_PASS_FUNC_IMPL(EnvironConversionPass)
REGISTER_PASS_FUNC_IMPL(SliceRecomputeActivationPass)
REGISTER_PASS_FUNC_IMPL(MicroInterLeavedOrderControlPass)
REGISTER_PASS_FUNC_IMPL(CommOpAddAttrs)
REGISTER_PASS_FUNC_IMPL(AddCommOpReusePass)

std::vector<PassItem> kVmPasses = {
  {kPyInterpretToExecute, PyInterpretToExecutePass},
  {kRewriterBeforeOptA, RewriterBeforeOptAPass},
  {"opt_a", OptPassAGroup},
  {kPyInterpretToExecuteAfterOptA, PyInterpretToExecutePass},
  {"slice_cell_reuse_recomputed_activation", SliceReuseRecomputedActivationPass},
  {kRewriterAfterOptA, RewriterAfterOptAPass},
  {kConvertAfterRewriter, ConvertAfterRewriterPass},
  {kOrderPyExecuteAfterRewriter, OrderPyExecuteAfterRewriterPass},
  {kMutableEliminate, MutableEliminatePass},
  {"opt_b", OptPassBGroup},
  {"optimize_parallel_all_gather_comm", OptimizeParallelAllGatherCommPass},
  {"overlap_param_gather", OptimizeParamGatherPass},
  {kCconv, CconvPass},
  {kLoopUnroll, LoopUnrollPass},
  {"opt_after_cconv", OptPassAfterCconvGroup},
  {kRemoveDupValue, RemoveValueNodeDuplicationsPass},
  {kTupleTransform, OptPassTransformGraphGroup},
  {kPartialUnusedArgsEliminate, PartialUnusedArgsEliminatePass},
  {kAddRecomputation, AddRecomputationPass},
  {kCseAfterRecomputation, OptAfterRecomputeGroup},
  {kEnvironConv, EnvironConversionPass},
  {"swap_dp_allreduce_reducescatter", SwapDpAllReduceReduceScatterPass},
  {"bias_add_comm_swap", BiasAddCommSwap},
  {"label_micro_interleaved_index", LabelMicroInterleavedIndexPass},
  {"label_fine_grained_interleaved_index", LabelFineGrainedInterleavedIndexPass},
  {"merge_cast_opt", MergeCastOpt},
  {"slice_recompute_activation", SliceRecomputeActivationPass},
  {"micro_interleaved_order_control", MicroInterLeavedOrderControlPass},
  {"assign_add_opt", AssignAddOpt},
  {"ForceFp32Comm", ForceFp32Comm},
  {"remove_cast_before_assign_add", RemoveCastBeforeAssignAdd},
  {"full_micro_interleaved_order_control", FullMicroInterLeavedOrderControlPass},
  {"reorder_send_recv_between_fp_bp", ReorderSendRecvBetweenFpBpPass},
  {"comm_op_add_attrs", CommOpAddAttrs},
  {"add_comm_op_reuse_tag", AddCommOpReusePass},
  {"interleave_split_concat_branches", InterleaveSplitConcatBranches},
  {"interleave_parallel_branches", InterleaveParallelBranches},
  {"overlap_opt_shard_in_pipeline", OverlapOptShardInPipelinePass},
  {"overlap_opt_shard_grad_in_pipeline", OverlapOptShardGradInPipelinePass},
  {"control_data_broadcast_order", ControlDataBroadcastOrderPass},
  {"grouped_pairwise_exchange_alltoall", GroupedPairwiseExchangeAllToAllPass},
  {"offloading_packed_experts", OffloadingPackedExpertFrontPass2},
  {"overlap_recompute_and_grad_model_parallel", OverlapRecomputeAndGradModelParallel},
  {"overlap_grad_matmul_and_grad_allreduce", OverlapGradMatmulAndGradAllreduce},
  {"overlap_recompute_allgather_and_fa_grad", OverlapRecomputeAllGatherAndFlashAttentionGradPass},
  {"overlap_recompute_comm", OverlapRecomputeCommPass},
  {"overlap_grad_ring_attention", OverlapGradRingAttentionPass},
  {"overlap_grad_flash_sp", OverlapGradFlashSP},
  {"begin_end_overlap_inline", BeginEndOverlapInlinePass},
  {"split_matmul_comm_elemetwise", SplitMatmulCommElementwiseOpFpPass},
  {"split_layernorm_comm", SplitLayerNormCommFpPass},
  // The pass cache hccl group, so the hccl group should be created before the pass
  {"handle_group_info", HandleGroupInfoPass},
  {"symbol_engine_optimizer", SymbolEngineOptGroup}};

std::vector<PassItem> kPynativePasses = {{"opt_a", OptPassAGroup},
                                         {"opt_b", OptPassBGroup},
                                         {kCconv, CconvPass},
                                         {"transform_top", TransformTopGraphPass},
                                         {"transform_graph", OptPassTransformGraphGroup}};

std::vector<PassItem> kInlinePasses = {{kRewriterBeforeOptA, RewriterBeforeOptAPass}, {"a1a2", OptPassA1A2}};
std::vector<PassItem> kAddAttrWithInlinePass = {{kAddAttrWithInline, OptPassAddAttr}};
}  // namespace pipeline
}  // namespace mindspore
