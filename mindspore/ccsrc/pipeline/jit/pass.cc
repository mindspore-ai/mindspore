/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/cse.h"
#include "frontend/optimizer/graph_kernel_reuse.h"
#include "frontend/optimizer/clean.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/control_depend.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "frontend/parallel/allreduce_fusion/step_allreduce_fusion.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace pipeline {
using OptPassGroupMap = opt::OptPassGroupMap;
using Optimizer = opt::Optimizer;
using CompileGraphs = compile::CompileGraphs;
using abstract::AnalysisResult;
using mindspore::abstract::AnalysisContextPtr;
using mindspore::validator::Validate;

bool SimplifyDataStructuresPass(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res->func_graph());

  FuncGraphPtr func_graph = res->func_graph();
  bool changed = opt::SimplifyDataStructures(func_graph, res->manager());

  abstract::AbstractBasePtrList args_spec;
  auto parameters = func_graph->parameters();
  (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_spec),
                       [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
  if (changed) {
    FuncGraphPtr new_fg = Renormalize(res, func_graph, args_spec);
    res->set_func_graph(new_fg);
  }
  res->set_args_spec(args_spec);
  return true;
}

bool CleanAfterOptAPass(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res->func_graph());

  FuncGraphPtr func_graph = res->func_graph();
  bool changed = opt::CleanAfterOptA(func_graph, res->manager());

  abstract::AbstractBasePtrList args_spec;
  auto parameters = func_graph->parameters();
  (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_spec),
                       [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
  if (changed) {
    FuncGraphPtr new_fg = Renormalize(res, func_graph, args_spec);
    res->set_func_graph(new_fg);
  }
  res->set_args_spec(args_spec);
  return true;
}

namespace {
OptPassGroupMap GetOptPassesA(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig a_1 = opt::OptPassConfig({
    irpass.switch_layer_defer_inline_,
    irpass.switch_simplify_,

    // Safe inlining
    irpass.inline_,
    irpass.partial_eliminate_,
    irpass.replace_applicator_,

    // Specialization
    irpass.specialize_transform_,

    // Miscellaneous
    irpass.item_tuple_eliminate_,
    irpass.env_get_item_eliminate_,
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

    // Safe inlining
    irpass.inline_,
    irpass.sparse_tensor_eliminate_,
  });
  opt::OptPassConfig a_2 = opt::OptPassConfig({
    irpass.merge_addn_,
    irpass.float_tuple_getitem_switch_,
    irpass.float_env_getitem_switch_,
    irpass.incorporate_getitem_set_,
    irpass.incorporate_call_,
    irpass.incorporate_call_switch_,
    irpass.incorporate_env_getitem_bypass_recursive_,
    irpass.incorporate_env_getitem_switch_,
    irpass.new_env_get_item_,
    irpass.depend_value_elim_,
    irpass.all_reduce_const_elim_,
  });
  opt::OptPassConfig a_after_grad = opt::OptPassConfig({
    irpass.inline_without_move_,
  });
  opt::OptPassConfig a_3 = opt::OptPassConfig({
    irpass.arithmetic_simplify2_,
    irpass.same_eliminate_,
    irpass.check_bprop_eliminate_,
    irpass.replace_applicator_,
  });
  opt::OptPassConfig virtual_dataset = opt::OptPassConfig({irpass.virtual_dataset_eliminate_});
  opt::OptPassConfig grad = opt::OptPassConfig({irpass.expand_jprim_}, true);
  opt::irpass::ResolveIRPassLib resolve_irpass;

  opt::OptPassConfig resolve_pass =
    opt::OptPassConfig({resolve_irpass.resolver_resolve_, resolve_irpass.resolver_getattr_,
                        irpass.get_make_ref_eliminate_, irpass.replace_old_param_});

  OptPassGroupMap map_a({{"a_1", a_1},
                         {"a_2", a_2},
                         {"auto_parallel", opt::OptPassConfig(parallel::StepAutoParallel)},
                         {"parallel", opt::OptPassConfig(parallel::StepParallel)},
                         {"allreduce_fusion", opt::OptPassConfig(parallel::StepAllreduceFusion)},
                         {"virtual_dataset", virtual_dataset},
                         {"grad", grad},
                         {"resolve", resolve_pass},
                         {"a_after_grad", a_after_grad},
                         {"renormalize", opt::OptPassConfig::Renormalize()},
                         {"cse", opt::OptPassConfig(opt::CSE(false))},
                         {"a_3", a_3}});

  return map_a;
}

OptPassGroupMap GetOptPassesAfterCconv(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig c_1 = opt::OptPassConfig({
    // Safe inlining
    irpass.inline_,
    irpass.partial_eliminate_,
  });

  OptPassGroupMap map_a({{"c_1", c_1}, {"renormalize", opt::OptPassConfig::Renormalize()}});

  return map_a;
}

OptPassGroupMap GetOptPassesB(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig b_1 = opt::OptPassConfig(
    {irpass.zero_like_fill_zero_, irpass.item_tuple_eliminate_, irpass.float_tuple_getitem_switch_,
     irpass.reset_defer_inline_, irpass.inline_, irpass.special_op_eliminate_, irpass.get_make_ref_eliminate_,
     irpass.incorporate_env_getitem_, irpass.incorporate_env_getitem_switch_, irpass.env_get_item_eliminate_,
     irpass.value_based_eliminate_});
  opt::OptPassConfig b_2 = opt::OptPassConfig({
    irpass.replace_refkey_by_param_,
    irpass.make_ref_eliminate_,
    irpass.get_ref_param_eliminate_,
    irpass.row_tensor_eliminate_,
  });
  OptPassGroupMap map({
    {"b_1", b_1},
    {"b_2", b_2},
    {"renormalize", opt::OptPassConfig::Renormalize()},
    {"cse", opt::OptPassConfig(opt::CSE(false))},
  });
  return map;
}

OptPassGroupMap GetOptPassesGraphKernelA(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig interface_fusion = opt::OptPassConfig({
    irpass.mark_interface_fusion_,
  });
  OptPassGroupMap map({
    {"graph_kernel_reuse", opt::OptPassConfig(opt::GraphKernelReuse())},
    {"interface_fusion", interface_fusion},
    {"renormalize", opt::OptPassConfig::Renormalize()},
    {"cse", opt::OptPassConfig(opt::CSE(false))},
  });
  return map;
}

OptPassGroupMap GetOptPassesGraphKernelB(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig elim_1 = opt::OptPassConfig({
    irpass.addn_eliminate_,
    irpass.incorporate_getitem_from_param_,
  });
  opt::OptPassConfig elim_2 = opt::OptPassConfig({
    irpass.unused_parameter_eliminate_,
    irpass.unused_output_eliminate_,
  });
  OptPassGroupMap map({
    {"elim_1", elim_1},
    {"renormalize", opt::OptPassConfig::Renormalize()},
    {"elim_2", elim_2},
  });
  return map;
}

OptPassGroupMap GetOptPassesC(const opt::irpass::OptimizeIRPassLib &irpass) {
  return OptPassGroupMap({{"renormalize", opt::OptPassConfig::Renormalize()}});
}

OptPassGroupMap GetControlPhases(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig control_group = opt::OptPassConfig({irpass.convert_switch_replacement_}, true);
  OptPassGroupMap map({
    {"control_group", control_group},
    {"renormalize", opt::OptPassConfig::Renormalize()},
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

static std::unordered_map<std::string, std::shared_ptr<Optimizer>> g_pass_opts = {};

void InitOpt(const ResourcePtr &res) {
  if (g_pass_opts.size() == 0) {
    opt::irpass::OptimizeIRPassLib irpass;
    g_pass_opts["opt_a"] = Optimizer::MakeOptimizer("opt_a", res, GetOptPassesA(irpass));
    g_pass_opts["opt_b"] = Optimizer::MakeOptimizer("opt_b", res, GetOptPassesB(irpass), false, true);
    g_pass_opts["opt_after_cconv"] =
      Optimizer::MakeOptimizer("opt_after_cconv", res, GetOptPassesAfterCconv(irpass), false, true);
    g_pass_opts["opt_graph_kernel_a"] =
      Optimizer::MakeOptimizer("opt_graph_kernel_a", res, GetOptPassesGraphKernelA(irpass), true);
    g_pass_opts["opt_graph_kernel_b"] =
      Optimizer::MakeOptimizer("opt_graph_kernel_b", res, GetOptPassesGraphKernelB(irpass), false);
    g_pass_opts["renormal"] = Optimizer::MakeOptimizer("renormal", res, GetOptPassesC(irpass));
    g_pass_opts["opt_control"] = Optimizer::MakeOptimizer("opt_control", res, GetControlPhases(irpass), false, true);
    g_pass_opts["opt_prepare"] = Optimizer::MakeOptimizer("opt_prepare", res, GetPreparePhases(irpass));
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    if (!(context_ptr->enable_graph_kernel())) {
      g_pass_opts["opt_graph_kernel_a"]->set_enable(false);
      g_pass_opts["opt_graph_kernel_b"]->set_enable(false);
    }
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
  if (res->func_graph() == nullptr) {
    MS_LOG(ERROR) << "Opt passes int error";
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

bool OptPassAGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_a"); }
bool OptPassBGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_b"); }
bool OptPassAfterCconvGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_after_cconv"); }
bool OptPassGraphKernelGroupA(const ResourcePtr &res) { return OptPassGroup(res, "opt_graph_kernel_a"); }
bool OptPassGraphKernelGroupB(const ResourcePtr &res) { return OptPassGroup(res, "opt_graph_kernel_b"); }
bool ControlGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_control"); }
bool PrepareGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_prepare"); }

bool OptPassRNGroup(const ResourcePtr &res) { return OptPassGroup(res, "renormal"); }

bool AddControlDependPass(const ResourcePtr &res) {
  FuncGraphPtr func_graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);

  if (func_graph->has_flag(GRAPH_FLAG_EFFECT_PATIAL_ORDER)) {
    opt::AddControlDepend(func_graph);
  }
  for (auto fg : func_graph->func_graphs_used_total()) {
    MS_EXCEPTION_IF_NULL(fg);
    if (fg->has_flag(GRAPH_FLAG_EFFECT_PATIAL_ORDER)) {
      opt::AddControlDepend(fg);
    }
  }
  return true;
}

bool MergeDupGraphPass(const ResourcePtr &res) {
  FuncGraphPtr func_graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(res->manager());
  if (res->manager()->func_graphs().size() <= 1) {
    return true;
  }
  return MergeDuplicateGraphs(res->manager());
}

bool RemoveValueNodeDuplicationsPass(const ResourcePtr &res) {
  if (res->func_graph() == nullptr) {
    MS_LOG(EXCEPTION) << "Remove value node duplications error.";
  }
  auto manager = res->manager();
  HashCache hash_cache;
  HashValue hashes;
  // Remove duplicated value nodes across all graphs in manager
  for (auto &fg : manager->func_graphs()) {
    auto value_nodes = fg->value_nodes();
    for (const auto &value_pair : value_nodes) {
      TryToDoReplace(manager.get(), value_pair.first, &hash_cache, &hashes);
    }
  }
  return true;
}

bool CconvPass(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res->func_graph());
  FuncGraphPtr func_graph = res->func_graph();
  FuncGraphPtr new_fg = LiftingClone(func_graph);
  res->set_func_graph(new_fg);
  return true;
}

bool ValidatePass(const ResourcePtr &res) {
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

std::vector<PassItem> kVmPasses = {{"simplify_data_structures", SimplifyDataStructuresPass},
                                   {"opt_a", OptPassAGroup},
                                   {"clean_after_opta", CleanAfterOptAPass},
                                   {"opt_b", OptPassBGroup},
                                   {"cconv", CconvPass},
                                   {"opt_after_cconv", OptPassAfterCconvGroup},
                                   {"remove_dup_value", RemoveValueNodeDuplicationsPass},
                                   {"opt_graph_kernel_a", OptPassGraphKernelGroupA},
                                   {"opt_graph_kernel_b", OptPassGraphKernelGroupB},
                                   {"add_control_depend", AddControlDependPass}};

std::vector<PassItem> kGePasses = {{"simplify_data_structures", SimplifyDataStructuresPass},
                                   {"opt_a", OptPassAGroup},
                                   {"clean_after_opta", CleanAfterOptAPass},
                                   {"opt_b", OptPassBGroup},
                                   {"add_control_depend", AddControlDependPass},
                                   {"opt_control", ControlGroup},
                                   {"opt_prepare", PrepareGroup},
                                   {"cconv", CconvPass}};

std::vector<PassItem> kPynativePasses = {{"opt_a", OptPassAGroup}, {"opt_b", OptPassBGroup}, {"cconv", CconvPass}};
}  // namespace pipeline
}  // namespace mindspore
