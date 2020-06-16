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

#include "pipeline/pass.h"

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <functional>

#include "ir/func_graph_cloner.h"
#include "pipeline/parse/parse_base.h"
#include "pipeline/parse/data_converter.h"
#include "pipeline/resource.h"
#include "pipeline/validator.h"
#include "optimizer/optimizer.h"
#include "optimizer/cse.h"
#include "optimizer/clean.h"
#include "optimizer/irpass.h"
#include "optimizer/control_depend.h"
#include "parallel/step_parallel.h"
#include "parallel/step_auto_parallel.h"
#include "parallel/allreduce_fusion/step_allreduce_fusion.h"
#include "utils/any.h"

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

namespace {
OptPassGroupMap GetOptPassesA(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig a_1 = opt::OptPassConfig({
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
  });
  opt::OptPassConfig a_2 = opt::OptPassConfig({
    irpass.merge_addn_,
    irpass.float_tuple_getitem_switch_,
    irpass.float_env_getitem_switch_,
    irpass.incorporate_getitem_set_,
    irpass.incorporate_call_,
    irpass.incorporate_call_switch_,
    irpass.incorporate_env_getitem_,
    irpass.incorporate_env_getitem_switch_,
    irpass.new_env_get_item_,
  });
  opt::OptPassConfig a_3 = opt::OptPassConfig({
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
                         {"renormalize", opt::OptPassConfig::Renormalize()},
                         {"cse", opt::OptPassConfig(opt::CSE(false))},
                         {"a_3", a_3}});

  return map_a;
}

OptPassGroupMap GetOptPassesB(const opt::irpass::OptimizeIRPassLib &irpass) {
  opt::OptPassConfig b_1 = opt::OptPassConfig({
    irpass.zero_like_fill_zero_,
    irpass.item_tuple_eliminate_,
    irpass.float_tuple_getitem_switch_,
    irpass.reset_defer_inline_,
    irpass.inline_,
    irpass.special_op_eliminate_,
    irpass.get_make_ref_eliminate_,
  });
  opt::OptPassConfig b_2 = opt::OptPassConfig({
    irpass.replace_refkey_by_param_,
    irpass.make_ref_eliminate_,
  });
  OptPassGroupMap map({
    {"b_1", b_1},
    {"b_2", b_2},
    {"renormalize", opt::OptPassConfig::Renormalize()},
    {"cse", opt::OptPassConfig(opt::CSE(false))},
  });
  return map;
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
    g_pass_opts["opt_control"] = Optimizer::MakeOptimizer("opt_control", res, GetControlPhases(irpass), false, true);
    g_pass_opts["opt_prepare"] = Optimizer::MakeOptimizer("opt_prepare", res, GetPreparePhases(irpass));
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
bool ControlGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_control"); }
bool PrepareGroup(const ResourcePtr &res) { return OptPassGroup(res, "opt_prepare"); }

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
                                   {"opt_b", OptPassBGroup},
                                   {"add_control_depend", AddControlDependPass},
                                   {"cconv", CconvPass}};

std::vector<PassItem> kGePasses = {{"simplify_data_structures", SimplifyDataStructuresPass},
                                   {"opt_a", OptPassAGroup},
                                   {"opt_b", OptPassBGroup},
                                   {"add_control_depend", AddControlDependPass},
                                   {"opt_control", ControlGroup},
                                   {"opt_prepare", PrepareGroup},
                                   {"cconv", CconvPass}};

std::vector<PassItem> kPynativePasses = {{"opt_a", OptPassAGroup}, {"opt_b", OptPassBGroup}, {"cconv", CconvPass}};
}  // namespace pipeline
}  // namespace mindspore
