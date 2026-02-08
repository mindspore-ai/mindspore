/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "include/frontend/optimizer/optimizer.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <initializer_list>

#include "tools/profiler/profiling.h"
#include "tools/profiler/mstx/mstx_impl.h"
#include "mindspore/ccsrc/utils/ir_dump/draw.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "frontend/optimizer/opt.h"
#include "frontend/jit/ps/resource.h"
#include "frontend/jit/ps/action.h"
#include "frontend/jit/ps/pass_config.h"
#include "utils/ms_context.h"
#include "ir/graph_utils.h"
#include "include/frontend/jit/ps/action_interface.h"

namespace mindspore {
namespace opt {

// OptPassConfig
OptPassConfig::OptPassConfig(const OptimizeGraphFunc &func, bool is_once) : func_(func), is_once_(is_once) {}
OptPassConfig::OptPassConfig(const std::vector<SubstitutionPtr> &list, bool is_once, bool global_sensitive)
    : list_(list), is_once_(is_once), global_sensitive_(global_sensitive) {}
OptPassConfig::OptPassConfig(const std::initializer_list<SubstitutionPtr> &list, bool is_once, bool global_sensitive)
    : list_(list), is_once_(is_once), global_sensitive_(global_sensitive) {}

const std::vector<SubstitutionPtr> &OptPassConfig::list() const { return list_; }
const OptimizeGraphFunc &OptPassConfig::func() const { return func_; }

OptPassConfig OptPassConfig::Renormalize(bool run_once) {
  auto config = OptPassConfig();
  config.is_once_ = run_once;
  return config;
}
const bool OptPassConfig::is_renormalize() const { return is_renormalize_; }
const bool OptPassConfig::is_once() const { return is_once_; }
const bool OptPassConfig::global_sensitive() const { return global_sensitive_; }
const bool OptPassConfig::disabled() const { return disabled_; }

void OptPassConfig::set_disabled(bool disabled) { disabled_ = disabled; }

OptPassConfig::OptPassConfig() : is_renormalize_(true) {}

// OptPass
OptPass::OptPass(const OptimizeGraphFunc &func, const std::string &jump_to) : pass_func_(func), jump_to_(jump_to) {}

bool OptPass::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) const {
  return pass_func_(func_graph, optimizer);
}

OptPass OptPass::Renormalize(bool is_once, const std::string &jump_to) { return OptPass(is_once, jump_to); }
const bool OptPass::is_renormalize() const { return is_renormalize_; }

bool OptPass::is_once() const { return is_once_; }
bool OptPass::alreay_run() const { return alreay_run_; }
void OptPass::set_alreay_run(bool alreay_run) { alreay_run_ = alreay_run; }
const std::string OptPass::jump_to() const { return jump_to_; }

OptPass::OptPass(bool is_once, const std::string &jump_to)
    : is_renormalize_(true), is_once_(is_once), jump_to_(jump_to) {}

// Optimizer
Optimizer::Optimizer(const std::string &name, const pipeline::ResourceBasePtr &resource, bool traverse_nodes_first)
    : name_(name),
      resource_(resource),
      run_only_once_(false),
      is_watch_renormalize_(false),
      is_enable_(true),
      is_untyped_generated_(false),
      traverse_nodes_first_(traverse_nodes_first),
      is_first_order_j_(true),
      changes_(false),
      changes_since_last_renorm_(false) {}

bool Optimizer::operator()(const pipeline::ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->func_graph() == nullptr) {
    MS_LOG(ERROR) << "Opt passes error";
    return false;
  }

  auto func_graph = resource->func_graph();
  MS_LOG(DEBUG) << "Start " << name_ << " func graph:" << func_graph->ToString() << ", "
                << func_graph->get_return()->DebugString(true);
  auto new_func_graph = step(func_graph, true, resource);
  resource->set_func_graph(new_func_graph);
  return true;
}

void Optimizer::Init(const OptPassGroupMap &passes, bool run_only_once) {
  run_only_once_ = run_only_once;
  is_watch_renormalize_ = false;
  is_untyped_generated_ = false;
  is_on_debug_ = IS_OUTPUT_ON(mindspore::kDebug);

  for (auto &iter : passes) {
    const OptPassConfig &config = iter.config;
    if (config.disabled()) {
      continue;
    }

    const std::string &name = iter.name;
    pass_names_.push_back(name);
    auto res = pass_name_idx.emplace(name, pass_names_.size() - 1);
    if (!res.second) {
      MS_LOG(INTERNAL_EXCEPTION) << "duplicate pass name: " << name << " in Optimizer " << name_;
    }

    if (config.is_renormalize()) {
      passes_.push_back(OptPass::Renormalize(config.is_once(), iter.jump_to));
      continue;
    }

    if (config.list().size() > 0) {
      OptimizeGraphFunc func = SubstitutionList(config.list(), config.is_once(), config.global_sensitive());
      (void)passes_.emplace_back(func, iter.jump_to);
      continue;
    }

    (void)passes_.emplace_back(config.func(), iter.jump_to);
  }

  if (passes_.size() == 1) {
    run_only_once_ = true;
  }
}

std::shared_ptr<Optimizer> Optimizer::MakeOptimizer(const std::string &name, const pipeline::ResourceBasePtr resource,
                                                    const OptPassGroupMap &passes, bool run_only_once,
                                                    bool watch_renormalize, bool traverse_nodes_first) {
  OptimizerPtr optimizer = std::make_shared<Optimizer>(name, resource, traverse_nodes_first);
  optimizer->Init(passes, run_only_once);
  if (watch_renormalize) {
    optimizer->enable_watch_renormalize();
  }
  return optimizer;
}

std::shared_ptr<Optimizer> Optimizer::MakeEmptyOptimizer(const pipeline::ResourceBasePtr resource) {
  OptimizerPtr optimizer = std::make_shared<Optimizer>("empty", resource, false);
  optimizer->Init(OptPassGroupMap{}, false);
  return optimizer;
}

void Optimizer::DumpStep(FuncGraphPtr func_graph, int counter, int index, int jump_counter) {
  static const auto enable_dump_pass = GetDumpConfig().enable_dump_pass_ir;
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  static const auto input_name = common::GetEnv("MS_DEV_DUMP_IR_PASSES");
  auto enable_dump_pass_ir = (input_name.size() != 0) || enable_dump_pass;
  if ((enable_dump_pass_ir && context->CanDump(kIntroductory)) || context->CanDump(kFully)) {
    auto fg_name = "opt_substep_" + name_ + "_r" + std::to_string(counter) + "_j" + std::to_string(jump_counter) + "_" +
                   std::to_string(index) + "_" + pass_names_[index];
    MS_LOG(DEBUG) << "The opt " << name_ << " round " << counter << " jump " << jump_counter << " OptPass "
                  << pass_names_[index] << " end.";
    static const auto switch_order = (common::GetEnv("MS_DEV_SAVE_GRAPHS_SORT_MODE") == "1");
    if (switch_order) {
      ExportIR(fg_name + ".ir", func_graph);
    } else {
      DumpIR(fg_name + ".ir", func_graph);
    }
    if (context->CanDump(kFully)) {
      draw::Draw(fg_name + ".dot", func_graph);
    }
    MS_LOG(DEBUG) << "Dump " << pass_names_[index] << " func graph.";
  }
}

FuncGraphPtr Optimizer::step(FuncGraphPtr func_graph, bool use_profile, pipeline::ResourceBasePtr res) {
  if (!is_enable_) {
    return func_graph;
  }
  func_graph_ = func_graph;
  if (res) {
    MS_LOG(INFO) << "Run at the custom passes.";
    resource_ = res;
  }
  // Optimizer step counter;
  int counter = 1;
  changes_ = true;
  // If no changes since last renormalization, then no need to do the renormalization again.
  // Set the initial value to true, so the renormalization can be executed once if it's the
  // only pass.
  changes_since_last_renorm_ = true;

  while (changes_) {
    changes_ = false;
    auto run_func = std::bind(&Optimizer::RunFunc, this, &counter, use_profile);
    use_profile ? (ProfileExecute(MsProfile::GetProfile()->Lap(counter), run_func)) : run_func();
    counter++;

    if (run_only_once_) {
      break;
    }
  }
  return func_graph_;
}

void Optimizer::RunFunc(int *counter, bool use_profile) {
  size_t i = 0;
  size_t jump_counter = 0;
  while (i < passes_.size()) {
    OptPass &opt = passes_[i];
    current_pass_ = {*counter, pass_names_[i]};
    auto opt_func = std::bind(&Optimizer::OptProcess, this, &opt);
    auto profiler_pass_name =
      name_ + ".r" + std::to_string(*counter) + ".j" + std::to_string(jump_counter) + "." + pass_names_[i];
    if (FilterPass(profiler_pass_name)) {
      ++i;
      continue;
    }

    uint64_t start_time = profiler::GetClockSyscnt();
    MS_LOG(INFO) << "Start " << profiler_pass_name;
    MSTX_RANGE_GUARD(profiler_pass_name.c_str(), nullptr, profiler::MSTX_DOMAIN_MODEL_PREPARATION);
    ProcessStatus::GetInstance().RecordStart(profiler_pass_name);
    auto last_version = FuncGraphManager::version();
    use_profile ? ProfileExecute(MsProfile::GetProfile()->Step(pass_names_[i]), opt_func) : opt_func();
    auto current_changed = (FuncGraphManager::version() != last_version);
    ProcessStatus::GetInstance().RecordEnd();
    MS_LOG(INFO) << "End " << profiler_pass_name << (current_changed ? ".changed" : ".unchanged");
    (void)profiler::CollectHostInfo(pipeline::kCompiler, pipeline::kOptimize, profiler_pass_name, start_time,
                                    profiler::GetClockSyscnt(), 0);
    if (current_changed) {
      UpdateRunningPasses(profiler_pass_name);
    }
#ifdef ENABLE_DUMP_IR
    DumpStep(func_graph_, *counter, i, jump_counter);
#endif
    if (current_changed && !opt.jump_to().empty()) {
      auto iter = pass_name_idx.find(opt.jump_to());
      if (iter == pass_name_idx.end()) {
        MS_LOG(INTERNAL_EXCEPTION) << "Jump failed, pass `" << opt.jump_to() << "` is not in optimizer " << name_;
      }
      MS_LOG(DEBUG) << "Jump from " << pass_names_[i] << " to " << iter->second << "in optimizer " << name_;
      i = iter->second;
      ++jump_counter;
    } else {
      ++i;
    }
  }
}

pipeline::ResourceBasePtr Optimizer::resource() const { return resource_; }
FuncGraphManagerPtr Optimizer::manager() const {
  if (resource_ != nullptr) {
    return resource_->manager();
  }
  MS_LOG(INTERNAL_EXCEPTION) << "No ResourceBase exists.";
}

const std::string Optimizer::name() const { return name_; }
void Optimizer::set_is_untyped_generated() { is_untyped_generated_ = true; }
void Optimizer::clear_is_untyped_generated() { is_untyped_generated_ = false; }
void Optimizer::enable_watch_renormalize() { is_watch_renormalize_ = true; }
void Optimizer::disable_watch_renormalize() { is_watch_renormalize_ = false; }
bool Optimizer::is_watch_renormalize() const { return is_watch_renormalize_; }
void Optimizer::set_enable(bool enable) { is_enable_ = enable; }
bool Optimizer::traverse_nodes_first() const { return traverse_nodes_first_; }
bool Optimizer::is_first_order_j() const { return is_first_order_j_; }
void Optimizer::set_is_first_order_j(bool is_first_order_j) { is_first_order_j_ = is_first_order_j; }

namespace {
FuncGraphPtr RunRenormalize(const pipeline::ResourcePtr &resource, const FuncGraphPtr &func_graph) {
  // StepParallel may replace the AbstractValue of the parameters of func_graph,
  // So generate the args_abs from parameters.
  abstract::AbstractBasePtrList maybe_new_args;
  std::transform(func_graph->parameters().begin(), func_graph->parameters().end(), std::back_inserter(maybe_new_args),
                 [](const AnfNodePtr &param) -> AbstractBasePtr { return param->abstract(); });

  if (common::GetCompileConfig("CHECK_PASS_NODE_SCOPE") == "1") {
    const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple);
    for (const auto &node : all_nodes) {
      validator::ValidateScope(node, "before_renormalize");
    }
  }

  return pipeline::Renormalize(resource, func_graph, maybe_new_args);
}
}  // namespace

void Optimizer::OptRenormalize() {
  auto resource = std::dynamic_pointer_cast<pipeline::Resource>(resource_);
  if (resource == nullptr) {
    return;
  }

  if (is_watch_renormalize_) {
    if (is_untyped_generated_) {
      func_graph_ = RunRenormalize(resource, func_graph_);
      clear_is_untyped_generated();
    } else {
      MS_LOG(DEBUG) << "Optimizer::step: Skipping Renormalize because is_untyped_generated_ is False.";
    }
  } else {
    func_graph_ = RunRenormalize(resource, func_graph_);
  }
}

void Optimizer::OptProcess(OptPass *opt) {
  if (opt->is_renormalize()) {
    if (!changes_since_last_renorm_) {
      return;
    }
    if (opt->is_once() && opt->alreay_run()) {
      return;
    }
    OptRenormalize();
    changes_since_last_renorm_ = false;
    opt->set_alreay_run(true);
  } else if ((*opt)(func_graph_, shared_from_this())) {
    changes_ = true;
    changes_since_last_renorm_ = true;
  }
  return;
}
}  // namespace opt
}  // namespace mindspore
