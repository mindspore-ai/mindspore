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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_OPTIMIZER_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_OPTIMIZER_H_

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <initializer_list>

#include "include/common/debug/draw.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/debug/anf_ir_utils.h"
#include "pipeline/jit/debug/trace.h"
#include "frontend/optimizer/opt.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/action.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
using OptimizeGraphFunc = std::function<bool(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer)>;

class OptPassConfig {
 public:
  explicit OptPassConfig(const OptimizeGraphFunc &func) : func_(func) {}
  explicit OptPassConfig(const std::vector<SubstitutionPtr> &list, bool is_once = false, bool global_sensitive = false)
      : list_(list), is_once_(is_once), global_sensitive_(global_sensitive) {}
  OptPassConfig(const std::initializer_list<SubstitutionPtr> &list, bool is_once = false, bool global_sensitive = false)
      : list_(list), is_once_(is_once), global_sensitive_(global_sensitive) {}
  ~OptPassConfig() = default;

  const std::vector<SubstitutionPtr> &list() const { return list_; }
  const OptimizeGraphFunc &func() const { return func_; }

  static OptPassConfig Renormalize() { return OptPassConfig(); }
  const bool is_renormalize() const { return is_renormalize_; }

  const bool is_once() const { return is_once_; }

  const bool global_sensitive() const { return global_sensitive_; }

 private:
  OptPassConfig() : is_renormalize_(true) {}

  OptimizeGraphFunc func_;
  std::vector<SubstitutionPtr> list_;
  bool is_renormalize_{false};
  bool is_once_{false};
  bool global_sensitive_{false};
};

class OptPass {
 public:
  explicit OptPass(const OptimizeGraphFunc &func) : pass_func_(func) {}
  ~OptPass() = default;

  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) const {
    return pass_func_(func_graph, optimizer);
  }

  static OptPass Renormalize() { return OptPass(); }
  const bool is_renormalize() const { return is_renormalize_; }

 private:
  OptPass() : is_renormalize_(true) {}

  OptimizeGraphFunc pass_func_;
  bool is_renormalize_{false};
};
using OptPassGroupMap = std::vector<std::pair<std::string, OptPassConfig>>;

class Optimizer : public std::enable_shared_from_this<Optimizer> {
 public:
  Optimizer(const std::string &name, const pipeline::ResourceBasePtr &resource, bool traverse_nodes_first = true)
      : name_(name),
        resource_(resource),
        run_only_once_(false),
        is_watch_renormalize_(false),
        is_enable_(true),
        is_untyped_generated_(false),
        traverse_nodes_first_(traverse_nodes_first),
        is_first_order_j_(true) {}
  virtual ~Optimizer() = default;

  void Init(const OptPassGroupMap &passes, bool run_only_once) {
    run_only_once_ = run_only_once;
    is_watch_renormalize_ = false;
    is_untyped_generated_ = false;
    is_on_debug_ = IS_OUTPUT_ON(mindspore::kDebug);

    for (auto &iter : passes) {
      const std::string &name = iter.first;
      pass_names_.push_back(name);

      const OptPassConfig &config = iter.second;
      if (config.is_renormalize()) {
        passes_.push_back(OptPass::Renormalize());
        continue;
      }

      if (config.list().size() > 0) {
        OptimizeGraphFunc func = SubstitutionList(config.list(), config.is_once(), config.global_sensitive());
        passes_.push_back(OptPass(func));
        continue;
      }

      passes_.push_back(OptPass(config.func()));
    }

    if (passes_.size() == 1) {
      run_only_once_ = true;
    }
  }

  static std::shared_ptr<Optimizer> MakeOptimizer(const std::string &name, const pipeline::ResourceBasePtr resource,
                                                  const OptPassGroupMap &passes, bool run_only_once = false,
                                                  bool watch_renormalize = false, bool traverse_nodes_first = true) {
    OptimizerPtr optimizer = std::make_shared<Optimizer>(name, resource, traverse_nodes_first);
    optimizer->Init(passes, run_only_once);
    if (watch_renormalize) {
      optimizer->enable_watch_renormalize();
    }
    return optimizer;
  }

  static std::shared_ptr<Optimizer> MakeEmptyOptimizer(const pipeline::ResourceBasePtr resource) {
    OptimizerPtr optimizer = std::make_shared<Optimizer>("empty", resource, false);
    optimizer->Init(OptPassGroupMap{}, false);
    return optimizer;
  }

  FuncGraphPtr step(FuncGraphPtr func_graph, bool use_profile = true) {
    if (!is_enable_) {
      return func_graph;
    }
    // Optimizer step counter;
    int counter = 1;
    bool changes = true;
    // If no changes since last renormalization, then no need to do the renormalization again.
    // Set the initial value to true, so the renormalization can be executed once if it's the
    // only pass.
    bool changes_since_last_renorm = true;

    while (changes) {
      changes = false;
      auto run_runc = [&counter, &func_graph, &changes, &changes_since_last_renorm, use_profile, this]() {
        for (size_t i = 0; i < passes_.size(); ++i) {
          const OptPass &opt = passes_[i];
          CurPass_ = {counter, pass_names_[i]};
          auto opt_func = [&func_graph, &changes, &opt, &changes_since_last_renorm, this]() {
            if (opt.is_renormalize()) {
              if (!changes_since_last_renorm) {
                return;
              }
              auto resource = std::dynamic_pointer_cast<pipeline::Resource>(resource_);
              if (resource != nullptr) {
                // StepParallel may replace the AbstractValue of the parameters of func_graph,
                // So generate the args_abs from parameters.
                abstract::AbstractBasePtrList maybe_new_args_spec;
                if (is_watch_renormalize_) {
                  if (is_untyped_generated_) {
                    std::transform(func_graph->parameters().begin(), func_graph->parameters().end(),
                                   std::back_inserter(maybe_new_args_spec),
                                   [](const AnfNodePtr &param) -> AbstractBasePtr { return param->abstract(); });
                    func_graph = pipeline::Renormalize(resource, func_graph, maybe_new_args_spec);
                    clear_is_untyped_generated();
                  } else {
                    MS_LOG(INFO) << "Optimizer::step: Skipping Renormalize because is_untyped_generated_ is False.";
                  }
                } else {
                  std::transform(func_graph->parameters().begin(), func_graph->parameters().end(),
                                 std::back_inserter(maybe_new_args_spec),
                                 [](const AnfNodePtr &param) -> AbstractBasePtr { return param->abstract(); });
                  func_graph = pipeline::Renormalize(resource, func_graph, maybe_new_args_spec);
                }
              }
              changes_since_last_renorm = false;
            } else if (opt(func_graph, shared_from_this())) {
              changes = true;
              changes_since_last_renorm = true;
            }
          };
          use_profile ? ProfileExecute(MsProfile::GetProfile()->Step(pass_names_[i]), opt_func) : opt_func();
#ifdef ENABLE_DUMP_IR
          static const auto enable_dump_pass_ir = GetDumpConfig().enable_dump_pass_ir;
          auto context = MsContext::GetInstance();
          MS_EXCEPTION_IF_NULL(context);
          if ((enable_dump_pass_ir && context->CanDump(kIntroductory)) || context->CanDump(kFully)) {
            auto fg_name =
              "opt_substep_" + name_ + "_r" + std::to_string(counter) + "_" + std::to_string(i) + "_" + pass_names_[i];
            MS_LOG(DEBUG) << "The opt " << name_ << " round " << counter << " OptPass " << pass_names_[i] << " end.";
            static const auto switch_order = (common::GetEnv("MS_DEV_SAVE_GRAPHS_SORT_MODE") == "1");
            if (switch_order) {
              ExportIR(fg_name + ".ir", func_graph);
            } else {
              DumpIR(fg_name + ".ir", func_graph);
            }
            if (context->CanDump(kFully)) {
              draw::Draw(fg_name + ".dot", func_graph);
            }
            MS_LOG(DEBUG) << "Dump " << pass_names_[i] << " func graph.";
          }
#endif
        }
      };
      use_profile ? (ProfileExecute(MsProfile::GetProfile()->Lap(counter), run_runc)) : run_runc();
      counter++;

      if (run_only_once_) {
        break;
      }
    }
    return func_graph;
  }

  pipeline::ResourceBasePtr resource() const { return resource_; }
  FuncGraphManagerPtr manager() const {
    if (resource_ != nullptr) {
      return resource_->manager();
    }
    MS_LOG(EXCEPTION) << "No ResourceBase exists.";
  }

  const std::string name() const { return name_; }

  void set_is_untyped_generated() { is_untyped_generated_ = true; }
  void clear_is_untyped_generated() { is_untyped_generated_ = false; }

  void enable_watch_renormalize() { is_watch_renormalize_ = true; }
  void disable_watch_renormalize() { is_watch_renormalize_ = false; }
  bool is_watch_renormalize() const { return is_watch_renormalize_; }
  void set_enable(bool enable) { is_enable_ = enable; }

  bool traverse_nodes_first() const { return traverse_nodes_first_; }

  bool is_first_order_j() const { return is_first_order_j_; }
  void set_is_first_order_j(bool is_first_order_j) { is_first_order_j_ = is_first_order_j; }

  struct {
    int64_t counter = 0;
    std::string name;
  } CurPass_;

  bool is_on_debug_{false};

 private:
  const std::string name_;
  pipeline::ResourceBasePtr resource_;
  std::vector<OptPass> passes_;
  std::vector<std::string> pass_names_;
  bool run_only_once_;
  bool is_watch_renormalize_;
  bool is_enable_;
  bool is_untyped_generated_;
  bool traverse_nodes_first_;
  // A flag to indicate if it's the first order J or innermost J in GraphMode.
  bool is_first_order_j_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_OPTIMIZER_H_
