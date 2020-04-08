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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_OPTIMIZER_H_
#define MINDSPORE_CCSRC_OPTIMIZER_OPTIMIZER_H_

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <initializer_list>

#ifdef DEBUG
#include "debug/draw.h"
#include "debug/anf_ir_dump.h"
#endif
#include "optimizer/opt.h"
#include "pipeline/resource.h"
#include "pipeline/action.h"
#include "debug/trace.h"

namespace mindspore {
namespace opt {
using OptimizeGraphFunc = std::function<bool(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer)>;

class OptPassConfig {
 public:
  explicit OptPassConfig(const OptimizeGraphFunc &func) : func_(func) {}
  explicit OptPassConfig(const std::vector<SubstitutionPtr> &list, bool is_once = false)
      : list_(list), is_once_(is_once) {}
  OptPassConfig(const std::initializer_list<SubstitutionPtr> &list, bool is_once = false)
      : list_(list), is_once_(is_once) {}
  ~OptPassConfig() = default;

  const std::vector<SubstitutionPtr> &list() const { return list_; }
  const OptimizeGraphFunc &func() const { return func_; }

  static OptPassConfig Renormalize() { return OptPassConfig(); }
  const bool is_renormalize() const { return is_renormalize_; }

  const bool is_once() const { return is_once_; }

 private:
  OptPassConfig() : is_renormalize_(true) {}

  OptimizeGraphFunc func_;
  std::vector<SubstitutionPtr> list_;
  bool is_renormalize_{false};
  bool is_once_{false};
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
  Optimizer(const std::string &name, const pipeline::ResourceBasePtr &resource_ptr)
      : name_(name), resource_(resource_ptr), run_only_once_(false), is_watch_renormalize_(false) {}
  virtual ~Optimizer() = default;

  void Init(const OptPassGroupMap &passes, bool run_only_once) {
    run_only_once_ = run_only_once;
    is_watch_renormalize_ = false;

    for (auto &iter : passes) {
      const std::string &name = iter.first;
      pass_names_.push_back(name);

      const OptPassConfig &config = iter.second;
      if (config.is_renormalize()) {
        passes_.push_back(OptPass::Renormalize());
        continue;
      }

      if (config.list().size() > 0) {
        OptimizeGraphFunc func = SubstitutionList(config.list(), config.is_once());
        passes_.push_back(OptPass(func));
        continue;
      }

      passes_.push_back(OptPass(config.func()));
    }

    if (passes_.size() == 1) {
      run_only_once_ = true;
    }
  }

  static std::shared_ptr<Optimizer> MakeOptimizer(const std::string &name, const pipeline::ResourceBasePtr resource_ptr,
                                                  const OptPassGroupMap &passes, bool run_only_once = false,
                                                  bool watch_renormalize = false) {
    OptimizerPtr optimizer = std::make_shared<Optimizer>(name, resource_ptr);
    optimizer->Init(passes, run_only_once);
    if (watch_renormalize) {
      optimizer->enable_watch_renormalize();
    }
    return optimizer;
  }

  FuncGraphPtr step(FuncGraphPtr func_graph, bool use_profile = true) {
    // Optimizer step counter;
    int counter = 1;
    bool changes = true;

    while (changes) {
      changes = false;
      auto run_runc = [&counter, &func_graph, &changes, use_profile, this]() {
        for (size_t i = 0; i < passes_.size(); ++i) {
          const OptPass &opt = passes_[i];
          auto opt_func = [&func_graph, &changes, &opt, this]() {
            if (opt.is_renormalize()) {
              auto resource_ptr = std::dynamic_pointer_cast<pipeline::Resource>(resource_);
              if (resource_ptr != nullptr) {
                // StepParallel may replace the AbstractValue of the parameters of func_graph,
                // So generate the args_spec from parameters.
                abstract::AbstractBasePtrList maybe_new_args_spec;
                if (is_watch_renormalize_) {
                  if (untyped_nodes_.size() > 0) {
                    std::transform(func_graph->parameters().begin(), func_graph->parameters().end(),
                                   std::back_inserter(maybe_new_args_spec),
                                   [](AnfNodePtr param) -> AbstractBasePtr { return param->abstract(); });
                    func_graph = pipeline::Renormalize(resource_ptr, func_graph, maybe_new_args_spec);
                    clear_untyped_nodes();
                  } else {
                    MS_LOG(INFO) << "Optimizer::step: Skipping Renormalize because untyped_nodes_ is empty.";
                  }
                } else {
                  std::transform(func_graph->parameters().begin(), func_graph->parameters().end(),
                                 std::back_inserter(maybe_new_args_spec),
                                 [](AnfNodePtr param) -> AbstractBasePtr { return param->abstract(); });
                  func_graph = pipeline::Renormalize(resource_ptr, func_graph, maybe_new_args_spec);
                }
              }
            } else if (opt(func_graph, shared_from_this())) {
              changes = true;
            }
          };
          use_profile ? (WITH(MsProfile::GetProfile()->Step(pass_names_[i])) opt_func) : opt_func();
#ifdef DEBUG
          MS_LOG(DEBUG) << "" << name_ << " round " << counter << " OptPass " << pass_names_[i] << " end.";
          auto fg_name = name_ + "_r" + std::to_string(counter) + "_" + std::to_string(i) + "_" + pass_names_[i];
          func_graph->DumpFuncGraph(fg_name);
          DumpIR(fg_name + ".ir", func_graph);
          MS_LOG(DEBUG) << "Dump " << pass_names_[i] << " func graph.";
#endif
        }
      };
      use_profile ? (WITH(MsProfile::GetProfile()->Lap(counter++)) run_runc) : run_runc();

      if (run_only_once_) {
        break;
      }
    }

    auto keep_root = [&func_graph, this]() {
      std::vector<FuncGraphPtr> func_graphs;
      func_graphs.push_back(func_graph);
      resource_->manager()->KeepRoots(func_graphs);
    };
    use_profile ? WITH(MsProfile::GetProfile()->Step("keep_roots")) keep_root : keep_root();
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

  void add_node_to_renormalize(AnfNodePtr anode) {
    if (std::find(untyped_nodes_.begin(), untyped_nodes_.end(), anode) == untyped_nodes_.end()) {
      untyped_nodes_.push_back(anode);
    }
  }

  void clear_untyped_nodes() { untyped_nodes_.clear(); }

  void enable_watch_renormalize() { is_watch_renormalize_ = true; }
  void disable_watch_renormalize() { is_watch_renormalize_ = false; }
  bool is_watch_renormalize() { return is_watch_renormalize_; }

 private:
  const std::string name_;
  pipeline::ResourceBasePtr resource_;
  std::vector<OptPass> passes_;
  std::vector<std::string> pass_names_;
  bool run_only_once_;
  std::vector<AnfNodePtr> untyped_nodes_;
  bool is_watch_renormalize_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_OPTIMIZER_H_
