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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPTIMIZER_OPTIMIZER_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPTIMIZER_OPTIMIZER_H_

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "base/base.h"
#include "frontend/optimizer/opt.h"
#include "frontend/jit/ps/validator.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/visible.h"
#include "frontend/jit/ps/resource.h"

namespace mindspore {
namespace opt {

using OptimizeGraphFunc = std::function<bool(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer)>;

class OptPassConfig {
 public:
  explicit OptPassConfig(const OptimizeGraphFunc &func, bool is_once = false);
  explicit OptPassConfig(const std::vector<SubstitutionPtr> &list, bool is_once = false, bool global_sensitive = false);
  OptPassConfig(const std::initializer_list<SubstitutionPtr> &list, bool is_once = false,
                bool global_sensitive = false);
  ~OptPassConfig() = default;

  const std::vector<SubstitutionPtr> &list() const;
  const OptimizeGraphFunc &func() const;
  const bool is_renormalize() const;
  const bool is_once() const;
  const bool global_sensitive() const;
  const bool disabled() const;
  void set_disabled(bool disabled);

  static OptPassConfig Renormalize(bool run_once = false);

 private:
  OptPassConfig();

  OptimizeGraphFunc func_;
  std::vector<SubstitutionPtr> list_;
  bool is_renormalize_{false};
  bool is_once_{false};
  bool global_sensitive_{false};
  bool disabled_{false};
};

class OptPass {
 public:
  explicit OptPass(const OptimizeGraphFunc &func, const std::string &jump_to = "");
  ~OptPass() = default;

  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) const;

  const bool is_renormalize() const;
  bool is_once() const;
  bool alreay_run() const;
  void set_alreay_run(bool alreay_run);
  const std::string jump_to() const;

  static OptPass Renormalize(bool is_once = false, const std::string &jump_to = "");

 private:
  explicit OptPass(bool is_once, const std::string &jump_to = "");

  OptimizeGraphFunc pass_func_;
  bool is_renormalize_{false};
  bool is_once_{false};
  bool alreay_run_{false};
  std::string jump_to_{""};
};

struct OptPassItem {
  std::string name;
  OptPassConfig config;
  std::string jump_to;
  OptPassItem(const std::string &name, const OptPassConfig &config) : name(name), config(config) {}
  OptPassItem(const std::string &name, const OptPassConfig &config, const std::string &jump_to)
      : name(name), config(config), jump_to(jump_to) {}
};

using OptPassGroupMap = std::vector<OptPassItem>;

class FRONTEND_EXPORT Optimizer : public std::enable_shared_from_this<Optimizer> {
 public:
  Optimizer(const std::string &name, const pipeline::ResourceBasePtr &resource, bool traverse_nodes_first = true);
  virtual ~Optimizer() = default;

  bool operator()(const pipeline::ResourcePtr &resource);

  void Init(const OptPassGroupMap &passes, bool run_only_once);

  static std::shared_ptr<Optimizer> MakeOptimizer(const std::string &name, const pipeline::ResourceBasePtr resource,
                                                  const OptPassGroupMap &passes, bool run_only_once = false,
                                                  bool watch_renormalize = false, bool traverse_nodes_first = true);

  static std::shared_ptr<Optimizer> MakeEmptyOptimizer(const pipeline::ResourceBasePtr resource);

  void DumpStep(FuncGraphPtr func_graph, int counter, int index, int jump_counter);
  FuncGraphPtr step(FuncGraphPtr func_graph, bool use_profile = true, pipeline::ResourceBasePtr res = nullptr);
  void RunFunc(int *counter, bool use_profile);

  pipeline::ResourceBasePtr resource() const;
  FuncGraphManagerPtr manager() const;
  const std::string name() const;
  void set_is_untyped_generated();
  void clear_is_untyped_generated();
  void enable_watch_renormalize();
  void disable_watch_renormalize();
  bool is_watch_renormalize() const;
  void set_enable(bool enable);
  bool traverse_nodes_first() const;
  bool is_first_order_j() const;
  void set_is_first_order_j(bool is_first_order_j);

  struct {
    int64_t counter = 0;
    std::string name;
  } current_pass_;

  bool is_on_debug_{false};

 private:
  void OptProcess(OptPass *opt);
  void OptRenormalize();
  const std::string name_;
  pipeline::ResourceBasePtr resource_;
  std::vector<OptPass> passes_;
  std::vector<std::string> pass_names_;
  mindspore::HashMap<std::string, size_t> pass_name_idx;
  bool run_only_once_;
  bool is_watch_renormalize_;
  bool is_enable_;
  bool is_untyped_generated_;
  bool traverse_nodes_first_;
  // A flag to indicate if it's the first order J or innermost J in GraphMode.
  bool is_first_order_j_;
  bool changes_;
  bool changes_since_last_renorm_;
  FuncGraphPtr func_graph_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPTIMIZER_OPTIMIZER_H_
