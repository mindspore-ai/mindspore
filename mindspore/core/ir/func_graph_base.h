/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CORE_IR_FUNC_GRAPH_BASE_H_
#define MINDSPORE_MINDSPORE_CORE_IR_FUNC_GRAPH_BASE_H_
#include <set>
#include <mutex>
#include <memory>
#include <string>

#include "ir/anf.h"

namespace mindspore {
class FuncGraphBase;
using FuncGraphBasePtr = std::shared_ptr<FuncGraphBase>;
class MS_CORE_API FuncGraphLoopBreaker {
 public:
  ~FuncGraphLoopBreaker();

  static FuncGraphLoopBreaker &Inst();

  void RegFuncGraphBase(FuncGraphBase *graph) {
    std::lock_guard<std::mutex> lock_set(func_mutex_);
    (void)func_set_.insert(graph);
  }
  void UnRegFuncGraphBase(FuncGraphBase *graph) {
    std::lock_guard<std::mutex> lock_set(func_mutex_);
    (void)func_set_.erase(graph);
  }

  void BreakLoop();

  void CleanMetaFuncGraphCache();

  void ClearCellGraphs(const std::string &phase);

 private:
  FuncGraphLoopBreaker() = default;
  std::set<FuncGraphBase *> func_set_;
  std::mutex func_mutex_;
};

class FuncGraphChecker {
 public:
  FuncGraphChecker() = default;
  template <typename... Ts>
  void AddCheckFunc(const std::shared_ptr<std::function<bool(const Ts &... args)>> &func) {
    func_ = func;
  }

  template <typename... Ts>
  bool Execute(const Ts &... args) const {
    if (func_ == nullptr) {
      return true;
    }
    auto func = reinterpret_cast<std::function<bool(const Ts &... args)> *>(func_.get());
    return (*func)(args...);
  }

 private:
  std::shared_ptr<void> func_{nullptr};
};

class FuncGraphBase : public Value {
 public:
  FuncGraphBase() {
    FuncGraphLoopBreaker::Inst().RegFuncGraphBase(this);
    reg_flg_ = true;
  }

  ~FuncGraphBase() override {
    if (reg_flg_) {
      FuncGraphLoopBreaker::Inst().UnRegFuncGraphBase(this);
    }
  }
  MS_DECLARE_PARENT(FuncGraphBase, Value);

  // Clear the member of FuncGraph to break loop
  virtual void DoBreakLoop() = 0;

  bool has_side_effect_node() const { return has_side_effect_node_; }
  void set_has_side_effect_node(bool has_side_effect_node) { has_side_effect_node_ = has_side_effect_node; }

  MS_CORE_API const FuncGraphChecker &GetChecker(const std::string &checker_name);

  MS_CORE_API void AddChecker(const std::string &checker_name, const std::shared_ptr<FuncGraphChecker> &new_checker);

 protected:
  friend FuncGraphLoopBreaker;
  bool reg_flg_{false};
  // If the subclass (such as FuncGraph) has started destructing.
  bool subclass_destruct_flag_{false};

 private:
  // If the nodes or their callee's nodes contain Depend CNode with isolated side-effect node.
  bool has_side_effect_node_{false};
  HashMap<std::string, std::shared_ptr<FuncGraphChecker>> checkers_;
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CORE_IR_FUNC_GRAPH_BASE_H_
