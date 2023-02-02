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
#include "ir/func_graph_base.h"

#include <list>
#include <algorithm>
#include "ir/func_graph.h"
#include "ir/meta_func_graph.h"

namespace mindspore {
FuncGraphLoopBreaker::~FuncGraphLoopBreaker() {
  std::lock_guard<std::mutex> lock_set(func_mutex_);
  for (auto fg : func_set_) {
    fg->reg_flg_ = false;
  }
}

FuncGraphLoopBreaker &FuncGraphLoopBreaker::Inst() {
  static FuncGraphLoopBreaker mgr;
  return mgr;
}

MS_CORE_API const FuncGraphChecker &FuncGraphBase::GetChecker(const std::string &checker_name) {
  auto it = checkers_.find(checker_name);
  if (it == checkers_.cend()) {
    static FuncGraphChecker empty_checker;
    return empty_checker;
  }
  return *(it->second);
}

MS_CORE_API void FuncGraphBase::AddChecker(const std::string &checker_name,
                                           const std::shared_ptr<FuncGraphChecker> &new_checker) {
  (void)checkers_.emplace(checker_name, new_checker);
}

void FuncGraphLoopBreaker::BreakLoop() {
  MS_LOG(INFO) << "Size of not recycled graph before break loop is:" << func_set_.size();
  std::list<FuncGraphBasePtr> func_list;

  // Generate shared_ptr for every graph, to avoid func_set_ changes while BreakLoop
  std::for_each(func_set_.begin(), func_set_.end(), [&func_list](FuncGraphBase *fun) {
    if (fun != nullptr && !fun->subclass_destruct_flag_) {
      func_list.emplace_back(fun->shared_from_base<FuncGraphBase>());
    }
  });
  for (auto &item : func_list) {
    if (item == nullptr) {
      continue;
    }
    item->DoBreakLoop();
  }
  func_list.clear();

  int func_graph_cnt = 0;
  for (auto item : func_set_) {
    if (item->isa<FuncGraph>()) {
      MS_LOG(INFO) << "Unfree graph info:" << item->ToString();
      func_graph_cnt++;
    }
  }
  if (func_graph_cnt > 0) {
    MS_LOG(INFO) << "Size of not recycled graph after break loop should be 0, but got:" << func_graph_cnt << "\n"
                 << "Please check the usage of clear_compile_cache or contact to the maintenance engineers.";
  }
}

void FuncGraphLoopBreaker::CleanMetaFuncGraphCache() {
  std::list<FuncGraphBasePtr> func_list;

  // Generate shared_ptr for every graph, to avoid func_set_ changes while BreakLoop
  std::for_each(func_set_.begin(), func_set_.end(), [&func_list](FuncGraphBase *fun) {
    if (fun != nullptr && !fun->subclass_destruct_flag_) {
      func_list.emplace_back(fun->shared_from_base<FuncGraphBase>());
    }
  });
  for (auto item : func_list) {
    if (item != nullptr && item->isa<MetaFuncGraph>()) {
      item->DoBreakLoop();
    }
  }
}

void FuncGraphLoopBreaker::ClearCellGraphs(const std::string &phase) {
  std::list<FuncGraphBasePtr> func_list;
  // Generate shared_ptr for every graph, to avoid func_set_ changes while BreakLoop
  std::for_each(func_set_.begin(), func_set_.end(), [&func_list](FuncGraphBase *fun) {
    if (fun != nullptr && !fun->subclass_destruct_flag_) {
      func_list.emplace_back(fun->shared_from_base<FuncGraphBase>());
    }
  });
  for (auto item : func_list) {
    if (item != nullptr && item->isa<FuncGraph>()) {
      auto func_graph = item->cast<FuncGraphPtr>();
      if (func_graph != nullptr && func_graph->phase() == phase) {
        func_graph->DoBreakLoop();
      }
    }
  }
}
}  // namespace mindspore
