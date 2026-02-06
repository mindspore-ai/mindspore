/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_SCHEDULER_BASE_PARAMETER_STORE_H_
#define MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_SCHEDULER_BASE_PARAMETER_STORE_H_

#include <string>
#include <memory>
#include <shared_mutex>

#include "utils/hash_map.h"
#include "utils/ms_utils.h"
#include "include/backend/visible.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_parameter_store.h"

namespace mindspore {
namespace runtime {
// The parameter store contains multiple graph parameter stores from different graphs.
// Only one graph parameter store will be chosen when run graph at step beginning.
class BACKEND_EXPORT ParameterStore {
 public:
  static ParameterStore &GetInstance() {
    static ParameterStore instance;
    return instance;
  }
  void Insert(const std::string &graph_name) {
    const auto &iter = graph_parameter_stores_.find(graph_name);
    if (iter != graph_parameter_stores_.end()) {
      MS_LOG(WARNING) << "Graph " << graph_name << " has already graph parameter store.";
      return;
    }
    auto graph_parameter_store = std::make_shared<GraphParameterStore>();
    graph_parameter_stores_[graph_name] = graph_parameter_store;
    chosen_graph_name_ = graph_name;
    chosen_graph_parameter_store_ = graph_parameter_stores_[chosen_graph_name_];
  }

  void Remove(const std::string &graph_name) {
    const auto &iter = graph_parameter_stores_.find(graph_name);
    if (iter != graph_parameter_stores_.end()) {
      graph_parameter_stores_.erase(iter);
    }
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (const auto &iter : graph_parameter_stores_) {
      auto graph_parameter_store = iter.second;
      MS_EXCEPTION_IF_NULL(graph_parameter_store);
      graph_parameter_store->Clear();
    }
    graph_parameter_stores_.clear();
    chosen_graph_parameter_store_ = nullptr;
  }

  void Clear(const string &graph_name) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    const auto &iter = graph_parameter_stores_.find(graph_name);
    if (iter == graph_parameter_stores_.end()) {
      MS_LOG(DEBUG) << "Graph " << graph_name << " has already clear.";
      return;
    }
    auto graph_parameter_store = iter->second;
    if (graph_parameter_store == nullptr) {
      return;
    }
    graph_parameter_store->Clear();
    graph_parameter_stores_.erase(graph_name);
  }

  void SetChosenGraphName(const std::string &graph_name) {
    chosen_graph_name_ = graph_name;
    const auto &iter = graph_parameter_stores_.find(graph_name);
    if (iter == graph_parameter_stores_.end()) {
      MS_LOG(EXCEPTION) << "Parameter stores have no graph " << graph_name;
      return;
    }
    chosen_graph_parameter_store_ = graph_parameter_stores_[chosen_graph_name_];
  }
  std::string &GetChosenGraphName() { return chosen_graph_name_; }
  GraphParameterStore *GetGraphParameterStore() { return chosen_graph_parameter_store_.get(); }

 private:
  ParameterStore() = default;
  ~ParameterStore() = default;
  DISABLE_COPY_AND_ASSIGN(ParameterStore);
  // Map the graph with graph parameter store.
  mindspore::HashMap<std::string, GraphParameterStorePtr> graph_parameter_stores_;
  // The chosen graph name.
  std::string chosen_graph_name_;
  // The chosen graph parameter store at the step beginning.
  GraphParameterStorePtr chosen_graph_parameter_store_;
  mutable std::shared_mutex mutex_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_SCHEDULER_BASE_PARAMETER_STORE_H_
