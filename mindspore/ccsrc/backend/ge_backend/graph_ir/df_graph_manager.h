/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_GRAPH_IR_GRAPH_MANAGER_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_GRAPH_IR_GRAPH_MANAGER_H_

#include <set>
#include <string>
#include <memory>
#include <vector>
#include <mutex>
#include <map>
#include "backend/ge_backend/graph_ir/types.h"
#include "ir/anf.h"
#include "include/backend/visible.h"

namespace mindspore::backend::ge_backend {
class GraphRunner;

class BACKEND_EXPORT DfGraphManager {
 public:
  ~DfGraphManager();
  void ClearGraph() noexcept;

  static DfGraphManager &GetInstance();
  Status AddGraph(const std::string &name, const DfGraphPtr &graph,
                  const DfGraphConfig &graph_config = DfGraphConfig({}, false, false, false));
  std::vector<DfGraphWrapperPtr> GetAllGraphs();
  std::set<string> GetSavedGraphs();
  void AddSavedGraphs(const std::string &id);
  DfGraphWrapperPtr GetGraphByName(const std::string &name);
  DfGraphManager(const DfGraphManager &) = delete;
  DfGraphManager &operator=(const DfGraphManager &) = delete;
  void SetAnfGraph(const std::string &name, const AnfGraphPtr &anf_graph_ptr);
  AnfGraphPtr GetAnfGraph(uint32_t graph_id);
  std::shared_ptr<backend::ge_backend::GraphRunner> GetGraphRunner();
  void SetGraphRunner(const std::shared_ptr<backend::ge_backend::GraphRunner> &graph_runner_ptr) noexcept;
  void DeleteGraphRunner() noexcept;
  void SetGeSession(const std::shared_ptr<::ge::Session> &sess_ptr);
  std::shared_ptr<::ge::Session> GetGeSession();
  void DeleteGeSession() noexcept;
  void AoeGeGraph();

 private:
  DfGraphManager();
  int GenerateId();

  std::mutex lock_;
  std::map<std::string, DfGraphWrapperPtr> graphs_;
  std::set<string> saved_graphs_;
  int graph_id_ = 0;
  std::map<uint32_t, AnfGraphPtr> anf_graphs_;
  std::shared_ptr<backend::ge_backend::GraphRunner> graph_runner_ptr_ = nullptr;
  std::shared_ptr<::ge::Session> sess_ptr_;
};
}  // namespace mindspore::backend::ge_backend

#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_GRAPH_IR_GRAPH_MANAGER_H_
