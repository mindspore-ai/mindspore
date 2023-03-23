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

#include "transform/graph_ir/df_graph_manager.h"

#include <sstream>

#ifndef ENABLE_LITE_ACL
#include "include/common/utils/python_adapter.h"
#endif

namespace mindspore {
namespace transform {
DfGraphWrapper::DfGraphWrapper(const std::string &name, const int &id, const DfGraphPtr &graph_ptr,
                               const OptionMap &options)
    : name_(name), id_(id), graph_ptr_(graph_ptr), options_(options) {}

DfGraphManager::DfGraphManager() {
  graph_id_ = 0;
  graph_runner_ptr_ = nullptr;
  sess_ptr_ = nullptr;
}

DfGraphManager::~DfGraphManager() {
  // in python first destroy after atexit but in c++ destoy before atexit
  DeleteGraphRunner();
  DeleteGeSession();
  ClearGraph();
#ifndef ENABLE_LITE_ACL
  python_adapter::set_python_env_flag(false);
#endif
}

DfGraphManager &DfGraphManager::GetInstance() {
  static DfGraphManager instance{};
  return instance;
}

int DfGraphManager::GenerateId() {
  graph_id_++;
  if (graph_id_ <= 0) {
    graph_id_ = 1;
  }
  MS_LOG(INFO) << "Generate graph Id : " << graph_id_;
  return graph_id_;
}

Status DfGraphManager::AddGraph(const std::string &name, const DfGraphPtr &graph_ptr, const OptionMap &options) {
  std::lock_guard<std::mutex> lg(lock_);
  if (name.empty()) {
    MS_LOG(ERROR) << "The graph name is null, add graph failed";
    return Status::INVALID_ARGUMENT;
  }

  if (graph_ptr == nullptr) {
    MS_LOG(INFO) << "The new graph {" << name << "}'s pointer is null, add graph failed";
    return Status::INVALID_ARGUMENT;
  }

  int id = GenerateId();
  DfGraphWrapperPtr wrap_ptr = std::make_shared<DfGraphWrapper>(name, id, graph_ptr, options);
  auto ret = graphs_.emplace(name, wrap_ptr);
  if (ret.second == false) {
    MS_LOG(WARNING) << "The graph name:{ " << name << " }is already exists! The old graph will be overwritten!!";
    ret.first->second = wrap_ptr;
  }
  MS_LOG(INFO) << "Add graph " << name << " to GraphManager success!";
  return Status::SUCCESS;
}

std::vector<DfGraphWrapperPtr> DfGraphManager::GetAllGraphs() {
  std::lock_guard<std::mutex> lg(lock_);
  std::vector<DfGraphWrapperPtr> ret;
  std::stringstream ss;
  ss << "{ ";
  for (auto it = graphs_.begin(); it != graphs_.end(); ++it) {
    ss << it->first << ", ";
    (void)ret.emplace_back(it->second);
  }
  ss << "}";
  MS_LOG(INFO) << "Return graphs: " << ss.str();
  return ret;
}
std::set<string> DfGraphManager::GetSavedGraphs() { return saved_graphs_; }

void DfGraphManager::AddSavedGraphs(const std::string &id) { saved_graphs_.insert(id); }

DfGraphWrapperPtr DfGraphManager::GetGraphByName(const std::string &name) {
  std::lock_guard<std::mutex> lg(lock_);
  if (name.empty()) {
    MS_LOG(ERROR) << "The graph name is null";
    return nullptr;
  }

  auto it = graphs_.find(name);
  if (it == graphs_.end()) {
    MS_LOG(INFO) << "Can't found graph name: " << name;
    return nullptr;
  }
  MS_LOG(INFO) << "Return graph: " << name;
  return it->second;
}

void DfGraphManager::ClearGraph() noexcept {
  std::lock_guard<std::mutex> lg(lock_);
  graphs_.clear();
  anf_graphs_.clear();
  MS_LOG(INFO) << "Remove all graphs in GraphManager";
}

void DfGraphManager::SetAnfGraph(const std::string &name, const AnfGraphPtr &anf_graph_ptr) {
  DfGraphWrapperPtr df_graph = GetGraphByName(name);
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Can't found graph name: " << name;
    return;
  }
  std::lock_guard<std::mutex> lg(lock_);
  anf_graphs_[df_graph->id_] = anf_graph_ptr;
}

AnfGraphPtr DfGraphManager::GetAnfGraph(uint32_t graph_id) {
  std::lock_guard<std::mutex> lg(lock_);
  auto iter = anf_graphs_.find(graph_id);
  if (iter == anf_graphs_.end()) {
    MS_LOG(ERROR) << "Can't found anf graph, graph_id = " << graph_id;
    return nullptr;
  }

  return iter->second;
}

void DfGraphManager::SetGeSession(const std::shared_ptr<::ge::Session> &sess_ptr) {
  std::lock_guard<std::mutex> lg(lock_);
  if (sess_ptr == nullptr) {
    return;
  }

  if (sess_ptr_ == nullptr) {
    MS_LOG(INFO) << "Add a new Ge Session success";
  } else {
    MS_LOG(INFO) << "Add a new Ge Session success, the old Ge Session will be overwritten!!";
  }
  sess_ptr_ = sess_ptr;
}

std::shared_ptr<::ge::Session> DfGraphManager::GetGeSession() {
  std::lock_guard<std::mutex> lg(lock_);
  return sess_ptr_;
}

void DfGraphManager::DeleteGeSession() noexcept {
  std::lock_guard<std::mutex> lg(lock_);
  if (sess_ptr_ == nullptr) {
    MS_LOG(INFO) << "Ge Session is not exist";
  } else {
    sess_ptr_ = nullptr;
    saved_graphs_.clear();
    MS_LOG(INFO) << "Delete Ge Session success";
  }
}

void DfGraphManager::SetGraphRunner(const std::shared_ptr<transform::GraphRunner> &graph_runner_ptr) noexcept {
  std::lock_guard<std::mutex> lg(lock_);
  if (graph_runner_ptr == nullptr) {
    MS_LOG(WARNING) << "You are adding a empty GraphRunner";
  }

  if (graph_runner_ptr_ == nullptr) {
    MS_LOG(INFO) << "Add a new GraphRunner success";
  } else {
    MS_LOG(INFO) << "Add a new GraphRunner success, the old GraphRunner will be overwritten!!";
  }
  graph_runner_ptr_ = graph_runner_ptr;
}

std::shared_ptr<transform::GraphRunner> DfGraphManager::GetGraphRunner() {
  std::lock_guard<std::mutex> lg(lock_);
  return graph_runner_ptr_;
}

void DfGraphManager::DeleteGraphRunner() noexcept {
  std::lock_guard<std::mutex> lg(lock_);
  if (graph_runner_ptr_ == nullptr) {
    MS_LOG(INFO) << "GraphRunner is not exist";
  } else {
    graph_runner_ptr_ = nullptr;
    MS_LOG(INFO) << "Delete GraphRunner success";
  }
}
}  // namespace transform
}  // namespace mindspore
