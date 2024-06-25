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

#include <sstream>
#include <set>
#include "mindspore/core/utils/file_utils.h"
#include "transform/graph_ir/df_graph_manager.h"
#include "transform/graph_ir/aoe_util.h"
#include "utils/ms_context.h"
#include "pipeline/jit/ps/base.h"
#include "utils/phase.h"
#ifndef ENABLE_LITE_ACL
#include "include/common/utils/python_adapter.h"
#endif
#include "include/common/utils/compile_cache_context.h"
#include "include/common/debug/common.h"

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

Status DfGraphManager::AddGraph(const std::string &name, const DfGraphPtr &graph_ptr, const OptionMap &options,
                                const bool &is_cloud) {
  std::lock_guard<std::mutex> lg(lock_);
  if (name.empty()) {
    MS_LOG(ERROR) << "The graph name is null, add graph failed";
    return Status::INVALID_ARGUMENT;
  }

  if (graph_ptr == nullptr) {
    MS_LOG(INFO) << "The new graph {" << name << "}'s pointer is null, cannot add graph.";
    return Status::INVALID_ARGUMENT;
  }

  int id = GenerateId();
  OptionMap new_options = options;
  auto ms_context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  auto soc_version = ms_context_ptr->ascend_soc_version();
  if (ms_context_ptr->get_param<std::string>(MS_CTX_PRECISION_MODE) != "") {
    (new_options)["ge.exec.precision_mode"] = ms_context_ptr->get_param<std::string>(MS_CTX_PRECISION_MODE);
    MS_LOG(INFO) << "Set precision_mode " << ms_context_ptr->get_param<std::string>(MS_CTX_PRECISION_MODE)
                 << " by user.";
  } else if (is_cloud) {
    if (soc_version == "ascend910b" || soc_version == "ascend910c") {
      (new_options)["ge.exec.precision_mode"] = "must_keep_origin_dtype";
      MS_LOG(INFO) << "Set precision_mode must_keep_origin_dtype, soc_version is " << soc_version << ".";
    } else {
      (new_options)["ge.exec.precision_mode"] = "allow_fp32_to_fp16";
      MS_LOG(INFO) << "Set precision_mode allow_fp32_to_fp16, soc_version is " << soc_version << ".";
    }
  } else {
    (new_options)["ge.exec.precision_mode"] = "force_fp16";
    MS_LOG(INFO) << "Set precision_mode force_fp16, soc_version is " << soc_version << ".";
  }
  auto &compile_cache_context = CompileCacheContext::GetInstance();
  auto init_compile_cache = compile_cache_context.init_compile_cache();
  auto enable_compile_cache = compile_cache_context.enable_compile_cache();
  auto dep_files_hash = compile_cache_context.CompileCacheDepFilesHash();
  if (enable_compile_cache && init_compile_cache) {
    auto ge_graph_key = IsEnableRefMode() ? name : std::to_string(id);
    if (!dep_files_hash.empty()) {
      ge_graph_key = dep_files_hash + "_" + ge_graph_key;
    }
    ge_graph_key = NormalizeString(ge_graph_key);
    new_options.insert_or_assign(kGeGraphKey, ge_graph_key);
    auto ge_cache_path = Common::GetCompilerCachePath() + kGeCache;
    (void)mindspore::FileUtils::CreateNotExistDirs(ge_cache_path, true);
    new_options.insert_or_assign(kGeGraphCompilerCacheDir, ge_cache_path);
    MS_LOG(INFO) << "Use GE graph compile cache, GE graph compile cache dir: " << ge_cache_path
                 << ", the ge.graph_key is " << ge_graph_key;
  }

  DfGraphWrapperPtr wrap_ptr = std::make_shared<DfGraphWrapper>(name, id, graph_ptr, new_options);
  auto ret = graphs_.emplace(name, wrap_ptr);
  if (!ret.second) {
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
  for (const auto &graph_id : graphs_) {
    MS_LOG(INFO) << "Remove graph, graph name: " << graph_id.first << ", graph id: " << graph_id.second->id_;
    if (sess_ptr_ != nullptr &&
        sess_ptr_->RemoveGraph(static_cast<uint32_t>(graph_id.second->id_)) != ::ge::GRAPH_SUCCESS) {
      MS_LOG(WARNING) << "Remove graph, graph name: " << graph_id.first << ", graph id: " << graph_id.second->id_;
    }
  }
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
    for (const auto &graph_id : graphs_) {
      MS_LOG(INFO) << "Remove graph, graph name: " << graph_id.first << ", graph id: " << graph_id.second->id_;
      if (sess_ptr_->RemoveGraph(static_cast<uint32_t>(graph_id.second->id_)) != ::ge::GRAPH_SUCCESS) {
        MS_LOG(WARNING) << "Remove graph, graph name: " << graph_id.first << ", graph id: " << graph_id.second->id_;
      }
    }
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

void DfGraphManager::AoeGeGraph() {
  std::set<string> wait_optimize_graphs_ = AoeUtil::GetInstance().GetWaitOptimizeGraph();
  if (wait_optimize_graphs_.empty()) {
    return;
  }
  MS_LOG(DEBUG) << "start optimized graph";
  std::set<string> optimized_graph_names_;
#ifndef ENABLE_LITE_ACL
  py::gil_scoped_release release;
#endif

  for (auto &graph_name : wait_optimize_graphs_) {
    auto wrapper = GetGraphByName(graph_name);
    MS_EXCEPTION_IF_NULL(wrapper);
    if (AoeUtil::GetInstance().IsSaveOptimizedGraph(wrapper->id_)) {
      continue;
    }
    Status status = AoeUtil::GetInstance().AoeOnlineGeGraph(GetGeSession(), wrapper->graph_ptr_);
    if (status == FAILED) {
      MS_LOG(ERROR) << "AOE tuning failed, graph name is " << graph_name << " id :" << wrapper->id_;
      return;
    }
    AoeUtil::GetInstance().SaveOptimizedGraph(wrapper->id_);
    optimized_graph_names_.insert(graph_name);
    MS_LOG(DEBUG) << "Optimized Graph " << graph_name << " success";
  }
  AoeUtil::GetInstance().RemoveWaitOptimizedGraph(optimized_graph_names_);
  optimized_graph_names_.clear();
  MS_LOG(DEBUG) << "optimized graph end";
}
}  // namespace transform
}  // namespace mindspore
