/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "include/common/utils/compile_cache_context.h"
#include "utils/ms_context.h"
#include "utils/log_adapter.h"

namespace mindspore {
bool CompileCacheEnable() {
  auto enable = mindspore::MsContext::GetInstance()->get_param<bool>(mindspore::MS_CTX_ENABLE_COMPILE_CACHE);
  if (!enable) {
    enable = !mindspore::common::GetEnv(mindspore::kCompilerCacheEnable).empty();
  }
  return enable;
}

AnfNodePtr CompileCacheContext::FindFrontNodeByFrontName(const std::string &name) const {
  auto iter = front_name_to_front_node_.find(name);
  if (iter != front_name_to_front_node_.end()) {
    return iter->second;
  }
  return nullptr;
}

void CompileCacheContext::InsertFrontNameToFrontNode(const std::string &name, const AnfNodePtr &node) {
  front_name_to_front_node_[name] = node;
}

AnfNodePtr CompileCacheContext::FindBackNodeByBackName(const std::string &name) const {
  auto iter = back_name_to_back_node_.find(name);
  if (iter != back_name_to_back_node_.end()) {
    return iter->second;
  }
  return nullptr;
}

void CompileCacheContext::InsertBackNameToBackNode(const std::string &name, const AnfNodePtr &node) {
  back_name_to_back_node_[name] = node;
}

void CompileCacheContext::SetFusionOpBuildInfoFlag(bool fusion_op_build_info_flag) {
  fusion_op_build_info_flag_ = fusion_op_build_info_flag;
}

void CompileCacheContext::SetChildGraphs(const std::vector<FuncGraphPtr> &child_graphs) {
  child_graphs_ = child_graphs;
}

void CompileCacheContext::SetGraphExecutionOrder(const FuncGraphPtr &graph, const std::vector<CNodePtr> &orders) {
  auto iter = graph_execution_order_map_.find(graph);
  if (iter != graph_execution_order_map_.end()) {
    return;
  }
  graph_execution_order_map_[graph] = orders;
}

std::vector<CNodePtr> CompileCacheContext::GraphExecutionOrder(const FuncGraphPtr &graph) const {
  auto iter = graph_execution_order_map_.find(graph);
  if (iter != graph_execution_order_map_.end()) {
    return iter->second;
  }
  return {};
}

std::string CompileCacheContext::GetBackendGraphDir() {
  return CompileCacheDir() + "/" + mindspore::kBackendGraphCacheSubDir;
}

std::string CompileCacheContext::GetKernelGraphCachePath(size_t frontend_idx) {
  const auto &dir = GetBackendGraphDir();
  return dir + "/" + kCompileCacheFileName + "_" + std::to_string(frontend_idx);
}

void CompileCacheContext::AddBackendGraphToFrontendGraph(const FuncGraphPtr &backend_graph, FuncGraph *frontend_graph) {
  backend_graph_to_frontend_graph_[backend_graph] = frontend_graph;
}

FuncGraph *CompileCacheContext::GetFrontendGraphByBackendGraph(const FuncGraphPtr &graph) const {
  auto iter = backend_graph_to_frontend_graph_.find(graph);
  if (iter != backend_graph_to_frontend_graph_.end()) {
    return iter->second;
  }
  return nullptr;
}

void CompileCacheContext::InsertBackendParamGenFromFrontendParam(const AnfNodePtr &node) {
  backend_param_gen_from_frontend_param_.insert(node);
}

void CompileCacheContext::PushFullnameIoSizeInfo(const std::string &fullname, const CachedIOSizeInfo &io_size) {
  fullname_io_size[fullname] = io_size;
}

CachedIOSizeInfo CompileCacheContext::GetIOSizeInfo(const std::string &fullname) const {
  auto iter = fullname_io_size.find(fullname);
  if (iter != fullname_io_size.end()) {
    return iter->second;
  }
  return CachedIOSizeInfo();
}

void CompileCacheContext::Clear() {
  front_name_to_front_node_.clear();
  back_name_to_back_node_.clear();
  front_graph_ = nullptr;
  use_compile_cache_ = false;
  fusion_op_build_info_flag_ = false;
  compile_id_ = 0;
  compile_cache_dir_ = "";
  role_ = "";
  child_graphs_.clear();
  graph_execution_order_map_.clear();
  backend_graph_to_frontend_graph_.clear();
  fullname_io_size.clear();
  backend_param_gen_from_frontend_param_.clear();
  param_name_to_node_.clear();
}
}  // namespace mindspore
