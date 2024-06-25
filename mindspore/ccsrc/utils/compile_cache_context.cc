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
// normalize name for ge regex check
std::string NormalizeString(const std::string &name) {
  std::string norm_str;
  std::for_each(name.begin(), name.end(), [&norm_str](const auto &a) {
    if (isalpha(a) || isalnum(a) || a == '_' || a == '-') {
      norm_str += a;
    }
  });
  const size_t limit_len = 128;
  if (norm_str.size() > limit_len) {
    norm_str = norm_str.substr(norm_str.size() - limit_len);
  }
  return norm_str;
}

CompileCacheContext &CompileCacheContext::GetInstance() noexcept {
  static CompileCacheContext instance{};
  return instance;
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

std::string CompileCacheContext::GetBackendGraphCachePath(const FuncGraphPtr &front_graph) const {
  auto iter = front_graph_to_backend_graph_cache_path_.find(front_graph);
  if (iter != front_graph_to_backend_graph_cache_path_.end()) {
    return iter->second;
  }
  return "";
}

void CompileCacheContext::InsertBackendGraphCachePath(const FuncGraphPtr &front_graph, const std::string &path) {
  front_graph_to_backend_graph_cache_path_[front_graph] = path;
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
  (void)(backend_param_gen_from_frontend_param_.insert(node));
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
  child_graphs_.clear();
  backend_graph_to_frontend_graph_.clear();
  fullname_io_size.clear();
  front_graph_to_backend_graph_cache_path_.clear();
  backend_param_gen_from_frontend_param_.clear();
  restricted_scenarios_ = false;
}
}  // namespace mindspore
