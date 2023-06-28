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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_COMPILE_CACHE_CONTEXT_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_COMPILE_CACHE_CONTEXT_H_

#include <string>
#include <map>
#include <vector>

#include "include/common/utils/utils.h"
#include "include/common/visible.h"
#include "ir/anf.h"

namespace mindspore {
constexpr char kGraphCacheSubDir[] = "graph_cache";
constexpr char kBackendGraphCacheSubDir[] = "backend_graph_cache";
constexpr char kCompileCacheFileName[] = "compile_cache";
constexpr char kMindIrSuffix[] = ".mindir";
constexpr char kJsonSuffix[] = ".json";
constexpr char kDepFilesHashPath[] = "compile_dependency.hash";
constexpr char kRoleServer[] = "server_";
constexpr char kRolePServer[] = "pserver_";
constexpr char kRolePScheduler[] = "pscheduler_";
constexpr char kGroupCkptFileName[] = "group.ckpt";

struct CachedIOSizeInfo {
  std::string json_name;
  std::vector<size_t> input_size_list;
  std::vector<size_t> output_size_list;
};

COMMON_EXPORT bool CompileCacheEnable();

class COMMON_EXPORT CompileCacheContext {
 public:
  CompileCacheContext(const CompileCacheContext &) = delete;
  CompileCacheContext &operator=(const CompileCacheContext &) = delete;
  static CompileCacheContext &GetInstance() noexcept {
    static CompileCacheContext instance;
    return instance;
  }
  void SetFrontNameToFrontNode(const HashMap<std::string, AnfNodePtr> &map) { front_name_to_front_node_ = map; }
  AnfNodePtr FindFrontNodeByFrontName(const std::string &name) const;
  void ClearFrontNameToFrontNode() { front_name_to_front_node_.clear(); }
  void InsertFrontNameToFrontNode(const std::string &name, const AnfNodePtr &node);

  void SetBackNameToBackNode(const HashMap<std::string, AnfNodePtr> &map) { back_name_to_back_node_ = map; }
  AnfNodePtr FindBackNodeByBackName(const std::string &name) const;
  void ClearBackNameToBackNode() { back_name_to_back_node_.clear(); }
  void InsertBackNameToBackNode(const std::string &name, const AnfNodePtr &node);

  bool UseCompileCache() const { return use_compile_cache_; }
  void SetUseCompileCache(bool use_compile_cache) { use_compile_cache_ = use_compile_cache; }

  bool FusionOpBuildInfoFlag() const { return fusion_op_build_info_flag_; }
  void SetFusionOpBuildInfoFlag(bool fusion_op_build_info_flag);

  void SetFrontGraph(const FuncGraphPtr &graph) { front_graph_ = graph; }
  FuncGraphPtr FrontGraph() const { return front_graph_; }

  size_t CompileId() const { return compile_id_; }
  void SetCompileId(const size_t &compile_id) { compile_id_ = compile_id; }

  void SetCompileCacheDir(const std::string &dir) { compile_cache_dir_ = dir; }
  std::string CompileCacheDir() const { return compile_cache_dir_; }
  void SetRole(const std::string &role) { role_ = role; }
  std::string Role() const { return role_; }

  void SetChildGraphs(const std::vector<FuncGraphPtr> &child_graphs);
  std::vector<FuncGraphPtr> GetChileGraphs() const { return child_graphs_; }

  void ClearChildGraphs() { child_graphs_.clear(); }

  void SetGraphExecutionOrder(const FuncGraphPtr &graph, const std::vector<CNodePtr> &orders);
  std::vector<CNodePtr> GraphExecutionOrder(const FuncGraphPtr &graph) const;
  void ClearGraphExecutionOrder() { graph_execution_order_map_.clear(); }

  std::string GetBackendGraphDir();

  std::string GetKernelGraphCachePath(size_t frontend_idx);

  void AddBackendGraphToFrontendGraph(const FuncGraphPtr &backend_graph, FuncGraph *frontend_graph);
  FuncGraph *GetFrontendGraphByBackendGraph(const FuncGraphPtr &graph) const;

  void PushFullnameIoSizeInfo(const std::string &fullname, const CachedIOSizeInfo &io_size);
  CachedIOSizeInfo GetIOSizeInfo(const std::string &fullname) const;

  void InsertParamNameToNode(const std::string &name, const AnfNodePtr &node) { param_name_to_node_[name] = node; }
  mindspore::HashMap<std::string, AnfNodePtr> GetParamNameToNodeMap() const { return param_name_to_node_; }

  void InsertBackendParamGenFromFrontendParam(const AnfNodePtr &node);
  bool IsBackendParamGenFromFrontendParam(const AnfNodePtr &node) const {
    return backend_param_gen_from_frontend_param_.count(node);
  }
  void Clear();

 private:
  CompileCacheContext() = default;
  ~CompileCacheContext() = default;
  // name is unique for node here.
  HashMap<std::string, AnfNodePtr> front_name_to_front_node_;
  HashMap<std::string, AnfNodePtr> back_name_to_back_node_;

  // The number of front-end and back-end graphs is one-to-many per compile.
  FuncGraphPtr front_graph_;
  bool use_compile_cache_{false};
  bool fusion_op_build_info_flag_{false};
  size_t compile_id_{0};
  std::string compile_cache_dir_;
  std::string role_;
  // key is parent funcgarph, values is child funcgraph.
  std::vector<FuncGraphPtr> child_graphs_;
  std::map<FuncGraphPtr, std::vector<CNodePtr>> graph_execution_order_map_;
  HashMap<FuncGraphPtr, FuncGraph *> backend_graph_to_frontend_graph_;
  std::map<std::string, CachedIOSizeInfo> fullname_io_size;
  // param is a backend node but we can find its correspond frontend param.
  mindspore::HashSet<AnfNodePtr> backend_param_gen_from_frontend_param_;
  mindspore::HashMap<std::string, AnfNodePtr> param_name_to_node_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_COMPILE_CACHE_CONTEXT_H_
