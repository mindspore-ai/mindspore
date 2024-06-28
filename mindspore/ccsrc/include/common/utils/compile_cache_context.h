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
constexpr char kBackendCompileCacheFileName[] = "backend_compile_cache";
constexpr char kMindIrSuffix[] = ".mindir";
constexpr char kJsonSuffix[] = ".json";
constexpr char kDepFilesHashPath[] = "compile_dependency.hash";
constexpr char kRoleServer[] = "server_";
constexpr char kRolePServer[] = "pserver_";
constexpr char kRolePScheduler[] = "pscheduler_";
constexpr char kGroupCkptFileName[] = "group.ckpt";
constexpr char kDataQueueNameCacheFileName[] = "data_queue_name.json";

struct CachedIOSizeInfo {
  std::string json_name;
  std::vector<size_t> input_size_list;
  std::vector<size_t> output_size_list;
};

COMMON_EXPORT bool CompileCacheEnable();

COMMON_EXPORT std::string NormalizeString(const std::string &name);

class COMMON_EXPORT CompileCacheContext {
 public:
  CompileCacheContext(const CompileCacheContext &) = delete;
  CompileCacheContext &operator=(const CompileCacheContext &) = delete;
  static CompileCacheContext &GetInstance() noexcept;
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

  void SetChildGraphs(const std::vector<FuncGraphPtr> &child_graphs);
  std::vector<FuncGraphPtr> GetChileGraphs() const { return child_graphs_; }
  void ClearChildGraphs() { child_graphs_.clear(); }

  // acquire backend graph cache path according to its correspond front_graph
  std::string GetBackendGraphCachePath(const FuncGraphPtr &front_graph) const;
  void InsertBackendGraphCachePath(const FuncGraphPtr &front_graph, const std::string &path);

  void AddBackendGraphToFrontendGraph(const FuncGraphPtr &backend_graph, FuncGraph *frontend_graph);
  FuncGraph *GetFrontendGraphByBackendGraph(const FuncGraphPtr &graph) const;

  void PushFullnameIoSizeInfo(const std::string &fullname, const CachedIOSizeInfo &io_size);
  CachedIOSizeInfo GetIOSizeInfo(const std::string &fullname) const;

  void InsertBackendParamGenFromFrontendParam(const AnfNodePtr &node);
  bool IsBackendParamGenFromFrontendParam(const AnfNodePtr &node) const {
    return backend_param_gen_from_frontend_param_.count(node) != 0;
  }
  bool RestrictedScenarios() const { return restricted_scenarios_; }
  void SetRestrictedScenarios(bool restricted_scenarios) { restricted_scenarios_ = restricted_scenarios; }
  void Clear();

  void SetCompileCacheDepFilesHash(const std::string &compile_cache_dep_files_hash) {
    compile_cache_dep_files_hash_ = compile_cache_dep_files_hash;
  }
  std::string CompileCacheDepFilesHash() { return compile_cache_dep_files_hash_; }

  void set_has_cached_queue_name(bool cached) { has_cached_queue_name_ = cached; }
  bool has_cached_queue_name() const { return has_cached_queue_name_; }

  void set_init_compile_cache(const bool &init) { init_compile_cache_ = init; }
  bool init_compile_cache() const { return init_compile_cache_; }

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
  // key is parent funcgarph, values is child funcgraph.
  std::vector<FuncGraphPtr> child_graphs_;
  HashMap<FuncGraphPtr, FuncGraph *> backend_graph_to_frontend_graph_;
  HashMap<FuncGraphPtr, std::string> front_graph_to_backend_graph_cache_path_;
  std::map<std::string, CachedIOSizeInfo> fullname_io_size;
  // param is a backend node but we can find its correspond frontend param.
  mindspore::HashSet<AnfNodePtr> backend_param_gen_from_frontend_param_;
  bool restricted_scenarios_{false};
  std::string compile_cache_dep_files_hash_ = "";
  bool has_cached_queue_name_{false};
  bool init_compile_cache_{false};
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_COMPILE_CACHE_CONTEXT_H_
