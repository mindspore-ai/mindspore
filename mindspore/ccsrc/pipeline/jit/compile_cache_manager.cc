/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/compile_cache_manager.h"
#include <vector>
#include <algorithm>
#include <map>
#include <utility>
#include <fstream>
#include "pipeline/jit/parse/data_converter.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/debug/common.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "utils/system/sha256.h"
#include "include/common/utils/utils.h"
#include "frontend/parallel/step_parallel.h"
#include "mindspore/core/utils/file_utils.h"

#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/cluster/cluster_context.h"
#include "include/backend/distributed/ps/ps_context.h"
#endif
#include "include/common/utils/compile_cache_context.h"

namespace mindspore {
#ifndef MINDIR_EXPORT_TENSOR_LAYOUT_CLIP
void BuildLayout(const FuncGraphPtr &func_graph, mind_ir::ModelProto *model) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(model);
  std::vector<AnfNodePtr> graph_params = func_graph->parameters();
  mind_ir::ParallelProto *parallel_proto = model->mutable_parallel();
  for (auto para : graph_params) {
    std::string name = std::static_pointer_cast<Parameter>(para)->name();
    auto tensor_layout = para->user_data<parallel::TensorLayout>();
    if (tensor_layout == nullptr) {
      MS_LOG(INFO) << "GetParameterLayout nullptr name = " << name;
    } else {
      mind_ir::LayoutProto *layoutProto = parallel_proto->add_layout();

      // Get all the information for layput
      auto device_arrangement = tensor_layout->device_arrangement().array();
      auto tensor_map = tensor_layout->tensor_map().array();
      auto slice_shape = tensor_layout->slice_shape().array();
      int64_t field_size = tensor_layout->get_field_size();
      bool uniform_split = tensor_layout->uniform_split();
      std::string opt_shard_group = tensor_layout->opt_shard_group();
      if (!opt_shard_group.empty()) {
        slice_shape = tensor_layout->opt_shard_slice_shape();
      }
      // Save all information to Layout Proto
      layoutProto->set_name(name);
      for (auto device_arrangement_element : device_arrangement) {
        layoutProto->add_device_arrangement_int(device_arrangement_element);
      }
      for (auto tensor_map_element : tensor_map) {
        layoutProto->add_tensor_map_int(tensor_map_element);
      }
      for (auto slice_shape_element : slice_shape) {
        layoutProto->add_slice_shape_int(slice_shape_element);
      }
      layoutProto->set_field_size(field_size);
      layoutProto->set_uniform_split(uniform_split);
      layoutProto->set_opt_shard_group(opt_shard_group);
    }
  }
}
#endif
namespace pipeline {
namespace {
std::string GetCompileCacheDir() {
  static const std::string user_defined_path = Common::GetUserDefineCachePath();
  static const uint32_t rank_id = IsStandAlone() ? 0 : GetRank();
  static const std::string compile_cache_dir = user_defined_path + "rank_" + std::to_string(rank_id);
  return compile_cache_dir;
}

std::string GetGraphCacheDir() { return GetCompileCacheDir() + "/" + kGraphCacheSubDir; }

std::string GetRole() {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (distributed::cluster::ClusterContext::instance()->initialized()) {
    auto node = distributed::cluster::ClusterContext::instance()->node();
    MS_EXCEPTION_IF_NULL(node);
    const auto &cluster_ctx = distributed::cluster::ClusterContext::instance();
    MS_EXCEPTION_IF_NULL(cluster_ctx);
    MS_LOG(INFO) << "Cluster is initialized. This node role is " << cluster_ctx->node_role();
    return cluster_ctx->node_role();
  }
#endif
  return "";
}

std::string GetCompileCachePath(size_t idx) {
  return GetGraphCacheDir() + "/" + GetRole() + kCompileCacheFileName + "_" + std::to_string(idx) + kMindIrSuffix;
}

std::string GetBackendCompileCachePathWithoutExtension(size_t idx) {
  return GetGraphCacheDir() + "/" + GetRole() + kBackendCompileCacheFileName + "_" + std::to_string(idx);
}

std::string GetDepFilesHashPath() {
  static const std::string dep_files_hash_path = GetGraphCacheDir() + "/" + GetRole() + kDepFilesHashPath;
  return dep_files_hash_path;
}

std::string GetGroupCkptSavePath() { return GetGraphCacheDir() + "/" + kGroupCkptFileName; }

std::string GetCompileDepFilesHash(const py::list &dep_files) {
  MS_LOG(DEBUG) << "Dependency files size: " << dep_files.size();
  std::vector<std::string> dep_files_path;
  for (auto dep_file : dep_files) {
    auto file_path = py::cast<std::string>(dep_file);
    MS_LOG(DEBUG) << "Dependency file path: " << file_path;
    (void)dep_files_path.emplace_back(file_path);
  }
  std::sort(dep_files_path.begin(), dep_files_path.end());
  std::string files_hash;
  for (const auto &path : dep_files_path) {
    std::string file_hash = system::sha256::GetHashFromFile(path);
    files_hash += file_hash;
  }
  return files_hash;
}

std::map<string, ValuePtr> GenerateWeightsValueMap(const py::dict &weights) {
  std::map<string, ValuePtr> ret{};
  for (auto weight = weights.begin(); weight != weights.end(); ++weight) {
    auto weight_name = py::cast<std::string>(weight->first);
    auto weight_value = parse::data_converter::PyDataToValue(py::cast<py::object>(weight->second));
    ret[weight_name] = weight_value;
  }
  return ret;
}

std::pair<FuncGraphPtr, LayoutMap> LoadFuncGraphFromMindIR(const py::dict &weights, bool has_parallel_info,
                                                           size_t idx) {
  LayoutMap layout_map;
  std::string compile_cache_path = GetCompileCachePath(idx);
  auto realpath = Common::CreatePrefixPath(compile_cache_path, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path of file " << compile_cache_path << " failed.";
    return std::make_pair(nullptr, layout_map);
  }
  std::ifstream f(realpath.value());
  bool file_is_good = f.good();
  f.close();
  if (!file_is_good) {
    MS_LOG(WARNING) << "Open the compilation cache file " << realpath.value() << " failed.";
    return std::make_pair(nullptr, layout_map);
  }
  MindIRLoader mindir_loader;
  mindir_loader.set_weights_value_map(GenerateWeightsValueMap(weights));
  mindir_loader.set_has_parallel_info(has_parallel_info);
  mindspore::HashMap<std::string, AnfNodePtr> name_to_node;
  auto fg = mindir_loader.LoadMindIR(realpath.value(), &name_to_node);
  auto &context = CompileCacheContext::GetInstance();
  context.SetFrontNameToFrontNode(name_to_node);
  context.SetFrontGraph(fg);
  context.InsertBackendGraphCachePath(fg, GetBackendCompileCachePathWithoutExtension(idx));
#if defined(__linux__) && defined(WITH_BACKEND)
  // compile cache does not support PS or cluster.
  if (ps::PSContext::instance()->is_ps_mode() || distributed::cluster::ClusterContext::instance()->initialized()) {
    context.SetPsOrClusterMode(true);
  }
#endif
  return std::make_pair(fg, mindir_loader.layout_map());
}

bool ExportFuncGraphToMindIR(const FuncGraphPtr &fg, const FuncGraphPtr &layout_fg, size_t idx) {
  std::string compile_cache_path = GetCompileCachePath(idx);
  auto proto = GenBinaryProto(fg);
  if (proto == nullptr) {
    MS_LOG(ERROR) << "Get binary proto for graph " << fg->ToString() << " failed.";
    return false;
  }
#ifndef MINDIR_EXPORT_TENSOR_LAYOUT_CLIP
  if (layout_fg) {
    BuildLayout(layout_fg, proto.get());
  }
#endif
  auto &context = CompileCacheContext::GetInstance();
  context.SetFrontGraph(fg);
  context.InsertBackendGraphCachePath(fg, GetBackendCompileCachePathWithoutExtension(idx));
#if defined(__linux__) && defined(WITH_BACKEND)
  // compile cache does not support PS or cluster.
  if (ps::PSContext::instance()->is_ps_mode() || distributed::cluster::ClusterContext::instance()->initialized()) {
    context.SetPsOrClusterMode(true);
  }
#endif
  MindIRExporter mindir_exporter;
  return mindir_exporter.SaveProtoToFile(proto.get(), compile_cache_path);
}

bool ExportDepFilesHash(const std::string &compile_cache_dep_files_hash) {
  std::string dep_files_hash_path = GetDepFilesHashPath();
  auto realpath = Common::CreatePrefixPath(dep_files_hash_path, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path of file " << dep_files_hash_path << " failed.";
    return false;
  }

  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream fout(realpath.value());
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open cache file '" << realpath.value() << "' failed!" << ErrnoToString(errno);
    return false;
  }
  fout << compile_cache_dep_files_hash;
  fout.close();
  ChangeFileMode(realpath.value(), S_IRUSR);
  return true;
}

bool CreateParallelGroupsByCkptFile() {
  static const std::string group_ckpt_save_path = GetGroupCkptSavePath();
  auto realpath = Common::CreatePrefixPath(group_ckpt_save_path, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path of file " << group_ckpt_save_path << " failed.";
    return false;
  }
  std::ifstream f(realpath.value());
  bool file_is_good = f.good();
  f.close();
  if (!file_is_good) {
    MS_LOG(ERROR) << "Open the group checkpoint file " << realpath.value() << " failed.";
    return false;
  }
  return parallel::CreateGroupsByCkptFile(group_ckpt_save_path);
}
}  // namespace

void CompileCacheManager::CacheFuncGraph(const FuncGraphPtr &fg, const FuncGraphPtr &layout_fg) {
  if (fg == nullptr) {
    MS_LOG(ERROR) << "The func_graph to be cached is null.";
    return;
  }

  SetCompileCacheDir(GetCompileCacheDir());

  if (!ExportFuncGraphToMindIR(fg, layout_fg, compile_cache_id_)) {
    MS_LOG(ERROR) << "Failed to cache graph: " << fg->ToString();
    return;
  }
  if (compile_cache_id_ == 0 && !ExportDepFilesHash(compile_cache_dep_files_hash_)) {
    MS_LOG(ERROR) << "Failed to cache the dependency files hash";
  }
}

void CompileCacheManager::InitCompileCacheHash(const py::list &compile_cache_dep_files) {
  compile_cache_dep_files_hash_ = GetCompileDepFilesHash(compile_cache_dep_files);
}

bool CompileCacheManager::CheckDepFilesHashConsistency() {
  if (compile_cache_dep_files_hash_.empty()) {
    MS_LOG(ERROR) << "Get current dependency files hash failed.";
    return false;
  }
  std::string dep_files_hash_path = GetDepFilesHashPath();
  auto realpath = Common::CreatePrefixPath(dep_files_hash_path, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path of file " << dep_files_hash_path << " failed.";
    return false;
  }
  std::fstream input(realpath.value(), std::ios::in | std::ios::binary);
  if (!input) {
    MS_LOG(WARNING) << "Open the hash file " << realpath.value() << " failed. The file may not exist."
                    << ErrnoToString(errno);
    return false;
  }
  std::string checkpoint_hash;
  input >> checkpoint_hash;
  if (checkpoint_hash.empty()) {
    MS_LOG(ERROR) << "Get the compilation dependency files hash from " << realpath.value() << " failed.";
    return false;
  }
  if (checkpoint_hash != compile_cache_dep_files_hash_) {
    MS_LOG(WARNING) << "The compilation dependency files are changed.";
    return false;
  }
  return true;
}

FuncGraphPtr CompileCacheManager::GetCachedFuncGraph(const FuncGraphManagerPtr &manager, const py::dict &weights,
                                                     const std::string &queue_name) {
  // Determine whether to load parallel information.
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  bool has_parallel_info = false;
  if ((parallel_mode == parallel::kAutoParallel) || (parallel_mode == parallel::kSemiAutoParallel)) {
    if (!CreateParallelGroupsByCkptFile()) {
      MS_LOG(WARNING) << "Failed to create the parallel groups info. Execute all the compilation actions.";
      return nullptr;
    }
    has_parallel_info = true;
  }
  // Load the compilation cache file.
  auto pair = LoadFuncGraphFromMindIR(weights, has_parallel_info, compile_cache_id_);
  if (pair.first == nullptr) {
    MS_LOG(WARNING) << "Failed to load the compilation cache file. Execute all the compilation actions.";
    return nullptr;
  }
  auto fg = pair.first;
  layout_map_ = pair.second;

  MS_LOG(WARNING) << "Use the compilation cache and execute the backend actions only. Be aware of correctness risks.";
  FuncGraphManagerPtr mng = fg->manager();
  if (mng == nullptr) {
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(fg);
    fg->set_manager(manager);
  }
  // The value of attr "shared_name" will changed every time.
  auto cnodes = fg->GetOrderedCnodes();
  for (const auto &cnode : cnodes) {
    auto prim = GetValuePtr<Primitive>(cnode->input(0));
    if (prim != nullptr && prim->HasAttr("shared_name")) {
      prim->set_attr("shared_name", MakeValue(queue_name));
      break;
    }
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("cache_loaded_graph_" + std::to_string(compile_cache_id_) + ".ir", fg);
  }
#endif
  return fg;
}

void CompileCacheManager::InitParallelGroupCkptSaveFile() {
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if ((parallel_mode == parallel::kAutoParallel) || (parallel_mode == parallel::kSemiAutoParallel)) {
    parallel::ParallelContext::GetInstance()->set_group_ckpt_save_file(GetGroupCkptSavePath());
  }
}
}  // namespace pipeline
}  // namespace mindspore
