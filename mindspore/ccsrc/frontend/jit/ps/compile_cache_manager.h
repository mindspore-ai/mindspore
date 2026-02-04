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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_COMPILE_CACHE_MANAGER_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_COMPILE_CACHE_MANAGER_H_

#include <string>
#include <memory>
#include "pybind11/pybind11.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "ir/func_graph.h"
#include "load_mindir/load_model.h"

namespace mindspore {
namespace pipeline {
namespace py = pybind11;

// A class for loading and caching the func_graph.
class CompileCacheManager {
 public:
  explicit CompileCacheManager(size_t compile_cache_id) : compile_cache_id_(compile_cache_id) {}
  CompileCacheManager() = default;

  ~CompileCacheManager() = default;

  // Get cached data queue name
  static std::string GetCachedDataQueueName(const std::string &dataset_phase);
  // Get the hash of dependent files when compiling graph.
  void InitCompileCacheHash(const py::list &compile_cache_dep_files);
  // Init group checkpoint file path for parallel mode.
  void InitParallelGroupCkptSaveFile();
  // Compare the dependency files hash.
  bool CanLoadCache();
  // Load the cached func_graph from mindir file.
  FuncGraphPtr GetCachedFuncGraph(const FuncGraphManagerPtr &manager, const py::dict &weights,
                                  const std::string &queue_name, bool backward_graph = false);
  // Export the func_graph to mindir file.
  void CacheFuncGraph(const FuncGraphPtr &fg, const FuncGraphPtr &layout_fg, bool cache_hash = true,
                      bool backward_graph = false);

  const LayoutMap &layout_map() const { return layout_map_; }

  void SetCompileCacheDir(const std::string &dir) { compile_cache_dir_ = dir; }
  std::string CompileCacheDir() const { return compile_cache_dir_; }
  void set_id_extension(const std::string &id_extension) { id_extension_ = id_extension; }
  static size_t data_queue_num_;

 private:
  size_t compile_cache_id_;
  std::string compile_cache_dep_files_hash_;
  LayoutMap layout_map_;
  std::string compile_cache_dir_;
  std::string id_extension_;
};
using CompileCacheManagerPtr = std::shared_ptr<CompileCacheManager>;
}  // namespace pipeline
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_COMPILE_CACHE_MANAGER_H_
