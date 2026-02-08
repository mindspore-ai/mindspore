/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_BACKEND_MANAGER_BACKEND_MANAGER_H_
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_BACKEND_MANAGER_BACKEND_MANAGER_H_

#include <memory>
#include <string>
#include <utility>
#include <map>
#include <mutex>
#include <vector>
#include "include/backend/backend_manager/backend_base.h"

namespace mindspore {
namespace backend {
using BackendCreator = std::function<std::shared_ptr<BackendBase>()>;
using BackendName = std::string;

// The backend name must be equal to the backend field of api "mindspore.jit".
const char kMSBackendName[] = "ms_backend";
const char kGEBackendName[] = "GE";

// The name of backend lib.
const char kGEBackendLibName[] = "libmindspore_ge_backend.so";

// The backend type enum, please add a new enumeration definition before kInvalidBackend when adding a new backend.
enum BackendType {
  kMSBackend = 0,       // 0 for ms_backend
  kGEBackend,           // 1 for GE
  kCustomBackend = 11,  // 2~11 for custom backend, support up to 10 custom backend
  kInvalidBackend,      // number of backend
};

class BACKEND_MANAGER_EXPORT BackendManager {
 public:
  static BackendManager &GetInstance();
  // Record the BackendCreator by the backend name.
  void Register(const std::string &backend_name, BackendCreator &&backend_creator);

  std::vector<GraphFragmentPtr> Split(const FuncGraphPtr &func_graph, const std::string &backend_name);

  // The processing entry of graph building by the given backend.
  // The return value are the selected backend type and the built graph id.
  std::pair<BackendType, BackendGraphId> Build(const FuncGraphPtr &func_graph,
                                               const BackendJitConfig &backend_jit_config,
                                               const std::string &backend_name = "");

  // The processing entry of graph running by the backend_type and graph_id
  // which are generated through the graph Build interface above.
  RunningStatus Run(BackendType backend_type, BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs);

  // Load backend plugin for GE backend or custom backend.
  bool LoadBackend(const BackendName &backend_name, const std::string &backend_path = "");

  // Clear the members.
  void Clear();

  // Clear the Build Graph.
  void ClearGraph(BackendType backend_type, BackendGraphId backend_graph_id);

  // convert mindir to ir_format
  void ConvertIR(const FuncGraphPtr &anf_graph,
                 const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors, IRFormat ir_format,
                 const std::string &backend_name = "");

  // export graph to ir_format. If is_save_to_file=True, save as file; if False, return as string
  string ExportIR(const FuncGraphPtr &anf_graph, const std::string &file_name, bool is_save_to_file, IRFormat ir_format,
                  const std::string &backend_name = "");

 private:
  BackendManager() = default;
  ~BackendManager() = default;

  void UnloadBackend();

  BackendBase *GetOrCreateBackend(const BackendType &backend_type);

  // BackendType -> BackendLoadHandle.
  std::map<BackendType, void *> backend_load_handle_;

  // BackendType -> BackendCreator.
  std::map<BackendType, BackendCreator> backend_creators_;

  // BackendType -> BackendBase.
  BackendBasePtr backends_[kInvalidBackend];

  std::mutex backend_mutex_;
};

class BackendRegister {
 public:
  BackendRegister(const std::string &backend_name, BackendCreator &&backend_creator) {
    BackendManager::GetInstance().Register(backend_name, std::move(backend_creator));
  }
  ~BackendRegister() = default;
};
}  // namespace backend
}  // namespace mindspore

// The register entry of new backend.
#define MS_REGISTER_BACKEND(BACKEND_NAME, BACKEND_CLASS)                           \
  static const mindspore::backend::BackendRegister g_backend_##BACKEND_NAME##_reg( \
    BACKEND_NAME, []() { return std::make_shared<BACKEND_CLASS>(); })
#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_BACKEND_MANAGER_BACKEND_MANAGER_H_
