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
#include <string>
#include <memory>
#include <vector>
#include "mindspore/include/custom_backend_api.h"

namespace mindspore {
namespace backend {
constexpr auto kCustomBackendName = "my_custom_backend";

// Use the built-in ms_backend to test the custom backend.
class MSCustomBackendBase : public BackendBase {
 public:
  BackendGraphId Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
    MS_LOG(WARNING) << "MSCustomBackendBase use the origin ms_backend to build the graph.";
    mindspore::backend::BackendManager::GetInstance().Build(func_graph, backend_jit_config, "ms_backend");
  }

  // The backend graph Run interface by the graph_id which are generated through the graph Build interface above.
  RunningStatus Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) {
    MS_LOG(WARNING) << "MSCustomBackendBase use the origin ms_backend to run the graph.";
    mindspore::backend::BackendManager::GetInstance().Run(BackendType::kMSBackend, graph_id, inputs, outputs);
  }
};
MS_REGISTER_BACKEND(kCustomBackendName, MSCustomBackendBase);
}  // namespace backend
}  // namespace mindspore
