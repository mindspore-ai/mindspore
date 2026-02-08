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
#ifndef MINDSPORE_CCSRC_BACKEND_MS_BACKEND_MSBACKEND_H_
#define MINDSPORE_CCSRC_BACKEND_MS_BACKEND_MSBACKEND_H_

#include <list>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "include/utils/contract.h"
#include "ir/anf.h"
#include "base/base_ref.h"
#include "backend/ms_backend/graph_partition.h"
#include "include/backend/common/kernel_graph/session_basic.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_scheduler.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_adapter.h"
#include "backend/ms_backend/ms_backend_base.h"
namespace mindspore {
namespace backend {
namespace ms_backend {
class MSBackend : public MSBackendBase {
 public:
  MSBackend() : MSBackendBase() {}
  ~MSBackend() override;

  // Execute all tasks in queue when lazy build is enabled in PyNative mode.
  void WaitTaskFinish() const override;

  // Sync default stream in PyNative mode.
  void SyncStream();

  KernelGraphPtr GetGraphById(GraphId graph_id);

 private:
  void RunGraphByCondition(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                           VectorRef *outputs) override;

  void ProcessBeforeRunActor(const GraphCompilerInfo &graph_compiler_info, const VectorRef &args);
  void RunGraphByActors(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                        VectorRef *outputs);

  void RunActorSet(BackendGraphId graph_id, runtime::ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info,
                   const VectorRef &args, bool no_multi_graph, VectorRef *outputs);

  pynative::GraphAdapter graph_adapter_;
};
}  // namespace ms_backend
}  // namespace backend
}  // namespace mindspore
#endif
