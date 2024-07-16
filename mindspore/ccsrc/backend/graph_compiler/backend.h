/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_VM_BACKEND_H_
#define MINDSPORE_CCSRC_VM_BACKEND_H_

#include <list>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "include/common/utils/contract.h"
#include "ir/anf.h"
#include "backend/graph_compiler/backend_base.h"
#include "backend/graph_compiler/segment_runner.h"
#include "backend/graph_compiler/graph_partition.h"
#include "backend/graph_compiler/vm.h"
#include "backend/graph_compiler/op_backend.h"
#include "backend/common/session/session_basic.h"
#include "runtime/hardware/device_context.h"
#include "runtime/graph_scheduler/graph_scheduler.h"
#include "runtime/pipeline/task/device_task.h"
#include "runtime/pynative/op_compiler.h"
#include "runtime/pynative/graph_adapter.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace compile {
class BACKEND_EXPORT MsBackend : public Backend {
 public:
  MsBackend(const std::string &name, const std::string &target, uint32_t device_id);
  ~MsBackend() override = default;

  LinConvertResult MsConvert(const GraphSegmentPtr &segment, const std::string &target = "");
  virtual VectorRef MsRunGraph(const GraphId &g, const VectorRef &args, const std::string &target = "");

  VectorRef MsSimuRunGraph(const GraphId &g);
  GraphId CompileGraph(const NotNull<FuncGraphPtr> &fg) override;
  VectorRef RunGraph(GraphId graph_id, const VectorRef &args);
  void ClearSessionGraphs();
  void CreateOtherSession(const std::string &target);

#ifdef ENABLE_DEBUGGER
  void SetDebugger() override;
#endif

 protected:
  session::SessionPtr target_sess_;
  session::SessionPtr other_sess_;
  std::string target_device_;
  std::string other_device_;
  mindspore::HashMap<GraphId, LinConvertResult> graph_id_map_;
};

class BACKEND_EXPORT MindRTBackend : public MindRTBackendBase {
 public:
  MindRTBackend(const std::string &backend_name, const std::string &device_name, uint32_t device_id)
      : MindRTBackendBase(backend_name, device_name, device_id) {}
  ~MindRTBackend() override = default;

  // Execute all tasks in queue when lazy build is enabled in PyNative mode.
  void WaitTaskFinish() const override;
  // Clear resource when python exit.
  void ClearOpExecutorResource() const;

  // Sync default stream in PyNative mode.
  void SyncStream();

  KernelGraphPtr GetGraphById(GraphId graph_id);

 private:
  void RunGraphByCondition(const ActorInfo &actor_info, const GraphCompilerInfo &graph_compiler_info,
                           const VectorRef &args, VectorRef *outputs) override;
  // Split complete kernel graph to single op graph in PyNative back
  // propagation, then compile and run single op graph or pyboost op(if op registered).
  void RunGraphBySingleOp(const GraphCompilerInfo &graph_compiler_info, const VectorRef &args, VectorRef *outputs);

  runtime::ActorSet *RealCompileGraphBeforeRunActor(const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                                                    bool no_multi_graph);
  void RunGraphByActors(const ActorInfo &actor_info, const GraphCompilerInfo &graph_compiler_info,
                        const VectorRef &args, VectorRef *outputs);

  void RunMsGradGraph(const CNodePtr &kernel, const VectorRef &args, VectorRef *outputs) const;

  // Clean the compilation cache to avoid memory leakage in dynamic shape scenarios.
  void ClearResource();

  // Cache output tensor ref count of kernels for back propagation graph in PyNative mode.
  std::map<GraphId, std::map<KernelWithIndex, size_t>> cnode_ref_counts_;

  // Cache forward op output value node tensor ref count of kernels for back propagation graph in PyNative mode.
  std::map<std::string, size_t> forward_op_output_tensor_id_;

  OpBackend op_backend_;
  pynative::GraphAdapter graph_adapter_;
};
using MindRTBackendPtr = std::shared_ptr<compile::MindRTBackend>;
}  // namespace compile
using BackendOpRunInfoPtr = std::shared_ptr<session::BackendOpRunInfo>;
}  // namespace mindspore
#endif
