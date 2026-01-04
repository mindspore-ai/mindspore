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
#include "backend/ms_backend/ms_backend.h"

#include <algorithm>
#include <vector>
#include <map>
#include <stack>
#include <unordered_map>
#include "primitive/sequence_ops.h"
#include "primitive/nn_op_name.h"
#include "primitive/structure_op_name.h"
#include "include/utils/parallel_context.h"
#include "include/backend/common/kernel_graph/session_factory.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/frontend/jit/ps/action_interface.h"
#include "backend/common/kernel_graph/jit_call_graph.h"
#include "ir/anf.h"
#include "mindspore/ccsrc/utils/base_ref_py.h"
#include "pybind_api/pybind_patch.h"
#include "include/utils/callbacks.h"
#include "utils/convert_utils.h"
#include "include/utils/convert_utils_py.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_compiler.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_adapter.h"
#include "backend/ms_backend/runtime/actors/base/actor_common.h"
#include "pybind_api/gil_scoped_long_running.h"

#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/cluster/topology/ps_context.h"
#endif

#include "include/backend/common/device_address_utils.h"
#include "include/backend/common/pass_manager/dynamic_shape_helper.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/runtime/pipeline/task/run_graph_task.h"
#include "include/utils/stub_tensor.h"

namespace mindspore {
namespace backend {
namespace ms_backend {
MSBackend::~MSBackend() {
  GilReleaseWithCheck gil_release;
  runtime::Pipeline::Get().frontend_stage()->Wait();
}

void MSBackend::ProcessBeforeRunActor(const GraphCompilerInfo &graph_compiler_info, const VectorRef &args) {
  WaitTaskFinish();
  WaitMultiStream(graph_compiler_info);
  CreateTensorArgs(args, graph_compiler_info);
  WaitTaskFinish();
  auto graphs = graph_compiler_info.graphs_;

  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->is_any_type_input()) {
      continue;
    }
    auto input_tensors = GetRunGraphInputs(graph_compiler_info, args);
    if (graph_compiler_info.enable_graph_pipeline_) {
      for (const auto &tensors : input_tensors) {
        for (const auto &tensor : tensors) {
          if (tensor) {
            tensor->set_need_pipeline_sync(true);
          }
        }
      }
    }
  }
}

void MSBackend::RunGraphByActors(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info,
                                 const VectorRef &args, VectorRef *outputs) {
  MS_LOG(INFO) << "Status record: begin run actor: " << graph_id;
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  auto graphs = graph_compiler_info.graphs_;

  // KernelByKernel: The size of control_nodes is at least 1 since there is return node in the graph.
  // GraphMode: No control nodes.
  bool no_multi_graph = graph_compiler_info.control_nodes_.size() <= 1 && graphs.size() == 1;
  ProcessBeforeRunActor(graph_compiler_info, args);
  auto actor_set = runtime::GraphScheduler::GetInstance().Fetch(graph_id);
  MS_EXCEPTION_IF_NULL(actor_set);

  if (graph_compiler_info.enable_graph_pipeline_) {
    // 1. Construct stub output.
    auto output_node = graph_compiler_info.root_func_graph_->output();
    MS_EXCEPTION_IF_NULL(output_node);
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kOutputProcess,
                                       "MakeStubNode");
    auto stub_output_pair = stub::MakeStubNode(output_node->abstract());
    if (stub_output_pair.second) {
      MS_LOG(DEBUG) << "Enable pynative graph pipeline for actor set: " << graph_id;
      // 2. Async run graph.
      auto &stub_output = stub_output_pair.first;
      MS_EXCEPTION_IF_NULL(stub_output);
      outputs->push_back(stub_output);

      auto run_graph_task = std::make_shared<runtime::RunGraphTask>(
        [=, &graph_compiler_info]() {
          actor_set->output_actor_->SetStubOutput(stub_output);
          RunActorSet(graph_id, actor_set, graph_compiler_info, args, no_multi_graph, outputs);
        },
        stub_output);
      GilReleaseWithCheck release_gil;
      runtime::Pipeline::Get().frontend_stage()->Push(run_graph_task);
      return;
    }
    graph_compiler_info.enable_graph_pipeline_ = false;
    MS_LOG(INFO)
      << "Failed to create Stub output, encountered an unsupported output type for graph: " << graph_id
      << ". Currently, only output types that include: Tensor, Scalar, String, fixed-length Sequence, are "
         "supported. The single op and graph pipeline has been disabled, so the performance will not be improved.";
  }

  RunActorSet(graph_id, actor_set, graph_compiler_info, args, no_multi_graph, outputs);
}

void MSBackend::RunActorSet(BackendGraphId graph_id, runtime::ActorSet *actor_set,
                            const GraphCompilerInfo &graph_compiler_info, const VectorRef &args, bool no_multi_graph,
                            VectorRef *outputs) {
  WaitTaskFinish();
  WaitMultiStream(graph_compiler_info);
  WaitTaskFinish();

  auto graphs = graph_compiler_info.graphs_;
  auto &device_contexts = graph_compiler_info.device_contexts_;
  if (graph_compiler_info.root_func_graph_->has_flag(kFlagIsPynativeBpropGraph)) {
    for (size_t i = 0; i < graphs.size(); ++i) {
      graph_adapter_.UpdateForwardOutputInBpropGraph(graphs[i], device_contexts[i], no_multi_graph);
    }
  }

  std::vector<std::vector<tensor::TensorPtr>> input_tensors;
  // make sure enable input optimize condition right.
  MS_LOG(INFO) << "Start to run graph, args size: " << args.size() << ", graph: " << actor_set->name_;
  runtime::ActorDispatcher::set_enable_sub_graph_execute_for_cur_actor_set(actor_set->enable_kbk_sub_graph_execute_);
  runtime::ActorDispatcher::set_enable_input_optimize_for_cur_actor_set(actor_set->enable_input_optimize_);
  if (!runtime::EnableInputOptimize() && !runtime::EnableParallelDispatchKernel()) {
    input_tensors = GetRunGraphInputs(graph_compiler_info, args);
    if (graphs.size() > input_tensors.size()) {
      MS_LOG(EXCEPTION) << "The actor_set " << actor_set->name_ << " graphs size " << graphs.size()
                        << " should less than or equal to inputs size " << input_tensors.size();
    }
    pynative::GraphAdapter::HandleHeterogeneousTensors(input_tensors, device_contexts, actor_set);
    // Release GIL and run actor DAG.
    GilReleaseWithCheck release_gil;
    VectorRef empty_args;
    runtime::GraphScheduler::GetInstance().Run(actor_set, input_tensors, empty_args);
  } else {
    GilReleaseWithCheck release_gil;
    runtime::GraphScheduler::GetInstance().Run(actor_set, input_tensors, args);
  }

  MS_EXCEPTION_IF_NULL(graph_compiler_);
  graph_compiler_->Summary(graph_compiler_info.graphs_);

  auto output = graph_compiler_info.root_func_graph_->output();
  if (output != nullptr) {
    MS_LOG(DEBUG) << "Current out " << output->DebugString();
  }
  ConstructOutputs(actor_set, outputs, graph_compiler_info.root_func_graph_,
                   graph_compiler_info.enable_graph_pipeline_);
  actor_set->output_actor_->FreeSummaryNodeMem();
  runtime::GraphScheduler::GetInstance().ClearActorData(actor_set);
  // Close abstract_lock for dynamic_shape
  AnfUtils::CloseAbstractLock();
  MS_LOG(INFO) << "Status record: end run actor: " << graph_id;
}

void MSBackend::RunGraphByCondition(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info,
                                    const VectorRef &args, VectorRef *outputs) {
  RunGraphByActors(graph_id, graph_compiler_info, args, outputs);
}

void MSBackend::WaitTaskFinish() const {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kWaitTaskFinish,
                                     runtime::kDefaultOpName);
  runtime::Pipeline::Get().WaitAll();
}

void MSBackend::SyncStream() {
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device::GetDeviceTypeByName(device_name_), device_id_});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

  auto ret = device_context->device_res_manager_->SyncAllStreams();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync Stream failed";
  }
}

KernelGraphPtr MSBackend::GetGraphById(GraphId graph_id) {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  return graph_compiler_->Fetch(graph_id);
}

MS_REGISTER_BACKEND(kMSBackendName, MSBackend);
}  // namespace ms_backend
}  // namespace backend
}  // namespace mindspore
