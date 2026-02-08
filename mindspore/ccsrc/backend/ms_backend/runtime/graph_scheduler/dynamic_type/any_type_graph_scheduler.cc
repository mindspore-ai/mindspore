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

#include "backend/ms_backend/runtime/graph_scheduler/dynamic_type/any_type_graph_scheduler.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_scheduler.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/scheduler_helper.h"

namespace mindspore {
namespace runtime {
std::vector<AnyTypeKernelActorPtr> AnyTypeGraphScheduler::Build(const GraphCompilerInfo &graph_compiler_info,
                                                                const AID &memory_manager_aid, const AID *debug_id) {
  std::vector<AnyTypeKernelActorPtr> any_type_kernel_actors;
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    if (!graph->is_any_type_input()) {
      continue;
    }

    const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
    for (const auto &output : outputs) {
      if (output.second == 0 || output.first == nullptr || output.first->abstract() == nullptr ||
          !output.first->abstract()->isa<abstract::AbstractSequence>()) {
        continue;
      }
      const auto &seq_abs = output.first->abstract()->cast<abstract::AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(seq_abs);
      if (seq_abs->dynamic_len()) {
        MS_LOG_WITH_NODE(EXCEPTION, output.first)
          << "Unsupported output: tuple output with dynamic len is not supported in JIT fallback, graph id:"
          << graph->graph_id() << ".";
      }
    }

    if (graph->execution_order().empty()) {
      MS_LOG(INFO) << "The graph " << graph->graph_id() << " is an empty graph and skips building.";
      continue;
    }

    auto actor_name = graph->ToString() + kAnyTypeKernelActorNameSuffix;
    auto any_type_kernel_actor =
      std::make_shared<AnyTypeKernelActor>(actor_name, graph, device_context, memory_manager_aid, debug_id, nullptr);
    MS_EXCEPTION_IF_NULL(any_type_kernel_actor);
    any_type_kernel_actor->compile_func_ = graph_compiler_info.compile_func_;
    InsertActor(any_type_kernel_actor.get());
    (void)any_type_kernel_actors.emplace_back(any_type_kernel_actor);
  }
  return any_type_kernel_actors;
}

void AnyTypeGraphScheduler::Optimize(const ActorSetPtr &actor_set,
                                     const std::map<KernelWithIndex, std::pair<AbstractActor *, KernelWithIndex>,
                                                    session::KernelWithIndexCmp> &graph_output_to_actor) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  for (const auto &any_type_kernel_actor : actor_set->any_type_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(any_type_kernel_actor);
    MS_EXCEPTION_IF_NULL(any_type_kernel_actor->graph());
    for (const auto &input_node : any_type_kernel_actor->graph()->input_nodes()) {
      MS_EXCEPTION_IF_NULL(input_node);
      auto front_node_with_index = any_type_kernel_actor->graph()->GetOriginFrontNodeByInternalParameter(input_node);
      auto front_node = front_node_with_index.first;
      // Update device tensor store key in any type kernel actor for value tuple node.
      if (front_node == nullptr || !front_node->isa<ValueNode>()) {
        continue;
      }
      const auto &value_node = front_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      const auto &value = value_node->value();
      if (value == nullptr || (!value->isa<ValueSequence>())) {
        continue;
      }
      auto iter = std::find_if(any_type_kernel_actor->device_tensor_store_keys_.begin(),
                               any_type_kernel_actor->device_tensor_store_keys_.end(),
                               [value_node](const auto &node_pair) { return node_pair.second == value_node; });
      if (iter == any_type_kernel_actor->device_tensor_store_keys_.end() || iter->second == nullptr) {
        continue;
      }
      MS_LOG(DEBUG) << "Prepare fix device tensor store key:" << iter->second->DebugString()
                    << " input index:" << iter->first
                    << " in any type kernel actor:" << any_type_kernel_actor->GetAID();
      const auto &output_pair_iter = graph_output_to_actor.find(front_node_with_index);
      if (output_pair_iter == graph_output_to_actor.end()) {
        MS_LOG(DEBUG) << "Failed to get device tensor store key:" << iter->second->DebugString()
                      << " input index:" << iter->first
                      << " in graph output any type kernel actor:" << any_type_kernel_actor->GetAID();
        continue;
      }
      const auto &backend_node_with_index = output_pair_iter->second.second;
      if (backend_node_with_index.first == nullptr || !backend_node_with_index.first->isa<ValueNode>()) {
        MS_LOG(WARNING) << "Failed to get backend by device tensor store key:" << iter->second->DebugString()
                        << " input index:" << iter->first
                        << " in graph output any type kernel actor:" << any_type_kernel_actor->GetAID();
        continue;
      }
      if (any_type_kernel_actor->device_contexts_.empty() || any_type_kernel_actor->device_contexts_[0] == nullptr) {
        MS_LOG(WARNING) << "Failed to get device context in any type kernel actor:" << any_type_kernel_actor->GetAID();
        continue;
      }
      const auto &device_context = any_type_kernel_actor->device_contexts_[0];
      if (DeviceTensorStore::GetInstance().Fetch(backend_node_with_index.first.get(),
                                                 device_context->GetDeviceType()) == nullptr) {
        MS_LOG(DEBUG) << "Fetch no device tensor store by:" << backend_node_with_index.first->fullname_with_scope()
                      << ", type:" << device_context->GetDeviceType()
                      << " for actor:" << any_type_kernel_actor->GetAID();
        const auto &device_addresses = DeviceTensorStore::GetInstance().Fetch(backend_node_with_index.first.get());
        const auto &kernel_tensors = DeviceTensorStore::GetInstance().Fetch(backend_node_with_index.first.get());
        if (kernel_tensors.empty() || kernel_tensors[0] == nullptr || kernel_tensors[0]->device_address() == nullptr) {
          MS_LOG(WARNING) << "Failed to get device tensor store by backend node:"
                          << backend_node_with_index.first->DebugString() << " input index:" << iter->first
                          << " in graph output any type kernel actor:" << any_type_kernel_actor->GetAID();
          continue;
        }

        MS_EXCEPTION_IF_NULL(backend_node_with_index.first->cast<ValueNodePtr>());
        const auto &kernel_tensor = AnfAlgo::CreateKernelTensor(
          kernel_tensors[0]->GetShape(), kernel_tensors[0]->GetType(),
          backend_node_with_index.first->cast<ValueNodePtr>()->value(), nullptr, kernel_tensors[0]->size(),
          kernel_tensors[0]->GetStringFormat(), kernel_tensors[0]->dtype_id(), kernel_tensors[0]->GetShapeVector(),
          device::GetDeviceNameByType(device_context->device_context_key().device_type_),
          device_context->device_context_key().device_id_);
        MS_LOG(INFO) << "Create kernel tensor without setting stream id.";
        MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString()
                      << " for actor:" << any_type_kernel_actor->GetAID();
        SchedulerHelper::AddDeviceTensorStore(backend_node_with_index.first, kernel_tensor);
      }

      MS_LOG(INFO) << "Get backend:" << backend_node_with_index.first->DebugString()
                   << " by device tensor store key:" << iter->second->DebugString() << " input index:" << iter->first
                   << " in graph output any type kernel actor:" << any_type_kernel_actor->GetAID();
      iter->second = backend_node_with_index.first;
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
