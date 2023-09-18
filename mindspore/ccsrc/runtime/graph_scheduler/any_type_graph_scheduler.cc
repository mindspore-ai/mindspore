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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include "runtime/graph_scheduler/any_type_graph_scheduler.h"
#include "runtime/graph_scheduler/graph_scheduler.h"
#include "runtime/graph_scheduler/scheduler_helper.h"

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
    if (graph->execution_order().empty()) {
      MS_LOG(INFO) << "The graph " << graph->graph_id() << " is an empty graph and skips building.";
      continue;
    }

    auto actor_name = graph->ToString() + kAnyTypeKernelActorNameSuffix;
    auto any_type_kernel_actor =
      std::make_shared<AnyTypeKernelActor>(actor_name, graph, device_context, memory_manager_aid, debug_id, nullptr);
    any_type_kernel_actor->compile_func_ = graph_compiler_info.compile_func_;
    any_type_kernel_actor->transform_func_ = [this, &graph_compiler_info](const KernelGraphPtr &model_graph,
                                                                          const KernelGraphPtr &real_graph,
                                                                          const DeviceContext *device_context) {
      return Transform(model_graph, real_graph, device_context, graph_compiler_info.origin_parameters_order_);
    };
    any_type_kernel_actor->schedule_func_ = [this](const std::vector<AbstractActorPtr> &actors) {
      auto actor_manager = ActorMgr::GetActorMgrRef();
      MS_EXCEPTION_IF_NULL(actor_manager);
      for (auto actor : actors) {
        MS_EXCEPTION_IF_NULL(actor);
        // The sub actors in the fusion actor do not participate in message interaction.
        if (actor->parent_fusion_actor_ == nullptr) {
          (void)actor_manager->Spawn(actor);
        } else {
          actor->Init();
        }
      }
    };
    MS_EXCEPTION_IF_NULL(any_type_kernel_actor);
    InsertActor(any_type_kernel_actor.get());
    (void)any_type_kernel_actors.emplace_back(any_type_kernel_actor);
  }
  return any_type_kernel_actors;
}

namespace {
std::shared_ptr<GraphCompilerInfo> ConstructGraphCompilerInfo(const KernelGraphPtr &model_graph,
                                                              const KernelGraphPtr &real_graph,
                                                              const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(model_graph);
  MS_EXCEPTION_IF_NULL(real_graph);
  MS_EXCEPTION_IF_NULL(device_context);
  std::vector<KernelGraphPtr> graphs{real_graph};
  auto mutable_device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_context->device_context_key()});

  std::vector<DeviceContext *> device_contexts{mutable_device_context};
  std::vector<std::vector<int64_t> *> tensors_mask;
  std::vector<std::vector<TensorPtr> *> input_tensors;
  std::vector<AnfNodePtr> control_nodes;
  std::vector<AnfNodePtr> origin_parameters_order = model_graph->parameters();
  ControlNodeParserPtr parser = std::make_shared<ControlNodeParser>();
  KernelMapPosition origin_outputs_order;
  MS_EXCEPTION_IF_NULL(model_graph->output());
  auto outputs = common::AnfAlgo::GetAllOutputWithOutMonadAndParameter(model_graph->output());
  for (size_t position = 0; position < outputs.size(); ++position) {
    (void)origin_outputs_order[outputs[position]].emplace_back(position);
  }
  size_t outputs_num = outputs.size();
  std::string name = model_graph->ToString() + "_" + real_graph->ToString();
  bool need_erase = false;
  GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline;
  CompileFunc comile_func{};
  return std::make_shared<GraphCompilerInfo>(graphs, device_contexts, tensors_mask, input_tensors, control_nodes,
                                             origin_parameters_order, parser, origin_outputs_order, outputs_num, name,
                                             need_erase, strategy, comile_func);
}
}  // namespace

void AnyTypeGraphScheduler::TransArrowInDataSourceActorToAnyTypeKernelActor(
  AnyTypeKernelActor *const any_type_kernel_actor, const DataSourceActorPtr &data_source_actor,
  const KernelGraphPtr &model_graph, const KernelGraphPtr &real_graph) {
  MS_EXCEPTION_IF_NULL(any_type_kernel_actor);
  MS_EXCEPTION_IF_NULL(data_source_actor);
  MS_EXCEPTION_IF_NULL(model_graph);
  MS_EXCEPTION_IF_NULL(real_graph);
  // Generate output data.
  data_source_actor->Init();
  auto id = any_type_kernel_actor->current_data_type();
  // Fix from index in output data arrow.
  for (size_t i = 0; i < data_source_actor->output_data_nodes_.size(); ++i) {
    const auto &node = data_source_actor->output_data_nodes_[i];
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "output data node:" << node->DebugString();
    const auto &front_node = real_graph->GetFrontAnfByBackendAnf(node);
    if (front_node == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get front node by data node:" << node->DebugString()
                        << " in real graph:" << real_graph->ToString();
    }
    MS_LOG(DEBUG) << "output data front node:" << front_node->DebugString();
    const auto &front_parameters = model_graph->input_nodes();
    const auto &iter = find(front_parameters.begin(), front_parameters.end(), front_node);
    if (iter == front_parameters.end()) {
      MS_LOG(EXCEPTION) << "Failed to find index by backend parameter:" << node->DebugString()
                        << " front node:" << front_node->DebugString();
    }
    MS_EXCEPTION_IF_NULL(data_source_actor->output_data_arrows_[i]);
    MS_EXCEPTION_IF_NULL(data_source_actor->output_data_[i].first);
    MS_LOG(DEBUG) << "change output index from arrow index:"
                  << data_source_actor->output_data_arrows_[i]->from_output_index_
                  << " and data index:" << data_source_actor->output_data_[i].first->index_
                  << " to:" << iter - front_parameters.begin()
                  << " arrow to actor:" << data_source_actor->output_data_arrows_[i]->to_op_id_
                  << " to index:" << data_source_actor->output_data_arrows_[i]->to_input_index_;

    data_source_actor->output_data_arrows_[i]->from_output_index_ = iter - front_parameters.begin();
  }
  // Collect arrows.
  any_type_kernel_actor->graph_input_data_arrows_[id] = data_source_actor->output_data_arrows_;
  any_type_kernel_actor->graph_input_control_arrows_[id] = data_source_actor->output_control_arrows_;
  any_type_kernel_actor->graph_input_data_nodes_[id] = data_source_actor->output_data_nodes_;
  any_type_kernel_actor->graph_input_data_[id].swap(data_source_actor->output_data_);
  any_type_kernel_actor->data_arrow_to_graph_input_actor_indexs_[id] =
    data_source_actor->data_arrow_to_fusion_actor_indexs_;
  any_type_kernel_actor->batch_graph_input_data_[id] = data_source_actor->batch_output_data_;
  any_type_kernel_actor->batch_graph_input_data_arrows_[id] = data_source_actor->batch_output_data_arrows_;
  any_type_kernel_actor->parent_fusion_actor_ = data_source_actor->parent_fusion_actor_;
}

void AnyTypeGraphScheduler::TransArrowInDataPrepareActorToAnyTypeKernelActor(
  AnyTypeKernelActor *const any_type_kernel_actor, const DataPrepareActorPtr &data_prepare_actor) {
  MS_EXCEPTION_IF_NULL(any_type_kernel_actor);
  MS_EXCEPTION_IF_NULL(data_prepare_actor);
  const auto &id = any_type_kernel_actor->current_data_type();
  for (const auto &control_arrow : data_prepare_actor->output_control_arrows_) {
    MS_EXCEPTION_IF_NULL(control_arrow);
    const auto &actor_name = control_arrow->to_op_id_.Name();
    const auto &actor = FetchActor(actor_name);
    if (actor == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get actor:" << actor_name
                        << " from data prepare actor:" << data_prepare_actor->GetAID();
    }
    if (actor->type() == KernelTransformType::kKernelActor || actor->type() == KernelTransformType::kCustomActor ||
        actor->type() == KernelTransformType::kSuperKernelActor || actor->type() == KernelTransformType::kFusionActor) {
      MS_LOG(DEBUG) << "Add control arrow:" << actor_name << " for actor:" << any_type_kernel_actor->GetAID()
                    << " id:" << id;
      any_type_kernel_actor->graph_input_control_arrows_[id].emplace_back(control_arrow);
      if (actor->type() == KernelTransformType::kFusionActor) {
        auto fusion_actor = dynamic_cast<FusionActor *>(actor);
        MS_EXCEPTION_IF_NULL(fusion_actor);
        const auto &iter = fusion_actor->real_input_controls_.find(data_prepare_actor->GetAID().Name());
        if (iter != fusion_actor->real_input_controls_.end()) {
          std::vector<AbstractActor *> real_inputs;
          for_each(iter->second.begin(), iter->second.end(), [&real_inputs](const auto &actor) {
            if (actor->type() == KernelTransformType::kKernelActor ||
                actor->type() == KernelTransformType::kCustomActor ||
                actor->type() == KernelTransformType::kSuperKernelActor ||
                actor->type() == KernelTransformType::kFusionActor) {
              real_inputs.emplace_back(actor);
            }
          });
          fusion_actor->real_input_controls_.erase(iter);
          fusion_actor->real_input_controls_[any_type_kernel_actor->GetAID().Name()] = real_inputs;
        }
      }
    }
  }
}

void AnyTypeGraphScheduler::TransArrowInLoopCountActorToAnyTypeKernelActor(
  AnyTypeKernelActor *const any_type_kernel_actor, const LoopCountActorPtr &loop_count_actor) {
  MS_EXCEPTION_IF_NULL(any_type_kernel_actor);
  MS_EXCEPTION_IF_NULL(loop_count_actor);
  const auto &id = any_type_kernel_actor->current_data_type();
  for (const auto &control_pair : loop_count_actor->input_control_arrow_aids()) {
    const auto &actor_name = control_pair.first.Name();
    const auto &actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    for (auto &output_control_arrow : actor->output_control_arrows_) {
      MS_EXCEPTION_IF_NULL(output_control_arrow);
      if (output_control_arrow->to_op_id_.Name() == loop_count_actor->GetAID().Name()) {
        output_control_arrow->to_op_id_ = any_type_kernel_actor->GetAID();
        any_type_kernel_actor->graph_output_control_num_[id]++;
      }
    }
  }
}

void AnyTypeGraphScheduler::TransArrowInOutputActorToAnyTypeKernelActor(AnyTypeKernelActor *const any_type_kernel_actor,
                                                                        const OutputActorPtr &output_actor) {
  MS_EXCEPTION_IF_NULL(any_type_kernel_actor);
  MS_EXCEPTION_IF_NULL(output_actor);
  const auto &id = any_type_kernel_actor->current_data_type();
  for (const auto &data_pair : output_actor->input_data_arrow_aids()) {
    const auto &actor_name = data_pair.first.Name();
    const auto &actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    if (actor->type_ != KernelTransformType::kKernelActor && actor->type_ != KernelTransformType::kSuperKernelActor) {
      continue;
    }
    for (auto &output_data_arrow : actor->output_data_arrows_) {
      MS_EXCEPTION_IF_NULL(output_data_arrow);
      if (output_data_arrow->to_op_id_.Name() == output_actor->GetAID().Name()) {
        output_data_arrow->to_op_id_ = any_type_kernel_actor->GetAID();
        MS_EXCEPTION_IF_NULL(any_type_kernel_actor->graph());
        output_data_arrow->to_input_index_ += any_type_kernel_actor->graph()->input_nodes().size();
        any_type_kernel_actor->graph_output_data_num_[id]++;
        MS_LOG(DEBUG) << "Add graph output data arrow for actor:" << any_type_kernel_actor->GetAID()
                      << " from actor:" << actor->GetAID() << " from index:" << output_data_arrow->from_output_index_
                      << " to index:" << output_data_arrow->to_input_index_;
      }
    }
    auto &batch_output_data_arrows = actor->batch_output_data_arrows_;
    if (batch_output_data_arrows.find(output_actor->GetAID().Name()) == batch_output_data_arrows.end()) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(any_type_kernel_actor->graph());
    auto &data_arrows = batch_output_data_arrows[output_actor->GetAID().Name()];
    MS_LOG(DEBUG) << "actor:" << actor->GetAID() << " to actor:" << output_actor->GetAID().Name()
                  << " data arrow size:" << data_arrows.size();
    for (auto &data_arrow : data_arrows) {
      MS_EXCEPTION_IF_NULL(data_arrow);
      if (data_arrow->to_op_id_.Name() == output_actor->GetAID().Name()) {
        data_arrow->to_op_id_ = any_type_kernel_actor->GetAID();
        data_arrow->to_input_index_ += any_type_kernel_actor->graph()->input_nodes().size();
        MS_LOG(DEBUG) << "Add graph output batch data arrow for actor:" << any_type_kernel_actor->GetAID()
                      << " from actor:" << actor->GetAID() << " from index:" << data_arrow->from_output_index_
                      << " to index:" << data_arrow->to_input_index_;
        any_type_kernel_actor->graph_output_data_num_[id]++;
      }
    }
    MS_LOG(DEBUG) << "Update batch arrow to actor:" << any_type_kernel_actor->GetAID().Name()
                  << " for actor:" << actor->GetAID();
    batch_output_data_arrows[any_type_kernel_actor->GetAID().Name()] =
      batch_output_data_arrows[output_actor->GetAID().Name()];
    batch_output_data_arrows.erase(output_actor->GetAID().Name());
    for (const auto &pair : actor->batch_output_data_arrows_) {
      MS_LOG(DEBUG) << "print actor:" << actor->GetAID() << " batch data arrow to actor:" << pair.first
                    << " arrow size:" << pair.second.size();
    }
  }
}
void AnyTypeGraphScheduler::CollectBackendParameterForDynamicShape(AnyTypeKernelActor *const any_type_kernel_actor,
                                                                   const KernelGraphPtr &model_graph,
                                                                   const KernelGraphPtr &real_graph) {
  MS_EXCEPTION_IF_NULL(any_type_kernel_actor);
  MS_EXCEPTION_IF_NULL(model_graph);
  MS_EXCEPTION_IF_NULL(real_graph);
  const auto &id = any_type_kernel_actor->current_data_type();
  std::vector<AnfNodePtr> dynamic_parameters(real_graph->input_nodes().size(), nullptr);
  for (const auto &input_node : real_graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(input_node);
    auto base_shape = input_node->Shape();
    MS_EXCEPTION_IF_NULL(base_shape);
    if (base_shape->IsDynamic() || common::AnfAlgo::IsDynamicSequence(input_node)) {
      const auto &front_node = real_graph->GetFrontAnfByBackendAnf(input_node);
      MS_EXCEPTION_IF_NULL(front_node);
      const auto &iter = find(model_graph->input_nodes().begin(), model_graph->input_nodes().end(), front_node);
      if (iter == model_graph->input_nodes().end()) {
        MS_LOG(EXCEPTION) << "Invalid input node:" << input_node->DebugString()
                          << " front node:" << front_node->DebugString()
                          << " for actor:" << any_type_kernel_actor->GetAID();
      }
      size_t index = iter - model_graph->input_nodes().begin();
      MS_LOG(DEBUG) << "Add dynamic parameter:" << input_node->DebugString() << " in graph:" << real_graph->ToString()
                    << " for actor:" << any_type_kernel_actor->GetAID() << " id:" << id;
      dynamic_parameters[index] = input_node;
    }
  }
  any_type_kernel_actor->graph_input_backend_parameters_[id].swap(dynamic_parameters);
}

void AnyTypeGraphScheduler::TransArrowInActorSetToAnyTypeKernelActor(const ActorSet *const actor_set,
                                                                     const KernelGraphPtr &model_graph,
                                                                     const KernelGraphPtr &real_graph) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(model_graph);
  MS_EXCEPTION_IF_NULL(real_graph);
  MS_LOG(DEBUG) << "Transform arrow in actor set:" << actor_set->name_ << " for model graph:" << model_graph->ToString()
                << " real graph:" << real_graph->ToString();
  const auto &any_type_kernel_actor_name = model_graph->ToString() + kAnyTypeKernelActorNameSuffix;
  auto base_actor = FetchActor(any_type_kernel_actor_name);
  MS_EXCEPTION_IF_NULL(base_actor);
  auto any_type_kernel_actor = dynamic_cast<AnyTypeKernelActor *>(base_actor);
  MS_EXCEPTION_IF_NULL(any_type_kernel_actor);

  // Transfer the arrow in the data source actor to any type kernel actor.
  for (const auto &actor : actor_set->data_source_actors_) {
    if (actor->type() == KernelTransformType::kDeviceDataSourceActor) {
      MS_LOG(EXCEPTION) << "Invalid data source actor:" << actor->GetAID();
    }
    TransArrowInDataSourceActorToAnyTypeKernelActor(any_type_kernel_actor, actor, model_graph, real_graph);
    break;
  }

  // Transfer the arrow in the data prepare actor to any type kernel actor.
  TransArrowInDataPrepareActorToAnyTypeKernelActor(any_type_kernel_actor, actor_set->data_prepare_actor_);

  // Transfer the arrow in the loop count actor to any type kernel actor.
  TransArrowInLoopCountActorToAnyTypeKernelActor(any_type_kernel_actor, actor_set->loop_count_actor_);

  // Transfer the arrow in the output actor to any type kernel actor.
  TransArrowInOutputActorToAnyTypeKernelActor(any_type_kernel_actor, actor_set->output_actor_);

  // Collect backend parameter for dynamic shape.
  CollectBackendParameterForDynamicShape(any_type_kernel_actor, model_graph, real_graph);
}

void PrepareDataForValueNode(const AnfNodePtr &node, const DeviceContext *const device_context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  if (!node->isa<ValueNode>()) {
    return;
  }
  MS_LOG(DEBUG) << "Prepare data for value node:" << node->DebugString() << " node addr:" << node;
  auto device_tensors = DeviceTensorStore::GetInstance().Fetch(node.get());
  for (const auto &device_tensor : device_tensors) {
    if (device_tensor == nullptr) {
      continue;
    }
    const auto &real_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_tensor->device_name(), device_tensor->device_id()});
    MS_EXCEPTION_IF_NULL(real_device_context);
    if (device_tensor->GetPtr() == nullptr) {
      if (!real_device_context->device_res_manager_->AllocateMemory(device_tensor.get())) {
        MS_LOG(EXCEPTION) << "Failed to allocate memory for device tensor store:" << device_tensor;
      }
      MS_LOG(DEBUG) << "Device address:" << device_tensor << " allocate ptr:" << device_tensor->GetPtr()
                    << " for value node:" << node->DebugString();
    } else {
      MS_LOG(DEBUG) << "Device address:" << device_tensor << " already has ptr:" << device_tensor->GetPtr()
                    << " for value node:" << node->DebugString();
    }
    const auto &value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    const auto &value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    tensor::TensorPtr tensor = nullptr;
    if (value->isa<tensor::Tensor>()) {
      tensor = value->cast<TensorPtr>();
    } else if (value->isa<Scalar>()) {
      tensor = ScalarToTensor(value->cast<ScalarPtr>());
    } else if (value->isa<StringImm>()) {
      auto string_value = GetValue<std::string>(value);
      size_t tensor_size = string_value.size();
      ShapeVector shape = {1, SizeToLong(tensor_size)};
      if (!device_tensor->SyncHostToDevice(shape, tensor_size, kObjectTypeString, string_value.data())) {
        MS_LOG(EXCEPTION) << "Failed to sync data for value node:" << node->DebugString();
      }
      MS_LOG(DEBUG) << "Device address:" << device_tensor << " ptr:" << device_tensor->GetPtr()
                    << " for value node:" << node->DebugString();
      return;
    } else {
      MS_LOG(EXCEPTION) << "Invalid value:" << value->ToString();
    }

    if (!device_tensor->SyncHostToDevice(tensor->shape(), tensor->Size(), tensor->data_type(), tensor->data_c())) {
      MS_LOG(EXCEPTION) << "Failed to sync data for value node:" << node->DebugString();
    }
    MS_LOG(DEBUG) << "Device address:" << device_tensor << " ptr:" << device_tensor->GetPtr()
                  << " for value node:" << node->DebugString();
  }
}

void AnyTypeGraphScheduler::FixDeviceTensorStoreKeyInActor(const std::vector<AbstractActorPtr> &actors,
                                                           AnyTypeKernelActor *const any_type_kernel_actor,
                                                           const KernelGraphPtr &model_graph,
                                                           const GraphCompilerInfo &graph_compiler_info,
                                                           const std::vector<AnfNodePtr> &front_parameters) {
  MS_EXCEPTION_IF_NULL(any_type_kernel_actor);
  MS_EXCEPTION_IF_NULL(model_graph);
  if (graph_compiler_info.graphs_.empty() || graph_compiler_info.device_contexts_.empty()) {
    MS_LOG(EXCEPTION) << "Invalid graph compiler info for any type actor:" << any_type_kernel_actor->GetAID();
  }
  auto real_graph = graph_compiler_info.graphs_[0];
  auto device_context = graph_compiler_info.device_contexts_[0];
  MS_EXCEPTION_IF_NULL(real_graph);
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &id = any_type_kernel_actor->current_data_type_;
  for (auto &actor : actors) {
    MS_EXCEPTION_IF_NULL(actor);
    std::vector<std::pair<size_t, AnfNodePtr>> device_tensor_store_keys;
    for (auto &pair : actor->device_tensor_store_keys_) {
      MS_EXCEPTION_IF_NULL(pair.second);
      const auto &front_node = model_graph->GetFrontAnfByBackendAnf(pair.second);
      if (front_node == nullptr) {
        MS_LOG(DEBUG) << "Failed to fetch front node for device tensor store key node:" << pair.second->DebugString()
                      << " node addr:" << pair.second << " index:" << pair.first << " for actor:" << actor->GetAID();
        device_tensor_store_keys.emplace_back(pair);
        PrepareDataForValueNode(pair.second, device_context);
        continue;
      }
      MS_LOG(DEBUG) << "Fix device tensor store key node from:" << pair.second->DebugString()
                    << " node addr:" << pair.second << " to:" << front_node->DebugString()
                    << " node addr:" << front_node << " for actor:" << actor->GetAID() << " index:" << pair.first
                    << " front node addr:" << front_node.get();
      if (!front_node->isa<Parameter>() ||
          find(front_parameters.begin(), front_parameters.end(), front_node) != front_parameters.end()) {
        // In any type kernel actor, if the fallback kernel has a device tensor store, the device type in model graph
        // and real graph will be different, and should be fixed.
        if (actor->device_contexts().size() != 0 && actor->device_contexts()[0] != nullptr &&
            DeviceTensorStore::GetInstance().Fetch(front_node.get(), actor->device_contexts_[0]->GetDeviceType()) ==
              nullptr) {
          auto device_tensor =
            DeviceTensorStore::GetInstance().Fetch(pair.second.get(), actor->device_contexts()[0]->GetDeviceType());
          if (device_tensor != nullptr) {
            MS_LOG(DEBUG) << "Add device tensor store for front node:" << front_node->DebugString()
                          << " by node:" << pair.second->DebugString() << " device tensor:" << device_tensor
                          << " for actor:" << actor->GetAID();
            SchedulerHelper::AddDeviceTensorStore(front_node.get(), device_tensor);
            PrepareDataForValueNode(pair.second, actor->device_contexts_[0]);
          } else {
            MS_LOG(WARNING) << "Failed to get device tensor store by front node:" << front_node->DebugString()
                            << " backend node:" << pair.second->DebugString()
                            << " device type:" << actor->device_contexts()[0]->GetDeviceType()
                            << " for actor:" << actor->GetAID();
          }
        }
        pair.second = front_node;
        device_tensor_store_keys.emplace_back(pair);
        continue;
      }
      MS_LOG(DEBUG) << "Front node:" << front_node->DebugString()
                    << " is a parameter in subgraph and Link data arrow from any type kernel actor:"
                    << any_type_kernel_actor->GetAID() << " to actor:" << actor->GetAID();
      size_t from_index = any_type_kernel_actor->FetchInputNodePosition(pair.second);

      if (actor->parent_fusion_actor() != nullptr) {
        auto base_actor = actor->parent_fusion_actor_;
        auto fusion_actor = dynamic_cast<FusionActor *>(base_actor);
        MS_EXCEPTION_IF_NULL(fusion_actor);
        auto data_arrow = std::make_shared<DataArrow>(from_index, fusion_actor->GetAID(), pair.first);
        auto data = std::make_unique<OpData<DeviceTensor>>(fusion_actor->GetAID(), nullptr, pair.first);
        any_type_kernel_actor->graph_input_data_[id].emplace_back(
          std::make_pair(std::move(data), kOutputDataFlagToFusion));
        fusion_actor->real_input_data_.emplace_back(actor.get(), pair.first);
        any_type_kernel_actor->data_arrow_to_graph_input_actor_indexs_[id][data_arrow.get()] =
          fusion_actor->input_data_arrow_aids_.size();
        (void)fusion_actor->input_data_arrow_aids_.emplace_back(
          std::make_pair(any_type_kernel_actor->GetAID(), data_arrow.get()));
        any_type_kernel_actor->graph_input_data_arrows_[id].emplace_back(data_arrow);
        MS_LOG(DEBUG) << "Any type actor:" << any_type_kernel_actor->GetAID() << " current type:" << id
                      << " add graph input node:" << real_graph->GetBackendAnfByFrontAnf(pair.second)->DebugString()
                      << " from index:" << from_index << " to actor:" << actor->GetAID() << " to index:" << pair.first;
        any_type_kernel_actor->graph_input_data_nodes_[id].emplace_back(
          real_graph->GetBackendAnfByFrontAnf(pair.second));
        actor->input_datas_num_++;
        (void)actor->input_data_arrow_aids_.emplace_back(
          std::make_pair(any_type_kernel_actor->GetAID(), data_arrow.get()));
      } else {
        auto data_arrow = std::make_shared<DataArrow>(from_index, actor->GetAID(), pair.first);
        auto data = std::make_unique<OpData<DeviceTensor>>(actor->GetAID(), nullptr, pair.first);
        any_type_kernel_actor->graph_input_data_arrows_[id].emplace_back(data_arrow);
        MS_LOG(DEBUG) << "Any type actor:" << any_type_kernel_actor->GetAID() << " current type:" << id
                      << " add graph input node:" << real_graph->GetBackendAnfByFrontAnf(pair.second)->DebugString()
                      << " from index:" << from_index << " to actor:" << actor->GetAID() << " to index:" << pair.first;
        any_type_kernel_actor->graph_input_data_nodes_[id].emplace_back(
          real_graph->GetBackendAnfByFrontAnf(pair.second));
        any_type_kernel_actor->graph_input_data_[id].emplace_back(std::make_pair(std::move(data), kOutputDataFlagInit));
        actor->input_datas_num_++;
        (void)actor->input_data_arrow_aids_.emplace_back(
          std::make_pair(any_type_kernel_actor->GetAID(), data_arrow.get()));
      }
    }
    actor->device_tensor_store_keys_.swap(device_tensor_store_keys);
  }
}

std::vector<AbstractActorPtr> AnyTypeGraphScheduler::Transform(const KernelGraphPtr &model_graph,
                                                               const KernelGraphPtr &real_graph,
                                                               const DeviceContext *device_context,
                                                               const std::vector<AnfNodePtr> &front_parameters) {
  MS_EXCEPTION_IF_NULL(model_graph);
  MS_EXCEPTION_IF_NULL(real_graph);
  auto graph_compiler_info = ConstructGraphCompilerInfo(model_graph, real_graph, device_context);
  std::vector<AbstractActorPtr> actors;
  MS_LOG(INFO) << "Start transform for model graph:" << model_graph->ToString()
               << " and real graph:" << real_graph->ToString();
  auto actor_set = GraphScheduler::GetInstance().Transform(*graph_compiler_info);
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_LOG(INFO) << "End transform for model graph:" << model_graph->ToString()
               << " and real graph:" << real_graph->ToString();

  TransArrowInActorSetToAnyTypeKernelActor(actor_set, model_graph, real_graph);
  // Collect actors.
  std::for_each(actor_set->custom_actors_.begin(), actor_set->custom_actors_.end(),
                [&actors](const AbstractActorPtr &actor) { actors.emplace_back(actor); });
  std::for_each(actor_set->kernel_actors_.begin(), actor_set->kernel_actors_.end(),
                [&actors](const AbstractActorPtr &actor) { actors.emplace_back(actor); });
  std::for_each(actor_set->super_kernel_actors_.begin(), actor_set->super_kernel_actors_.end(),
                [&actors](const AbstractActorPtr &actor) { actors.emplace_back(actor); });
  std::for_each(actor_set->fusion_actors_.begin(), actor_set->fusion_actors_.end(),
                [&actors](const AbstractActorPtr &actor) { actors.emplace_back(actor); });
  std::for_each(actor_set->copy_actors_.begin(), actor_set->copy_actors_.end(),
                [&actors](const AbstractActorPtr &actor) { actors.emplace_back(actor); });

  const auto &any_type_kernel_actor_name = model_graph->ToString() + kAnyTypeKernelActorNameSuffix;
  auto base_actor = FetchActor(any_type_kernel_actor_name);
  MS_EXCEPTION_IF_NULL(base_actor);
  auto any_type_kernel_actor = dynamic_cast<AnyTypeKernelActor *>(base_actor);
  FixDeviceTensorStoreKeyInActor(actors, any_type_kernel_actor, model_graph, *graph_compiler_info, front_parameters);
  // Prevent the device address in the graph from being freed.
  graph_compiler_info->graphs_.clear();
  return actors;
}
}  // namespace runtime
}  // namespace mindspore
