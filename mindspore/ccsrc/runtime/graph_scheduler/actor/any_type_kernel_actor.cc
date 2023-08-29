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

#include "runtime/graph_scheduler/actor/any_type_kernel_actor.h"
#include <set>
#include "include/common/debug/anf_ir_dump.h"
#include "mindspore/core/ops/framework_ops.h"

namespace mindspore {
namespace runtime {
AnyTypeKernelActor::AnyTypeKernelActor(const std::string &name, const KernelGraphPtr &graph,
                                       const DeviceContext *device_context, const AID &memory_manager_aid,
                                       const AID *debug_aid, const AID *recorder_aid, KernelTransformType type)
    : SuperKernelActor(name, graph, device_context, memory_manager_aid, debug_aid, recorder_aid, type) {}

void AnyTypeKernelActor::RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  if (!input_data->data_->IsPtrValid() && !TEST_FLAG(input_data->data_->flag(), device::kDeviceAddressFlagNotUsed)) {
    MS_LOG(EXCEPTION) << "The input_data does not have a valid ptr of actor:" << GetAID().Name()
                      << " with index:" << input_data->index_ << ", flag:" << input_data->data_->flag()
                      << " device address:" << input_data->data_ << " ref count:" << input_data->data_->ref_count()
                      << " dynamic ref count:" << input_data->data_->dynamic_ref_count();
  }
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op data:" << input_data->data_
                << " index:" << input_data->index_ << ", size:" << input_data->data_->GetSize()
                << " user data:" << input_data->data_->user_data() << " input num:" << input_datas_num_
                << " input device tensor size:" << input_device_tensors_.size()
                << " ref count:" << input_data->data_->ref_count()
                << " dynamic ref count:" << input_data->data_->dynamic_ref_count();
  if (input_data->index_ < SizeToLong(graph()->input_nodes().size())) {
    // Collect graph input data.
    input_op_datas_[sequential_num].emplace_back(input_data);
    if (CheckRunningCondition(context)) {
      RunForGraphInput(context);
    }
  } else {
    // Collect graph output data.
    graph_output_op_data_[sequential_num].emplace_back(input_data);
    if (CheckGraphOutputRunningCondition(context)) {
      RunForGraphOutput(context);
    }
  }
}

void AnyTypeKernelActor::RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op control:" << input_control->Name();
  if (std::any_of(
        input_control_arrow_aids_.begin(), input_control_arrow_aids_.end(),
        [input_control](const auto &arrow_pair) { return arrow_pair.first.Name() == input_control->Name(); })) {
    (void)input_op_controls_[sequential_num].emplace_back(input_control);
    if (CheckRunningCondition(context)) {
      RunForGraphInput(context);
    }
  } else {
    graph_output_op_control_[sequential_num].emplace_back(input_control);
    if (CheckGraphOutputRunningCondition(context)) {
      RunForGraphOutput(context);
    }
  }
}

void AnyTypeKernelActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *const context) {
  std::vector<DeviceTensor *> memory_free_list = graph_ouput_device_tensors_;
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter == input_op_datas_.end()) {
    return;
  }
  for (auto &input_data : data_iter->second) {
    MS_EXCEPTION_IF_NULL(input_data);
    MS_EXCEPTION_IF_NULL(input_data->data_);
    size_t index = IntToSize(input_data->index_);
    if (index >= input_device_tensors_.size()) {
      std::string error_info = "Invalid input index:" + std::to_string(index) +
                               " total:" + std::to_string(input_device_tensors_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_device_tensors_[index] = input_data->data_;
    if (input_data->data_->ref_count() != SIZE_MAX) {
      (void)memory_free_list.emplace_back(input_data->data_);
    }
  }
  memory_free_lists_.push(memory_free_list);

  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto device_tensor = DeviceTensorStore::GetInstance()
                           .Fetch(device_tensor_store_key.second.get(), device_contexts_[0]->GetDeviceType())
                           .get();
    if (device_tensor == nullptr) {
      MS_LOG(EXCEPTION) << "Failed get device tensor for node:" << device_tensor_store_key.second->DebugString()
                        << " index:" << device_tensor_store_key.first;
      continue;
    }
    if (device_tensor_store_key.first >= input_device_tensors_.size()) {
      std::string error_info = "Invalid input index:" + std::to_string(device_tensor_store_key.first) +
                               " total:" + std::to_string(input_device_tensors_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_device_tensors_[device_tensor_store_key.first] = device_tensor;
  }
}

bool AnyTypeKernelActor::CheckGraphOutputRunningCondition(const OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_LOG(DEBUG) << "graph output data num:" << graph_output_data_num_[current_data_type_]
                << " control num:" << graph_output_control_num_[current_data_type_];
  if (graph_output_data_num_[current_data_type_] != 0) {
    const auto &data_iter = graph_output_op_data_.find(context->sequential_num_);
    if (data_iter == graph_output_op_data_.end()) {
      return false;
    }
    if (data_iter->second.size() < graph_output_data_num_[current_data_type_]) {
      return false;
    } else if (data_iter->second.size() > graph_output_data_num_[current_data_type_]) {
      MS_LOG(ERROR) << "Invalid graph output data num:" << data_iter->second.size()
                    << " need:" << graph_output_data_num_[current_data_type_] << " for actor:" << GetAID()
                    << ", sequential num:" << context->sequential_num_;
      return false;
    }
  }

  if (graph_output_control_num_[current_data_type_] != 0) {
    const auto &control_iter = graph_output_op_control_.find(context->sequential_num_);
    if (control_iter == graph_output_op_control_.end()) {
      return false;
    }
    if (control_iter->second.size() < graph_output_control_num_[current_data_type_]) {
      return false;
    } else if (control_iter->second.size() > graph_output_control_num_[current_data_type_]) {
      MS_LOG(ERROR) << "Invalid input control num:" << control_iter->second.size()
                    << " need:" << graph_output_control_num_[current_data_type_] << " for actor:" << GetAID()
                    << ", sequential num:" << context->sequential_num_;
      return false;
    }
  }
  return true;
}
namespace {
GraphSegmentPtr BuildSegmentByGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> nodes;
  std::vector<AnfNodePtr> all_nodes = TopoSort(graph->get_return());
  for (const auto &node : all_nodes) {
    if (node == nullptr || (!node->isa<CNode>()) || common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
      continue;
    }
    MS_LOG(DEBUG) << "build new segment node:" << node->DebugString();
    nodes.emplace_back(node);
  }
  return std::make_shared<GraphSegment>(nodes, false);
}

std::string GenerateIDForGraph(const std::vector<DeviceTensor *> &device_tensors, const std::vector<size_t> &indexes) {
  std::string id;
  auto get_shape_and_type_string = [&id](const ShapeVector &shape_vector, TypeId type_id) {
    id += "shape_";
    (void)std::for_each(shape_vector.begin(), shape_vector.end(), [&id](int64_t shape) {
      id += std::to_string(shape);
      id += "_";
    });
    id = id + "type_" + std::to_string(type_id) + "_";
  };
  for (const auto &index : indexes) {
    if (index >= device_tensors.size()) {
      MS_LOG(EXCEPTION) << "Invalid parameter index:" << index << " for device tensor num:" << device_tensors.size();
    }
    id = id + "index_" + std::to_string(index) + "_";
    const auto &device_tensor = device_tensors[index];
    if (device_tensor == nullptr) {
      MS_LOG(EXCEPTION) << "Empty device tensor index:" << index;
    }
    const auto &node = device_tensor->GetNodeIndex().first;
    if (node == nullptr || node->abstract() == nullptr) {
      get_shape_and_type_string(device_tensor->host_shape(), device_tensor->type_id());
      continue;
    }
    if (node->abstract()->isa<abstract::AbstractSequence>()) {
      auto sequence_abs = node->abstract()->cast<abstract::AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(sequence_abs);
      id = id + "Tuple_" + std::to_string(sequence_abs->size()) + "_";
    } else if (node->abstract()->isa<abstract::AbstractScalar>()) {
      id = id + "Scalar_";
    } else if (node->abstract()->isa<abstract::AbstractTensor>()) {
      id = id + "Tensor_";
    }
    get_shape_and_type_string(trans::GetRuntimePaddingShape(node, device_tensor->GetNodeIndex().second),
                              device_tensor->type_id());
  }
  return id;
}

void InferParameterAbstractForModelGraph(const KernelGraphPtr &graph, const std::vector<DeviceTensor *> &device_tensors,
                                         const std::vector<size_t> &indexes) {
  MS_EXCEPTION_IF_NULL(graph);
  for (size_t index : indexes) {
    if (index >= device_tensors.size() || index >= graph->input_nodes().size()) {
      MS_LOG(EXCEPTION) << "Invalid index:" << index << " for input device tensor size:" << device_tensors.size()
                        << " for graph:" << graph->ToString();
    }
    const auto &device_tensor = device_tensors[index];
    MS_EXCEPTION_IF_NULL(device_tensor);
    auto input_node = graph->input_nodes()[index];
    MS_EXCEPTION_IF_NULL(input_node);
    const auto &node = device_tensor->GetNodeIndex().first;
    if (node != nullptr && node->abstract() != nullptr) {
      MS_LOG(DEBUG) << "node:" << node->DebugString() << " abstract:" << node->abstract();
      if (node->abstract()->isa<abstract::AbstractSequence>()) {
        auto new_abstract = node->abstract()->Clone();
        MS_EXCEPTION_IF_NULL(new_abstract);
        auto seq_abstract = new_abstract->cast<abstract::AbstractSequencePtr>();
        MS_EXCEPTION_IF_NULL(seq_abstract);
        // Dynamic len element is used to check if the sequence is dynamic len.
        if (!seq_abstract->elements().empty() && seq_abstract->elements()[0] != nullptr) {
          seq_abstract->set_dynamic_len_element_abs(seq_abstract->elements()[0]->Clone());
        }
        seq_abstract->set_dynamic_len(true);
        input_node->set_abstract(seq_abstract);
        MS_LOG(INFO) << "Set abstract:" << seq_abstract->ToString() << " to input node:" << input_node->DebugString()
                     << " by device address:" << device_tensor << " type id:" << device_tensor->type_id();
        continue;
      }
      MS_LOG(INFO) << "Set abstract:" << node->abstract()->ToString() << " to input node:" << input_node->DebugString()
                   << device_tensor << " type id:" << device_tensor->type_id();
      input_node->set_abstract(node->abstract());
      continue;
    }
    ShapeVector shape;
    if (node == nullptr) {
      shape = device_tensor->host_shape();
      MS_LOG(DEBUG) << "Fetch shape from device tensor by index:" << index << " device tensor:" << device_tensor;
    } else {
      MS_LOG(DEBUG) << "Fetch shape from node:" << node->DebugString() << " node:" << node->fullname_with_scope()
                    << " by index:" << index;
      shape = trans::GetRuntimePaddingShape(node, device_tensors[index]->GetNodeIndex().second);
    }
    const auto &abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(device_tensor->type_id()), shape);
    MS_LOG(DEBUG) << "set abstract:" << abstract->ToString() << " for input node:" << input_node->DebugString()
                  << device_tensor << " type id:" << device_tensor->type_id();
    input_node->set_abstract(abstract);
  }
}

TypeId GetElementType(const abstract::AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  TypePtr type = nullptr;
  if (abstract->isa<abstract::AbstractScalar>()) {
    type = abstract->BuildType();
  } else if (abstract->isa<abstract::AbstractTensor>()) {
    const auto &tensor_abs = abstract->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor_abs);
    MS_EXCEPTION_IF_NULL(tensor_abs->element());
    type = tensor_abs->element()->BuildType();
  } else if (abstract->isa<abstract::AbstractSequence>()) {
    const auto &sequence_abs = abstract->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(sequence_abs);
    if (sequence_abs->dynamic_len() || sequence_abs->elements().empty() || sequence_abs->elements()[0] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid abstract:" << abstract->ToString();
    }
    return GetElementType(sequence_abs->elements()[0]);
  } else {
    MS_LOG(EXCEPTION) << "Invalid abstract:" << abstract->ToString();
  }
  MS_EXCEPTION_IF_NULL(type);
  return type->type_id();
}
}  // namespace

void AnyTypeKernelActor::UpdataDynamicShapeParameter(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (graph_input_backend_parameters_.find(current_data_type_) == graph_input_backend_parameters_.end()) {
    return;
  }
  for (size_t i = 0; i < graph_input_backend_parameters_[current_data_type_].size(); ++i) {
    const auto &parameter = graph_input_backend_parameters_[current_data_type_][i];
    if (parameter == nullptr) {
      continue;
    }
    if (i >= input_device_tensors_.size() || input_device_tensors_[i] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid input index " << i << " for device tensor size:" << input_device_tensors_.size();
    }
    auto node = input_device_tensors_[i]->GetNodeIndex().first;
    MS_EXCEPTION_IF_NULL(node);
    if (common::AnfAlgo::IsDynamicSequence(parameter)) {
      MS_LOG(DEBUG) << "Update abstract for parameter:" << parameter << " "
                    << " input device tensor:" << input_device_tensors_[i]
                    << " type:" << input_device_tensors_[i]->type_id();
      const auto &shapes = BaseShapeToShapeVector(node->Shape());
      MS_EXCEPTION_IF_NULL(node->Type());
      std::vector<TypeId> types = std::vector(shapes.size(), GetElementType(node->abstract()));
      common::AnfAlgo::SetScalarTupleOutputInferType(types, shapes, parameter);
      continue;
    }
    MS_EXCEPTION_IF_NULL(node->Type());
    auto shape = trans::GetRuntimePaddingShape(node, input_device_tensors_[i]->GetNodeIndex().second);
    common::AnfAlgo::SetOutputInferTypeAndShape({node->Type()->type_id()}, {shape}, parameter.get());
  }
}

void AnyTypeKernelActor::RunForGraphInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  actor_state_ = AnyTypeKernelActorState::kAnyTypeKernelActorSendInput;
  MS_LOG(DEBUG) << "Any type kernel actor:" << GetAID() << " run for graph input.";
  FetchInputDeviceTensor(context);
  current_data_type_ = GenerateIDForGraph(input_device_tensors_, any_type_parameter_indexes_);
  MS_LOG(DEBUG) << "Current data type:" << current_data_type_ << " for actor:" << GetAID();
  vector<AbstractActorPtr> actors;
  if (real_graphs_.find(current_data_type_) == real_graphs_.end()) {
    InferParameterAbstractForModelGraph(graph(), input_device_tensors_, any_type_parameter_indexes_);
    graph()->InferType();
    const auto &return_node = graph()->get_return();
    MS_EXCEPTION_IF_NULL(return_node);
    if (!return_node->isa<CNode>() || return_node->cast<CNodePtr>()->inputs().size() <= 1) {
      MS_LOG(EXCEPTION) << "Invalid return node:" << return_node->DebugString() << " for graph:" << graph()->ToString();
    }
    if (device_contexts().empty() || device_contexts()[0] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid device context for actor:" << GetAID();
    }
    AnfNodePtrList inputs{};
    AnfNodePtrList outputs{return_node->cast<CNodePtr>()->input(1)};
    auto io_nodes = std::make_pair(inputs, outputs);
    auto new_graph = compile_func_(BuildSegmentByGraph(graph()), io_nodes, device_contexts()[0], graph()->RunMode());
    MS_EXCEPTION_IF_NULL(new_graph);
    MS_LOG(INFO) << "Add new kernel graph:" << new_graph->ToString() << " for graph:" << graph()->ToString();
    real_graphs_[current_data_type_] = new_graph;
    actors = transform_func_(graph(), new_graph, device_contexts()[0]);
    actors_[current_data_type_] = actors;
    schedule_func_(actors);
  }
  UpdataDynamicShapeParameter(context);
  EraseInput(context);
  if (memory_alloc_list_.size() > 0) {
    MS_LOG(EXCEPTION) << "Any type kernel actor:" << GetAID() << "cannot send memory alloc message.";
  } else {
    OnMemoryAllocFinish(context);
  }
}

size_t FetchInputIndexByBackendParameter(const AnfNodePtr &backend_node, const KernelGraphPtr &front_graph,
                                         const KernelGraphPtr &backend_graph) {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(front_graph);
  MS_EXCEPTION_IF_NULL(backend_graph);
  const auto &front_node = backend_graph->GetFrontAnfByBackendAnf(backend_node);
  MS_EXCEPTION_IF_NULL(front_node);
  const auto &front_parameters = front_graph->input_nodes();
  const auto &iter = find(front_parameters.begin(), front_parameters.end(), front_node);
  if (iter == front_parameters.end()) {
    MS_LOG(EXCEPTION) << "Invalid front parameter:" << front_node->DebugString()
                      << " for graph:" << front_graph->ToString();
  }
  return iter - front_parameters.begin();
}
void AnyTypeKernelActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(graph());
  if (real_graphs_.find(current_data_type_) == real_graphs_.end()) {
    MS_LOG(EXCEPTION) << "Invalid index:" << current_data_type_ << " for any type kernel actor:" << GetAID();
  }
  const auto &real_graph = real_graphs_[current_data_type_];
  MS_EXCEPTION_IF_NULL(real_graph);
  if (real_graph->input_nodes().size() != graph()->input_nodes().size()) {
    MS_LOG(EXCEPTION) << "Invalid input node num:" << real_graph->input_nodes().size()
                      << " in graph:" << real_graph->ToString() << " for model graph:" << graph()->ToString()
                      << " input num:" << graph()->input_nodes().size() << " for actor:" << GetAID();
  }
  for (size_t i = 0; i < node_device_tensors_.size(); ++i) {
    const auto &input_node = real_graph->input_nodes()[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (HasAbstractMonad(input_node)) {
      continue;
    }
    size_t from_index = FetchInputIndexByBackendParameter(input_node, graph(), real_graph);
    if (!AnfAlgo::OutputAddrExist(input_node, 0, false)) {
      MS_LOG(EXCEPTION) << "Input node:" << input_node->DebugString()
                        << " has no device address for actor:" << GetAID();
    }
    auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
    MS_EXCEPTION_IF_NULL(device_address);
    if (from_index >= node_device_tensors_.size() || from_index >= input_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid from index:" << from_index
                        << " node device tensor size:" << node_device_tensors_.size()
                        << " input device tensor size:" << input_device_tensors_.size() << " for actor:" << GetAID();
    }
    node_device_tensors_[from_index] = device_address;
    if (input_device_tensors_[from_index] == nullptr) {
      MS_LOG(EXCEPTION) << "actor:" << GetAID() << " real graph:" << real_graph->ToString()
                        << " input node:" << input_node->DebugString() << " index : " << i << " is nullptr ";
    }
    node_device_tensors_[from_index]->SetNodeIndex(input_device_tensors_[from_index]->node_index().first.lock(),
                                                   input_device_tensors_[from_index]->node_index().second);
    MS_LOG(DEBUG) << "Actor:" << GetAID() << " input " << from_index << ":"
                  << " device address:" << device_address
                  << " original ref count:" << device_address->original_ref_count()
                  << " ref count:" << device_address->ref_count()
                  << " dynamic ref count:" << device_address->dynamic_ref_count();
  }
  CopyInputData(context, real_graphs_[current_data_type_]);
  SendOutput(context);
}

void AnyTypeKernelActor::EraseGraphOutput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if ((graph_output_data_num_[current_data_type_] != 0) && (!graph_output_op_data_.empty())) {
    auto ret = graph_output_op_data_.erase(context->sequential_num_);
    if (ret == 0) {
      MS_LOG(WARNING) << "Erase graph output data failed: " << GetAID().Name()
                      << ", sequential_num: " << context->sequential_num_;
      return;
    }
  }

  if ((graph_output_control_num_[current_data_type_] != 0) && (!graph_output_op_control_.empty())) {
    auto ret = graph_output_op_control_.erase(context->sequential_num_);
    if (ret == 0) {
      MS_LOG(WARNING) << "Erase graph output controls failed: " << GetAID().Name()
                      << ", sequential_num: " << context->sequential_num_;
      return;
    }
  }
}

void AnyTypeKernelActor::RunForGraphOutput(OpContext<DeviceTensor> *const context) {
  MS_LOG(DEBUG) << "actor:" << GetAID() << " run for graph output start";
  actor_state_ = AnyTypeKernelActorState::kAnyTypeKernelActorSendOutput;
  FetchGraphOutput(context);
  EraseGraphOutput(context);
  SendMemoryFreeReq(context);
  AbstractActor::SendOutput(context);
}

void AnyTypeKernelActor::Init() {
  MS_LOG(DEBUG) << "actor:" << GetAID() << " init";
  SuperKernelActor::Init();
  memory_alloc_list_.clear();
  for (size_t i = 0; i < graph()->input_nodes().size(); ++i) {
    const auto &input = graph()->input_nodes()[i];
    MS_EXCEPTION_IF_NULL(input);
    const auto &abs = input->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (abs->isa<abstract::AbstractAny>()) {
      any_type_parameter_indexes_.emplace_back(i);
      MS_LOG(DEBUG) << "Add any type parameter index:" << i << " by parameter:" << input->DebugString()
                    << " for actor:" << GetAID();
    }
  }
  for (const auto &node_with_index : common::AnfAlgo::GetAllOutputWithOutMonadAndParameter(graph()->output())) {
    MS_EXCEPTION_IF_NULL(node_with_index.first);
    if (!AnfAlgo::OutputAddrExist(node_with_index.first, node_with_index.second)) {
      MS_LOG(EXCEPTION) << "Failed to get output address from node:" << node_with_index.first->DebugString()
                        << " index:" << node_with_index.second << " for actor:" << GetAID();
    }
    graph_ouput_device_tensors_.emplace_back(
      AnfAlgo::GetMutableOutputAddr(node_with_index.first, node_with_index.second).get());
  }
}

void AnyTypeKernelActor::FetchGraphOutput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  const auto &data_iter = graph_output_op_data_.find(context->sequential_num_);
  if (data_iter != graph_output_op_data_.end()) {
    std::set<DeviceTensor *> clear_device_tensors;
    for (auto &graph_output_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(graph_output_data);
      MS_EXCEPTION_IF_NULL(graph_output_data->data_);
      size_t index = IntToSize(graph_output_data->index_);
      if (index < graph()->input_nodes().size()) {
        MS_LOG(WARNING) << "Invalid graph output index:" << index << " input num:" << input_datas_num_
                        << " for actor:" << GetAID();
        continue;
      }
      index -= graph()->input_nodes().size();
      if (index >= graph_ouput_device_tensors_.size()) {
        std::string error_info = "Invalid input index:" + std::to_string(index) +
                                 " total:" + std::to_string(graph_ouput_device_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      MS_LOG(DEBUG) << "Fetch graph output index:" << index << " set ptr:" << graph_output_data->data_->GetMutablePtr()
                    << " size:" << graph_output_data->data_->GetSize()
                    << " from device address:" << graph_output_data->data_
                    << " to:" << graph_ouput_device_tensors_[index] << " for actor:" << GetAID();
      graph_ouput_device_tensors_[index]->set_ptr(graph_output_data->data_->GetMutablePtr());
      clear_device_tensors.emplace(graph_output_data->data_);
      graph_ouput_device_tensors_[index]->SetSize(graph_output_data->data_->GetSize());
      auto node_with_index = graph_output_data->data_->node_index();
      graph_ouput_device_tensors_[index]->SetNodeIndex(node_with_index.first.lock(), node_with_index.second);
      MS_LOG(DEBUG) << "src device address:" << graph_output_data->data_
                    << " shape:" << graph_output_data->data_->host_shape()
                    << " type:" << graph_output_data->data_->type_id()
                    << "dst device address:" << graph_ouput_device_tensors_[index]
                    << " shape:" << graph_ouput_device_tensors_[index]->host_shape()
                    << " type:" << graph_ouput_device_tensors_[index]->type_id();
      graph_ouput_device_tensors_[index]->set_type_id(graph_output_data->data_->type_id());
      graph_ouput_device_tensors_[index]->set_host_shape(graph_output_data->data_->host_shape());
      graph_ouput_device_tensors_[index]->set_user_data(graph_output_data->data_->user_data());
    }
    for_each(clear_device_tensors.begin(), clear_device_tensors.end(),
             [](DeviceTensor *device_tensor) { device_tensor->set_ptr(nullptr); });
  }
}

void AnyTypeKernelActor::UpdateOutputData(OpData<DeviceTensor> *const output_data, const DataArrowPtr &data_arrow,
                                          const AnfNodePtr &output_node, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output_data);
  MS_EXCEPTION_IF_NULL(data_arrow);
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph());
  if (actor_state_ == AnyTypeKernelActorState::kAnyTypeKernelActorSendOutput) {
    size_t index = data_arrow->from_output_index_;
    const auto &real_output = common::AnfAlgo::GetAllOutputWithOutMonadAndParameter(graph()->output());
    const auto &output_iter = find(real_output.begin(), real_output.end(), std::make_pair(output_node, index));
    if (output_iter == real_output.end()) {
      MS_LOG(EXCEPTION) << "Invalid output node:" << output_node->DebugString() << " index:" << index
                        << " for graph:" << graph()->ToString();
    }
    size_t real_output_index = output_iter - real_output.begin();

    if (real_output_index >= graph_ouput_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid input index:" << real_output_index << " by node:" << output_node->DebugString()
                        << " for actor:" << GetAID();
    }
    MS_LOG(DEBUG) << "actor:" << GetAID() << " output node:" << output_node->DebugString()
                  << " to actor:" << data_arrow->to_op_id_ << " from index:" << real_output_index;
    MS_EXCEPTION_IF_NULL(graph_ouput_device_tensors_[real_output_index]);
    output_data->data_ = graph_ouput_device_tensors_[real_output_index];
    return;
  }

  const auto &real_graph = real_graphs_[current_data_type_];
  MS_EXCEPTION_IF_NULL(real_graph);
  const auto &front_node = real_graph->GetFrontAnfByBackendAnf(output_node);
  MS_EXCEPTION_IF_NULL(front_node);
  const auto &model_graph = SuperKernelActor::graph();
  MS_EXCEPTION_IF_NULL(model_graph);
  auto &input_nodes = model_graph->input_nodes();
  const auto &iter = find(input_nodes.begin(), input_nodes.end(), front_node);
  if (iter == input_nodes.end()) {
    MS_LOG(EXCEPTION) << "Invalid input node:" << output_node->DebugString()
                      << " front node:" << front_node->DebugString();
  }
  size_t index = iter - input_nodes.begin();
  if (index >= node_device_tensors_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input index:" << index << " by node:" << output_node->DebugString()
                      << " for actor:" << GetAID();
  }
  if (node_device_tensors_[index] == nullptr) {
    MS_LOG(EXCEPTION) << "failed to get input index:" << index << " for actor:" << GetAID();
  }
  output_data->data_ = node_device_tensors_[index].get();
}

void AnyTypeKernelActor::SendOutput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_LOG(DEBUG) << "Any type actor:" << GetAID() << " send output";
  // Must be the execution order: send data --> send control, avoid the illegal timing problem.
  SendOutputData(context, graph_input_data_nodes_[current_data_type_], graph_input_data_arrows_[current_data_type_],
                 graph_input_data_[current_data_type_], data_arrow_to_graph_input_actor_indexs_[current_data_type_],
                 &batch_graph_input_data_[current_data_type_]);

  // 2.Send output control.
  if (graph_input_control_arrows_[current_data_type_].size() > 0) {
    auto from_aid = const_cast<AID *>(&GetAID());
    for (auto &output_control : graph_input_control_arrows_[current_data_type_]) {
      MS_EXCEPTION_IF_NULL(output_control);
      if (TEST_FLAG(output_control->flag_, kOutputDataFlagBetweenFusion)) {
        const auto &to_actor = FetchSubActorInFusionActor(output_control->to_op_id_.Name());
        ActorDispatcher::SendSync(to_actor, &OpActor::RunOpControl, from_aid, context);
      } else {
        ActorDispatcher::Send(output_control->to_op_id_, &OpActor::RunOpControl, from_aid, context);
      }
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
