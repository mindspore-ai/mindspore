/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/runtime/actors/base/output_actor.h"
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "backend/ms_backend/runtime/actors/base/memory_manager_actor.h"
#include "backend/ms_backend/runtime/graph_executor/kernel_capture/graph_capture_manager.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/backend/common/device_address_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "ir/map_tensor.h"
#include "ir/tensor_new.h"
#include "ir/dtype/tensor_type.h"
#include "utils/convert_utils.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/parameter_store.h"

namespace mindspore {
namespace runtime {
using distributed::collective::CollectiveManager;

void UpdateOutputTensorShape(const std::vector<TensorPtr> &output_tensors,
                             const std::vector<KernelWithIndex> &output_nodes) {
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_tensors[i]);
    if (output_tensors[i]->isa<tensor::MapTensor>()) {
      continue;
    }
    auto shape = common::AnfAlgo::GetOutputInferShape(output_nodes[i].first, output_nodes[i].second);
    (void)output_tensors[i]->set_shape(shape);
  }
}

void UpdateDynamicSequenceType(const AnfNodePtr &output_node, const kernel::KernelTensorPtr &output_kernel_tensor) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);

  if (!common::AnfAlgo::IsDynamicSequence(output_node)) {
    return;
  }

  if (output_node->abstract() == nullptr || (!output_node->abstract()->isa<abstract::AbstractSequence>())) {
    MS_LOG(WARNING) << "Skip update type for output node:" << output_node->DebugString();
    return;
  }

  if (output_kernel_tensor->GetShape() == nullptr ||
      (!output_kernel_tensor->GetShape()->isa<abstract::SequenceShape>())) {
    MS_LOG(WARNING) << "Skip update type for output node:" << output_node->DebugString() << " as invalid shape:"
                    << (output_kernel_tensor->GetShape() == nullptr ? "nullptr"
                                                                    : output_kernel_tensor->GetShape()->ToString());
    return;
  }

  abstract::AbstractBasePtr element_abstract = nullptr;
  const auto &sequence_abstract = output_node->abstract()->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(sequence_abstract);
  if (sequence_abstract->dynamic_len()) {
    if (sequence_abstract->dynamic_len_element_abs() != nullptr) {
      element_abstract = sequence_abstract->dynamic_len_element_abs();
    }
  } else if (sequence_abstract->size() != 0) {
    element_abstract = sequence_abstract->elements()[0];
  }

  TypePtr element_type = TypeIdToType(output_kernel_tensor->dtype_id());
  if (element_abstract != nullptr && element_abstract->isa<abstract::AbstractTensor>()) {
    element_type = std::make_shared<TensorType>(element_type);
  }

  const auto &sequence_shape = output_kernel_tensor->GetShape()->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(sequence_shape);
  TypePtrList types(sequence_shape->size(), element_type);
  if (sequence_abstract->isa<abstract::AbstractTuple>()) {
    output_kernel_tensor->SetType(std::make_shared<Tuple>(types));
    return;
  }
  output_kernel_tensor->SetType(std::make_shared<List>(types));
}

device::DeviceAddressPtr MakeTensorContiguousCallback(const tensor::TensorPtr &input_tensor) {
  const auto &address = input_tensor->device_address();
  MS_EXCEPTION_IF_NULL(address);
  const auto &storage_info = input_tensor->storage_info();
  if (storage_info == nullptr) {
    return address;
  }
  return DeviceAddressUtils::ConvertContiguousDeviceAddress(nullptr, input_tensor);
}

void SyncOutputFromTensor(const TensorPtr &dst_tensor, const KernelTensorPtr &src_kernel_tensor,
                          const AnfNodePtr &output_node) {
  MS_EXCEPTION_IF_NULL(dst_tensor);
  MS_EXCEPTION_IF_NULL(src_kernel_tensor);
  const auto &tensor_device_address = dst_tensor->device_address();
  const auto &device_tensor = src_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(tensor_device_address);
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (runtime::IsDisableRuntimeConfig(runtime::kRuntimeCopyAsync)) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Sync device data from device tensor: " << device_tensor << ", to device tensor: " << tensor_device_address
      << ", size: " << device_tensor->GetSize();
    if (!SyncCopy(dst_tensor, src_kernel_tensor.get(), kDefaultStreamIndex) ||
        !SyncAllStreamForDeviceAddress(tensor_device_address, device_tensor)) {
      MS_LOG_WITH_NODE(EXCEPTION, output_node)
        << "Sync device to device failed, device type: " << tensor_device_address->GetDeviceType()
        << ", output node: " << output_node->fullname_with_scope();
    }
  } else {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Async device data from device tensor: " << device_tensor << ", to device tensor: " << tensor_device_address
      << ", size: " << device_tensor->GetSize();
    if (!AsyncCopy(dst_tensor, src_kernel_tensor.get(), kDefaultStreamIndex)) {
      MS_LOG_WITH_NODE(EXCEPTION, output_node)
        << "Async device to device failed, device type: " << tensor_device_address->GetDeviceType()
        << ", output node: " << output_node->fullname_with_scope();
    }
  }
}

void RecordOutputTypes(std::vector<TypePtr> *output_types, const std::vector<KernelTensorPtr> &output_kernel_tensors,
                       size_t size) {
  MS_EXCEPTION_IF_NULL(output_types);
  output_types->clear();
  std::transform(output_kernel_tensors.begin(), output_kernel_tensors.begin() + size, std::back_inserter(*output_types),
                 [](const KernelTensorPtr &kernel_tensor) {
                   return (kernel_tensor != nullptr) ? kernel_tensor->GetType() : nullptr;
                 });
}

void OutputActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != output_nodes_.size()) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }
  // Check outputs number.
  if (output_nodes_.size() != outputs_.size()) {
    MS_LOG(EXCEPTION) << "The outputs number is wrong.";
  }
  // Check output device tensors number.
  if (outputs_.size() != output_kernel_tensors_.size()) {
    MS_LOG(EXCEPTION) << "The output device tensors number is wrong.";
  }

  // Set the number of actor running dependent messages.
  running_dependent_msg_num_ = SizeToInt(outputs_num_ - device_tensor_store_keys_.size());
}

void OutputActor::FreeOutputNodeMem() {
  for (size_t i = 0; i < output_nodes_.size(); ++i) {
    auto &output_node = output_nodes_[i].first;
    auto &output_kernel_tensor = output_kernel_tensors_[i];
    if (output_kernel_tensor == nullptr) {
      continue;
    }
    auto &output_device_tensor = output_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(output_device_tensor);
    // The output_device_tensor may be repeated.
    if ((output_node == nullptr) || (output_device_tensor == nullptr) || (output_device_tensor->GetPtr() == nullptr)) {
      continue;
    }
    const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {output_device_tensor->GetDeviceType(), output_device_tensor->device_id()});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Free device address:" << output_device_tensor << " for actor:" << GetAID();
    MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(output_kernel_tensor.get(), device_context,
                                                            GetAID().Name());
  }
}

void OutputActor::FreeSummaryNodeMem() {
  for (size_t i = 0; i < summary_nodes_.size(); ++i) {
    auto &summary_node = summary_nodes_[i].first;
    auto index = summary_nodes_[i].second;
    if (summary_node == nullptr) {
      continue;
    }
    auto output_device_addr = AnfAlgo::GetMutableOutputAddr(summary_node, index, false);
    auto output_kernel_tensor = AnfAlgo::GetOutputKernelTensor(summary_node, index, false);
    if (output_kernel_tensor == nullptr) {
      continue;
    }
    if ((output_device_addr == nullptr) || (output_device_addr->GetPtr() == nullptr)) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Empty ptr in summary node:" << summary_node->DebugString()
                                                   << " index:" << index << " device address:" << output_device_addr;
      continue;
    }
    if (!IsOutputAddressPersisted(output_kernel_tensor, summary_nodes_[i])) {
      FreeMemoryByDeviceContext(output_kernel_tensor->device_address().get(), nullptr);
    }
  }
}

void OutputActor::ClearOutputCache() {
  old_to_new_kernel_tensor_.clear();
  outputs_.clear();
  outputs_.resize(outputs_num_);
  output_nodes_.clear();
  output_nodes_.resize(outputs_num_);
  output_kernel_tensors_.clear();
  output_kernel_tensors_.resize(outputs_num_);

  current_outputs_num_ = 0;
  current_count_ = 0;
}

void OutputActor::FetchParameterInput(OpContext<KernelTensor> *const context) {
  if (!enable_input_optimize_) {
    return;
  }
  for (const auto &parameter_index : parameter_indexs_) {
    auto output_position = parameter_index.first;
    if (output_position >= device_contexts_.size()) {
      MS_LOG(ERROR) << "The output position is of range: " << output_position;
      return;
    }
    auto front_node_with_idx = parameter_index.second.first;
    auto output_node = front_node_with_idx.first;
    MS_EXCEPTION_IF_NULL(output_node);
    auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
    MS_EXCEPTION_IF_NULL(graph_parameter_store);
    if (!graph_parameter_store->IsFrontNodeInStore(output_node.get())) {
      MS_LOG(EXCEPTION) << "Output node " << output_node->DebugString() << " is not in graph parameter store.";
    }
    auto outer_idx = graph_parameter_store->GetFrontNodeToIndex(output_node.get());
    auto tensor = graph_parameter_store->FetchTensor(outer_idx, {output_node, front_node_with_idx.second});
    MS_EXCEPTION_IF_NULL(tensor);

    const auto new_tensor = tensor::from_spec(tensor->data_type(), tensor->shape(), device::DeviceType::kNone);
    auto parameter_kernel_tensor = FetchParameter(parameter_index.second, GetAID());
    MS_EXCEPTION_IF_NULL(parameter_kernel_tensor);
    auto device_tensor = parameter_kernel_tensor->device_address().get();
    MS_EXCEPTION_IF_NULL(device_tensor);
    MS_EXCEPTION_IF_NULL(device_contexts_[output_position]);
    if (device_contexts_[output_position]->GetDeviceType() != device_tensor->GetDeviceType()) {
      device_contexts_[output_position] = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {device_tensor->GetDeviceType(), device_tensor->device_id()});
    }
    // Create the device address and put it into host tensor.
    const auto &iter = old_to_new_kernel_tensor_.find(parameter_kernel_tensor.get());
    if (iter != old_to_new_kernel_tensor_.end()) {
      MS_EXCEPTION_IF_NULL(iter->second);
      new_tensor->set_device_address(iter->second->device_address());
      new_tensor->set_storage_info(iter->second->tensor_storage_info());
      new_tensor->set_format(iter->second->format());
    } else {
      auto kernel_tensor = AnfAlgo::CreateKernelTensor(
        nullptr, device_tensor->GetSize(), parameter_kernel_tensor->format(), parameter_kernel_tensor->dtype_id(),
        parameter_kernel_tensor->GetShapeVector(),
        device::GetDeviceNameByType(device_contexts_[output_position]->device_context_key().device_type_),
        device_contexts_[output_position]->device_context_key().device_id_);
      kernel_tensor->SetType(parameter_kernel_tensor->GetType());
      kernel_tensor->SetShape(parameter_kernel_tensor->GetShape());
      kernel_tensor->set_stream_id(device_tensor->stream_id());
      kernel_tensor->set_tensor_storage_info(parameter_kernel_tensor->tensor_storage_info());
      auto tensor_device_address = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(tensor_device_address);
      if (parameter_kernel_tensor->tensor_storage_info() != nullptr) {
        new_tensor->set_shape(parameter_kernel_tensor->GetShapeVector());
      }
      // SetShape will calculate a default size by host shape, need to set real device size for special format.
      kernel_tensor->set_size(device_tensor->GetSize());
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Create kernel tensor:" << kernel_tensor->ToString() << " output node:" << output_node->fullname_with_scope()
        << " output position:" << output_position << ", origin output device tensor: " << device_tensor;
      old_to_new_kernel_tensor_[parameter_kernel_tensor.get()] = kernel_tensor;
      new_tensor->set_device_address(tensor_device_address);
      new_tensor->set_storage_info(kernel_tensor->tensor_storage_info());
    }
    if (parameter_kernel_tensor->tensor_storage_info() != nullptr) {
      new_tensor->set_contiguous_callback(
        [this](const tensor::TensorPtr &self) -> DeviceAddressPtr { return MakeTensorContiguousCallback(self); });
    }
    outputs_[output_position] = new_tensor;
    if (outputs_[output_position] == nullptr) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, "Create output tensor failed.");
    }
    if (!flatten_stub_nodes_.empty()) {
      const auto &stub_node = flatten_stub_nodes_.at(output_position);
      MS_EXCEPTION_IF_NULL(stub_node);
      outputs_[output_position]->set_need_pipeline_sync(true);
      stub_node->SetValue(outputs_[output_position]);
    }
    output_kernel_tensors_[output_position] = parameter_kernel_tensor;
    output_nodes_[output_position] = front_node_with_idx;
  }
}

void OutputActor::RunOpControl(AID *const, OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kOutputProcess, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);
  ++current_count_;
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op control and current count:" << current_count_;

  // The last loop.
  if (loop_count_ == current_count_) {
    if (current_outputs_num_ + device_tensor_store_keys_.size() + parameter_indexs_.size() != outputs_num_) {
      std::string error_info = "The outputs num is wrong, the total outputs num: " + std::to_string(outputs_num_) +
                               ", the current outputs num: " + std::to_string(current_outputs_num_) +
                               ", the device tensor store num: " + std::to_string(device_tensor_store_keys_.size()) +
                               ", the parameter index num: " + std::to_string(parameter_indexs_.size());
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    // Because device tensor store can't send data, so fetch the output result of device tensor store in running end.
    for (const auto &device_tensor_store_key : device_tensor_store_keys_) {
      if (device_tensor_store_key.first >= outputs_.size()) {
        std::stringstream ofs;
        ofs << "Invalid device tensor store index:" << device_tensor_store_key.first
            << " output device tensor size:" << outputs_.size() << " for actor:" << GetAID();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), ofs.str());
      }
      if (device_tensor_store_key.second != nullptr && device_tensor_store_key.second->isa<ValueNode>()) {
        const auto &value = device_tensor_store_key.second->cast<ValueNodePtr>()->value();
        MS_EXCEPTION_IF_NULL(value);
        if (value->isa<tensor::Tensor>()) {
          outputs_[device_tensor_store_key.first] = value->cast<tensor::TensorPtr>();
          if (!flatten_stub_nodes_.empty()) {
            const auto &stub_node = flatten_stub_nodes_.at(device_tensor_store_key.first);
            MS_EXCEPTION_IF_NULL(stub_node);
            MS_LOG(DEBUG) << "Begin set tensor value for: " << device_tensor_store_key.second->fullname_with_scope()
                          << ", value: " << value->ToString();
            stub_node->SetValue(value);
          }
          continue;
        } else if (value->isa<Scalar>()) {
          outputs_[device_tensor_store_key.first] = ScalarToTensor(value->cast<ScalarPtr>());
          if (!flatten_stub_nodes_.empty()) {
            const auto &stub_node = flatten_stub_nodes_.at(device_tensor_store_key.first);
            MS_EXCEPTION_IF_NULL(stub_node);
            MS_LOG(DEBUG) << "Begin set scalar value for: " << device_tensor_store_key.second->fullname_with_scope()
                          << ", value: " << value->ToString();
            stub_node->SetValue(value);
          }
          continue;
        } else if (value->isa<StringImm>()) {
          MS_LOG(DEBUG) << "Output value node:" << device_tensor_store_key.second->DebugString();
          continue;
        } else {
          MS_LOG(DEBUG) << "Output value node:" << device_tensor_store_key.second->DebugString();
        }
      }
      outputs_[device_tensor_store_key.first] =
        CreateOutputTensor(device_tensor_store_key.second, 0, device_tensor_store_key.first, context);
      if (outputs_[device_tensor_store_key.first] == nullptr) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, "Create output tensor failed.");
      }
      if (!flatten_stub_nodes_.empty()) {
        const auto &stub_node = flatten_stub_nodes_.at(device_tensor_store_key.first);
        MS_EXCEPTION_IF_NULL(stub_node);
        outputs_[device_tensor_store_key.first]->set_need_pipeline_sync(true);
        stub_node->SetValue(outputs_[device_tensor_store_key.first]);
      }
      output_nodes_[device_tensor_store_key.first] = {device_tensor_store_key.second, 0};
      const auto &kernel_tensor = AnfAlgo::GetOutputKernelTensor(device_tensor_store_key.second, 0, false);
      output_kernel_tensors_[device_tensor_store_key.first] = kernel_tensor;
    }

    FetchParameterInput(context);

    current_outputs_num_ = 0;
    current_count_ = 0;
    SET_OPCONTEXT_SUCCESS_RET((*context));
  }

  // Maybe the output node is the dynamic shape, need free the output node address to alloc new address by the new shape
  // and size in the next step running.
  FreeOutputNodeMem();

  // Free summary node input after usage.
  FreeSummaryNodeMem();

  // Send control arrow to trigger next step running.
  auto from_aid = const_cast<AID *>(&GetAID());
  for (auto &output_control : output_control_arrows_) {
    MS_EXCEPTION_IF_NULL(output_control);
    ActorDispatcher::Send(output_control->to_op_id_, &OpRTActor::RunOpControl, from_aid, context);
  }
}

void OutputActor::RunOpData(OpData<KernelTensor> *const input_data, OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kOutputProcess, GetAID().Name());
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  MS_EXCEPTION_IF_NULL(context);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
    << "Actor(" << GetAID().Name() << ") receive the input op data and output position:" << input_data->index_
    << " output node:"
    << (input_data->data_->device_address()->GetNodeIndex().first == nullptr
          ? "null"
          : input_data->data_->device_address()->GetNodeIndex().first->DebugString())
    << " index:" << input_data->data_->device_address()->GetNodeIndex().second
    << " input data:" << input_data->data_->ToString();
  auto output_position = IntToSize(input_data->index_);
  if (output_position >= outputs_.size()) {
    std::stringstream ofs;
    ofs << "Invalid output position:" << output_position << " output device tensor size:" << outputs_.size()
        << " for actor:" << GetAID();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), ofs.str());
  }
  // Save the output nodes and output device tensors.
  auto node_with_index = input_data->data_->device_address()->GetNodeIndex();
  MS_EXCEPTION_IF_NULL(node_with_index.first);
  output_nodes_[output_position] = node_with_index;
  output_kernel_tensors_[output_position] = input_data->data_;

  // Collect the output result in the last loop which is represented by "loop_count_ - current_count_ == 1".
  if (loop_count_ - current_count_ != 1) {
    return;
  }

  auto tensor = CreateOutputTensor(node_with_index.first, node_with_index.second, output_position, context,
                                   output_kernel_tensors_[output_position]);
  if (tensor == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, "Create output tensor failed.");
  }
  outputs_[output_position] = tensor;
  if (!flatten_stub_nodes_.empty()) {
    const auto &stub_node = flatten_stub_nodes_.at(output_position);
    MS_EXCEPTION_IF_NULL(stub_node);
    tensor->set_need_pipeline_sync(true);
    if (!stub_node->isa<stub::StringNode>()) {
      stub_node->SetValue(tensor);
    }
  }
  current_outputs_num_++;
}

TensorPtr OutputActor::CreateOutputTensor(const AnfNodePtr &output_node, size_t output_index, size_t output_position,
                                          OpContext<KernelTensor> *const context,
                                          const KernelTensorPtr &old_kernel_tensor) {
  MS_EXCEPTION_IF_NULL(output_node);
  bool is_dynamic_shape_output =
    common::AnfAlgo::IsDynamicShape(output_node) || common::AnfAlgo::IsDynamicSequence(output_node);
  // Wait pipeline for dynamic shape output node if need.
  // Note: In dynamic shape case, when actor thread number <= 3, maybe only enable async launch kernel, we should check
  // weather enable async launch kernel rather than whether enable multi pipeline here.
  if (ActorDispatcher::enable_async_launch_kernel() && is_dynamic_shape_output) {
    // Need wait all kernel launch task finish to update output shape and size for computed depend kernel.
    bool is_computed_depend_kernel = false;
    if (!output_node->isa<CNode>()) {
      is_computed_depend_kernel = false;
    } else {
      auto kernel_mod = AnfAlgo::GetKernelMod(output_node);
      if (kernel_mod && kernel_mod->IsNeedUpdateOutputShapeAndSize()) {
        is_computed_depend_kernel = true;
      }
    }

    if (!WaitRuntimePipelineFinish(context, GetAID().Name(), is_computed_depend_kernel)) {
      MS_LOG(INFO) << "Run graph failed and please check error log.";
      return nullptr;
    }
  }

  auto output_kernel_tensor =
    old_kernel_tensor == nullptr ? AnfAlgo::GetOutputKernelTensor(output_node, output_index, false) : old_kernel_tensor;
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
    << "Create output tensor, output node: " << output_node->fullname_with_scope()
    << " debug string:" << output_node->DebugString() << ", output index: " << output_index
    << ", output position: " << output_position << ", output kernel tensor: " << output_kernel_tensor->ToString();

  // For dynamice sequence output, the Type(Tuple) hasn't been re-inferred, only Shape has been re-inferred, need update
  // real Type of Tuple into kernel tensor to restore the tuple output.
  UpdateDynamicSequenceType(output_node, output_kernel_tensor);

  // If output is an empty sequence return an empty tensor directly.
  const auto &output_shape = output_kernel_tensor->GetShape();
  if (output_shape != nullptr && output_shape->isa<abstract::SequenceShape>() &&
      output_shape->cast<abstract::SequenceShapePtr>()->size() == 0) {
    ShapeVector shape_vector = {0};
    TypeId type_id = (output_kernel_tensor->dtype_id() == TypeId::kTypeUnknown ? TypeId::kNumberTypeInt64
                                                                               : output_kernel_tensor->dtype_id());
    const auto &tensor = tensor::from_spec(type_id, shape_vector, device::DeviceType::kNone);
    tensor->set_base_shape(output_shape);
    if (output_position < output_kernel_tensors_.size() && output_kernel_tensors_[output_position] &&
        output_kernel_tensors_[output_position]->user_data() != nullptr && output_position < device_contexts_.size() &&
        device_contexts_[output_position] != nullptr) {
      auto shape = std::make_shared<abstract::TupleShape>();
      auto type = std::make_shared<Tuple>();
      auto temp_device_address =
        std::make_shared<device::DeviceAddress>(device_contexts_[output_position]->GetDeviceType());
      auto kernel_tensor = std::make_shared<kernel::KernelTensor>(temp_device_address, nullptr, shape, type, nullptr);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString();
      auto tensor_device_address = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(tensor_device_address);
      tensor->set_device_address(tensor_device_address);
      tensor->set_storage_info(kernel_tensor->tensor_storage_info());
      tensor->set_format(kernel_tensor->format());
    }
    return tensor;
  }

  const auto &abstract = AnfAlgo::GetNodeAbstractByIndex(output_node, output_index);
  MS_EXCEPTION_IF_NULL(abstract);

  if (!flatten_stub_nodes_.empty() && is_dynamic_shape_output) {
    const auto &clone_abs = abstract->Clone();
    MS_EXCEPTION_IF_NULL(clone_abs);
    clone_abs->set_shape(output_kernel_tensor->GetShape()->Clone());

    const auto &stub_node = flatten_stub_nodes_.at(output_position);
    MS_EXCEPTION_IF_NULL(stub_node);
    if (!stub_node->SetAbstract(clone_abs)) {
      MS_LOG(EXCEPTION) << "Set abstract for stub node failed, output position: " << output_position
                        << ", output node: " << output_node->fullname_with_scope();
    }
  }

  // Create host tensor, the output tensor should use the infer type, it will be handed correctly by tensor data sync
  // when infer type is not equal to device type.
  auto type_id = common::AnfAlgo::GetOutputInferDataType(output_node, output_index);
  const auto &shape = output_kernel_tensor->GetShapeVector();
  auto tensor = tensor::from_spec(type_id, shape, device::DeviceType::kNone);
  MS_EXCEPTION_IF_NULL(tensor);
  // Set tensor base shape for restoring the tuple output when output node is dynamic sequence.
  if (common::AnfAlgo::IsDynamicSequence(output_node)) {
    tensor->set_base_shape(output_kernel_tensor->GetShape());
  }

  if (output_position >= device_contexts_.size()) {
    MS_LOG(ERROR) << "The output position is of range: " << output_position;
    return nullptr;
  }
  auto &device_context = device_contexts_[output_position];
  MS_EXCEPTION_IF_NULL(device_context);
  KernelTensorPtr node_kernel_tensor = old_kernel_tensor;
  if (node_kernel_tensor == nullptr) {
    node_kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, output_index, false);
  }
  MS_EXCEPTION_IF_NULL(node_kernel_tensor);
  const auto &device_tensor = node_kernel_tensor->device_address().get();
  MS_EXCEPTION_IF_NULL(device_tensor);
  device_tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(output_node, output_index));
  if (device_context->GetDeviceType() != device_tensor->GetDeviceType()) {
    auto old_device_context = device_context;
    device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_tensor->GetDeviceType(), device_tensor->device_id()});
    MS_LOG(INFO) << "Update device context from:" << old_device_context->GetDeviceType()
                 << " to:" << device_context->GetDeviceType();
  }

  // Create the device address and put it into host tensor.
  const auto &iter = old_to_new_kernel_tensor_.find(node_kernel_tensor.get());
  if (iter != old_to_new_kernel_tensor_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    tensor->set_device_address(iter->second->device_address());
    tensor->set_storage_info(iter->second->tensor_storage_info());
  } else {
    auto kernel_tensor =
      AnfAlgo::CreateKernelTensor(nullptr, device_tensor->GetSize(), node_kernel_tensor->format(),
                                  node_kernel_tensor->dtype_id(), node_kernel_tensor->GetShapeVector(),
                                  device::GetDeviceNameByType(device_context->device_context_key().device_type_),
                                  device_context->device_context_key().device_id_);
    kernel_tensor->SetType(output_kernel_tensor->GetType());
    kernel_tensor->SetShape(output_kernel_tensor->GetShape());
    kernel_tensor->set_tensor_storage_info(output_kernel_tensor->tensor_storage_info());
    kernel_tensor->set_stream_id(device_tensor->stream_id());
    // SetShape will calculate a default size by host shape, need to set real device size for special format.
    kernel_tensor->set_size(device_tensor->GetSize());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Create kernel tensor:" << kernel_tensor->ToString() << " output node:" << output_node->fullname_with_scope()
      << " output index:" << output_index << " output position:" << output_position
      << ", origin output device tensor: " << device_tensor;
    tensor->set_device_address(kernel_tensor->device_address());
    tensor->set_storage_info(kernel_tensor->tensor_storage_info());
    old_to_new_kernel_tensor_[node_kernel_tensor.get()] = kernel_tensor;
  }

  if (output_kernel_tensor->tensor_storage_info()) {
    tensor->set_contiguous_callback(
      [this](const tensor::TensorPtr &self) -> DeviceAddressPtr { return MakeTensorContiguousCallback(self); });
  }
  return tensor;
}

void OutputActor::SetStubOutput(const stub::StubNodePtr &stub_output) {
  MS_EXCEPTION_IF_NULL(stub_output);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kOutputProcess,
                                     "FlattenStubNode");
  stub::FlattenStubNode(stub_output, &flatten_stub_nodes_);
  if (flatten_stub_nodes_.size() != outputs_num()) {
    MS_LOG(EXCEPTION) << "Invalid output number for flatten stub output nodes, expect: " << outputs_num()
                      << ", but got: " << flatten_stub_nodes_.size();
  }
}

namespace {
void HandleEmptySequenceOutput(KernelTensor *const kernel_tensor, const tensor::TensorPtr &tensor, size_t index,
                               const std::string &actor_name) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address().get();
  MS_EXCEPTION_IF_NULL(device_tensor);
  MS_EXCEPTION_IF_NULL(tensor);
  MS_LOG(DEBUG) << "Empty sequence shape for input tensor index:" << index;
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_tensor->GetDeviceType(), device_tensor->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Free kernel tensor:" << kernel_tensor << " for actor:" << actor_name;
  MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(kernel_tensor, device_context, actor_name);
  if (kernel_tensor->user_data() != nullptr && tensor != nullptr) {
    tensor->CloneUserData(*(kernel_tensor->user_data()));
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Set user data from device address:" << device_tensor << " to tensor:" << tensor;
  }
}

bool IsEmptySequence(const tensor::TensorPtr &tensor) {
  return tensor->base_shape_ptr() != nullptr && tensor->base_shape_ptr()->isa<abstract::SequenceShape>() &&
         tensor->base_shape_ptr()->cast<abstract::SequenceShapePtr>()->size() == 0;
}

void HandleSameOutput(const KernelTensorPtr &kernel_tensor, const TensorPtr &tensor,
                      device::DeviceContext *const real_device_context, const AID &aid) {
  // The outputs have the same output node but different tensors, user data needs to be set
  // on the other same nodes.
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  MS_EXCEPTION_IF_NULL(tensor);
  if (kernel_tensor->user_data()) {
    tensor->CloneUserData(*(kernel_tensor->user_data()));
  }
  // For trace memory, the new ref count will always be 0.
  if (!ActorDispatcher::enable_use_trace_memory() || kernel_tensor->new_ref_count() > 0) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Free same kernel tensor:" << kernel_tensor << " for actor:" << aid;
    MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(kernel_tensor.get(), real_device_context, aid.Name());
  }
}

void SetDeviceAddressForDiffDtype(const tensor::TensorPtr &tensor, const KernelTensorPtr &kernel_tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (kernel_tensor->GetDeviceType() != device::DeviceType::kCPU) {
    MS_LOG(EXCEPTION) << "Not support trans data type for device type:" << kernel_tensor->GetDeviceType()
                      << " kernel tensor:" << kernel_tensor->ToString() << " tensor:" << tensor->ToString();
  }
  auto new_tensor = tensor::from_spec(tensor->data_type(), tensor->shape(), device::DeviceType::kCPU);
  tensor->set_device_address(new_tensor->device_address());
  tensor->set_storage_info(nullptr);
  MS_LOG(DEBUG) << "Copy output from kernel tensor:" << kernel_tensor->ToString()
                << " to tensor:" << tensor->ToString();
  if (!SyncCopy(tensor, kernel_tensor.get(), kernel_tensor->stream_id())) {
    MS_LOG(EXCEPTION) << "Failed to copy data from kernel tensor:" << kernel_tensor->ToString()
                      << " to tensor:" << tensor->ToString();
  }
}
}  // namespace

void OutputActor::HandleOutput() {
  auto repeat_index = GetRepeatDeviceAddressIndexPair(output_kernel_tensors_);
  for (size_t i = 0; i < output_nodes_.size(); ++i) {
    auto &output_node = output_nodes_[i].first;
    if (i >= output_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid index:" << i << " current:" << output_kernel_tensors_.size();
    }
    auto kernel_tensor = output_kernel_tensors_[i];
    if (kernel_tensor == nullptr) {
      MS_LOG(INFO) << "The kernel tensor is nullptr, need check whether affect the result.";
      continue;
    }
    auto device_tensor = kernel_tensor->device_address();
    if (output_node == nullptr || device_tensor == nullptr) {
      MS_LOG(INFO) << "The output node or device tensor is nullptr, need check whether affect the result.";
      continue;
    }

    auto &tensor = outputs_[i];
    MS_EXCEPTION_IF_NULL(tensor);
    if (IsEmptySequence(tensor)) {
      HandleEmptySequenceOutput(kernel_tensor.get(), tensor, i, GetAID().Name());
      continue;
    }
    auto tensor_device_address = tensor->device_address();
    MS_EXCEPTION_IF_NULL(tensor_device_address);
    // Update tensor device address by device tensor of output node.
    auto node_with_index = device_tensor->GetNodeIndex();
    tensor_device_address->SetNodeIndex(node_with_index.first, node_with_index.second);
    tensor_device_address->set_from_persistent_mem(device_tensor->from_persistent_mem());

    const auto &real_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_tensor->GetDeviceType(), device_tensor->device_id()});
    MS_EXCEPTION_IF_NULL(real_device_context);
    // The outputs may have the same output node, so need skip when the node has been done.
    if (tensor_device_address->GetPtr() != nullptr) {
      HandleSameOutput(kernel_tensor, tensor, real_device_context, GetAID());
      continue;
    }

    auto device_context = device_contexts_[i];
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
    if (repeat_index.find(i) != repeat_index.end() && i > repeat_index[i] && outputs_[repeat_index[i]] != nullptr) {
      if (kernel_tensor->user_data()) {
        tensor->CloneUserData(*(kernel_tensor->user_data()));
      }
      const auto &src_address = outputs_[repeat_index[i]]->device_address();
      MS_EXCEPTION_IF_NULL(src_address);
      tensor_device_address->set_device_pointer(src_address->device_pointer());
      tensor_device_address->SetDeviceType(src_address->GetDeviceType());
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Output actor share the same pointer:" << src_address->device_pointer()
        << " between device address:" << tensor_device_address->ToString() << " and:" << src_address->ToString()
        << " kernel tensor:" << kernel_tensor->ToString();
      continue;
    }
    // If the output node whose output address ptr can't be changed, then alloc the new device memory and copy the data:
    if (IsOutputAddressPersisted(kernel_tensor, output_nodes_[i])) {
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, GetAID().Name(), memory::mem_pool::MemType::kOther,
                                                     tensor_device_address->GetSize(), tensor_device_address.get());
      if (!device_context->device_res_manager_->AllocateMemory(tensor_device_address.get(), kDefaultStreamIndex)) {
        MS_LOG_WITH_NODE(EXCEPTION, output_node)
          << "Device(id:" << device_context->device_context_key().device_id_
          << ") memory isn't enough and alloc failed in output actor, kernel name: "
          << output_node->fullname_with_scope() << ", alloc size: " << tensor_device_address->GetSize() << "B.";
      }
      SyncOutputFromTensor(tensor, kernel_tensor, output_node);
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Copy graph output from device address:" << device_tensor->ToString()
        << " to:" << tensor_device_address->ToString();
    } else {
      if (tensor->data_type() != kernel_tensor->dtype_id()) {
        if (kernel_tensor->GetDeviceType() != device::DeviceType::kCPU) {
          MS_LOG(EXCEPTION) << "Invalid output type:" << tensor->data_type()
                            << " for kernel tensor:" << kernel_tensor->ToString();
        }
        SetDeviceAddressForDiffDtype(tensor, kernel_tensor);
        MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
          << "Tran data type from kernel tensor:" << kernel_tensor->ToString() << " to tensor:" << tensor->ToString();
        if (kernel_tensor->user_data()) {
          tensor->CloneUserData(*(kernel_tensor->user_data()));
        }
        MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
          << "Free kernel tensor:" << kernel_tensor << " for diff type in actor:" << GetAID();
        MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(kernel_tensor.get(), real_device_context,
                                                                GetAID().Name());
        continue;
      }
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Swap ptr:" << device_tensor->GetPtr() << " from kernel tensor:" << kernel_tensor->ToString()
        << " device type:" << device_tensor->GetDeviceType() << " to :" << tensor_device_address->ToString()
        << " tensor:" << tensor->ToString() << " tensor format:" << tensor->format();
      // Move the device ptr from device_tensor to tensor_device_address.
      device_tensor->Swap(tensor_device_address.get());
      tensor->set_format(kernel_tensor->format());
      if (kernel_tensor->user_data()) {
        tensor->CloneUserData(*(kernel_tensor->user_data()));
      }
    }
    // If enable kernel launch capture, the kernel output as graph output will be captured and can not release device
    // memory.
    if (GraphCaptureManager::GetInstance().GetEnableGraphCapture() && GraphCaptureManager::GetInstance().InReplay()) {
      GraphCaptureManager::GetInstance().SetInReplay(false);
      kernel_tensor->set_device_ptr(nullptr);
    } else {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Free kernel tensor:" << kernel_tensor << " for actor:" << GetAID();
      MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(kernel_tensor.get(), real_device_context,
                                                              GetAID().Name());
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
      MarkTensorAsOutput, GetAID().Name(), device::GetDeviceNameByType(device_tensor->GetDeviceType()),
      device_tensor->GetPtr(), kernel_tensor->dtype_id(), kernel_tensor->GetShapeVector(),
      kernel_tensor->tensor_storage_info());
  }
}

void OutputActor::UpdateOutputDeviceAddress() {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kOutputProcess, "UpdateOutputDeviceAddress");
  // In the running end, when the device ptr of graph output node is set into host tensor, the graph output node
  // need be set new device ptr, to avoid that the device ptr context of host tensor be rewritten in the next
  // step or next loop. But the graph output nodes corresponding to device tensor store need to be skipped, because
  // they are fixed addresses and persistent.
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), "UpdateOutputDeviceAddress", "");
  HandleOutput();

  // output types used for construct outputs.
  RecordOutputTypes(&output_types_, output_kernel_tensors_, output_nodes_.size());
  old_to_new_kernel_tensor_.clear();
  output_nodes_.clear();
  output_nodes_.resize(outputs_num_);
  output_kernel_tensors_.clear();
  output_kernel_tensors_.resize(outputs_num_);
  if (!flatten_stub_nodes_.empty()) {
    flatten_stub_nodes_.clear();
  }
}
}  // namespace runtime
}  // namespace mindspore
