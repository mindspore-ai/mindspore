/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/log_adapter.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "include/backend/distributed/collective/collective_manager.h"

namespace mindspore {
namespace runtime {
using distributed::collective::CollectiveManager;
using distributed::recovery::RecoveryContext;

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
  if (outputs_.size() != output_device_tensors_.size()) {
    MS_LOG(EXCEPTION) << "The output device tensors number is wrong.";
  }

  // Set the number of actor running dependent messages.
  running_dependent_msg_num_ = SizeToInt(outputs_num_ - device_tensor_store_keys_.size());
}

void OutputActor::FreeOutputNodeMem() {
  for (size_t i = 0; i < output_nodes_.size(); ++i) {
    auto &output_node = output_nodes_[i].first;
    auto &output_device_tensor = output_device_tensors_[i];
    // The output_device_tensor may be repeated.
    if ((output_node == nullptr) || (output_device_tensor == nullptr) || (output_device_tensor->GetPtr() == nullptr)) {
      continue;
    }
    if (!IsOutputAddressPersisted(output_device_tensor, output_nodes_[i])) {
      FreeMemoryByDeviceContext(output_device_tensor, device_contexts_[i]);
    }
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
    if ((output_device_addr == nullptr) || (output_device_addr->GetPtr() == nullptr)) {
      continue;
    }
    if (!IsOutputAddressPersisted(output_device_addr.get(), summary_nodes_[i])) {
      FreeMemoryByDeviceContext(output_device_addr.get(), nullptr);
    }
  }
}

void OutputActor::ClearOutputCache() {
  output_node_to_tensor_device_address_.clear();
  outputs_.clear();
  outputs_.resize(outputs_num_);
  output_nodes_.clear();
  output_nodes_.resize(outputs_num_);
  output_device_tensors_.clear();
  output_device_tensors_.resize(outputs_num_);

  current_outputs_num_ = 0;
  current_count_ = 0;
}

void OutputActor::RunOpControl(AID *const, OpContext<DeviceTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kOutputProcess, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);
  ++current_count_;
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op control and current count:" << current_count_;

  // Trigger disaster recovery and return empty output.
  if (RecoveryContext::GetInstance()->enable_recovery() && CollectiveManager::instance()->need_reinit()) {
    FreeOutputNodeMem();
    ClearOutputCache();
    SET_OPCONTEXT_SUCCESS_RET((*context));
  }

  // The last loop.
  if (loop_count_ == current_count_) {
    if (current_outputs_num_ + device_tensor_store_keys_.size() != outputs_num_) {
      std::string error_info = "The outputs num is wrong, the total outputs num: " + std::to_string(outputs_num_) +
                               ", the current outputs num: " + std::to_string(current_outputs_num_) +
                               ", the device tensor store num: " + std::to_string(device_tensor_store_keys_.size());
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    // Because device tensor store can't send data, so fetch the output result of device tensor store in running end.
    for (const auto &device_tensor_store_key : device_tensor_store_keys_) {
      if (device_tensor_store_key.first >= outputs_.size()) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The input index is of range.");
      }
      if (device_tensor_store_key.second != nullptr && device_tensor_store_key.second->isa<ValueNode>()) {
        const auto &value = device_tensor_store_key.second->cast<ValueNodePtr>()->value();
        MS_EXCEPTION_IF_NULL(value);
        if (value->isa<tensor::Tensor>()) {
          outputs_[device_tensor_store_key.first] = value->cast<tensor::TensorPtr>();
          continue;
        } else if (value->isa<Scalar>()) {
          outputs_[device_tensor_store_key.first] = ScalarToTensor(value->cast<ScalarPtr>());
          continue;
        } else {
          MS_LOG(DEBUG) << "Output value node:" << device_tensor_store_key.second->DebugString();
        }
      }
      outputs_[device_tensor_store_key.first] =
        CreateOutputTensor(device_tensor_store_key.second, 0, device_tensor_store_key.first);
      if (outputs_[device_tensor_store_key.first] == nullptr) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, "Create output tensor failed.");
      }
      output_nodes_[device_tensor_store_key.first] = {device_tensor_store_key.second, 0};
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(device_tensor_store_key.second, 0, false);
      output_device_tensors_[device_tensor_store_key.first] = device_tensor.get();
    }

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
    ActorDispatcher::Send(output_control->to_op_id_, &OpActor::RunOpControl, from_aid, context);
  }
}

void OutputActor::RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kOutputProcess, GetAID().Name());
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  MS_EXCEPTION_IF_NULL(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name()
                << ") receive the input op data and output position:" << input_data->index_
                << " device tensor:" << input_data->data_ << " ptr:" << input_data->data_->GetPtr();
  auto output_position = IntToSize(input_data->index_);
  if (output_position >= outputs_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The input index is of range.");
  }
  // Save the output nodes and output device tensors.
  auto node_with_index = input_data->data_->GetNodeIndex();
  MS_EXCEPTION_IF_NULL(node_with_index.first);
  output_nodes_[output_position] = node_with_index;
  output_device_tensors_[output_position] = input_data->data_;

  // Collect the output result in the last loop which is represented by "loop_count_ - current_count_ == 1".
  if (loop_count_ - current_count_ != 1) {
    return;
  }

  auto tensor = CreateOutputTensor(node_with_index.first, node_with_index.second, output_position);
  if (tensor == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, "Create output tensor failed.");
  }
  // The persisted address can not released by the tensor print.
  if (!IsOutputAddressPersisted(input_data->data_, node_with_index)) {
    tensor->set_need_release_device_mem(true);
  }
  outputs_[output_position] = tensor;
  current_outputs_num_++;
}

TensorPtr OutputActor::CreateOutputTensor(const AnfNodePtr &output_node, size_t output_index, size_t output_position) {
  MS_EXCEPTION_IF_NULL(output_node);
  // Note: In dynamic shape case, when actor thread number <= 3, maybe only enable async launch kernel, we should check
  // weather enable async launch kernel rather than whether enable multi pipeline here.
  if (ActorDispatcher::enable_async_launch_kernel()) {
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

    WaitRuntimePipelineFinish(is_computed_depend_kernel);
  }

  const auto &output_kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, output_index);
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);
  MS_LOG(DEBUG) << "Create output tensor, output node: " << output_node->fullname_with_scope()
                << " debug string:" << output_node->DebugString() << ", output index: " << output_index
                << ", output position: " << output_position
                << ", output kernel tensor: " << output_kernel_tensor->ToString();

  // For dynamice sequence output, the Type(Tuple) hasn't been re-inferred, only Shape has been re-inferred, need update
  // real Type of Tuple into kernel tensor to restore the tuple output.
  UpdateDynamicSequenceType(output_node, output_kernel_tensor);

  // If output is an empty sequence return an empty tensor directly.
  const auto &output_shape = output_kernel_tensor->GetShape();
  if (output_shape != nullptr && output_shape->isa<abstract::SequenceShape>() &&
      output_shape->cast<abstract::SequenceShapePtr>()->size() == 0) {
    ShapeVector shape = {0};
    TypeId type_id = (output_kernel_tensor->dtype_id() == TypeId::kTypeUnknown ? TypeId::kNumberTypeInt64
                                                                               : output_kernel_tensor->dtype_id());
    const auto &tensor = std::make_shared<tensor::Tensor>(type_id, shape);
    tensor->set_base_shape(output_shape);
    return tensor;
  }

  const auto &abstract = AnfAlgo::GetNodeAbstractByIndex(output_node, output_index);
  if (abstract != nullptr && abstract->isa<abstract::AbstractMapTensor>()) {
    return AnfAlgo::CreateMapTensor(output_node, output_index);
  }
  // Create host tensor, the output tensor should use the infer type, it will be handed correctly by tensor data sync
  // when infer type is not equal to device type.
  auto type_id = common::AnfAlgo::GetOutputInferDataType(output_node, output_index);
  const auto &shape = output_kernel_tensor->GetShapeVector();
  auto tensor = std::make_shared<tensor::Tensor>(type_id, shape);
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
  const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(output_node, output_index, false);
  MS_EXCEPTION_IF_NULL(device_tensor);
  device_tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(output_node, output_index));
  if (device_context->GetDeviceType() != device_tensor->GetDeviceType()) {
    auto old_device_context = device_context;
    device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_tensor->device_name(), device_tensor->device_id()});
    MS_LOG(INFO) << "Update device context from:" << old_device_context->GetDeviceType()
                 << " to:" << device_context->GetDeviceType();
  }

  // Create the device address and put it into host tensor.
  if (output_node_to_tensor_device_address_.count({output_node, output_index}) > 0) {
    tensor->set_device_address(output_node_to_tensor_device_address_[{output_node, output_index}]);
  } else {
    auto kernel_tensor = std::make_shared<kernel::KernelTensor>(
      nullptr, device_tensor->GetSize(), kernel::GetFormatFromStrToEnum(device_tensor->format()),
      device_tensor->type_id(), device_tensor->host_shape(), device_context->device_context_key().device_name_,
      device_context->device_context_key().device_id_);
    kernel_tensor->SetType(output_kernel_tensor->GetType());
    kernel_tensor->SetShape(output_kernel_tensor->GetShape());
    kernel_tensor->set_stream_id(device_tensor->stream_id());
    auto tensor_device_address = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
    MS_EXCEPTION_IF_NULL(tensor_device_address);
    MS_LOG(DEBUG) << "Create device tensor:" << tensor_device_address << " type:" << tensor_device_address->type_id()
                  << " output node:" << output_node->fullname_with_scope() << " output index:" << output_index
                  << " output position:" << output_position;
    tensor->set_device_address(tensor_device_address);
    output_node_to_tensor_device_address_[{output_node, output_index}] = tensor_device_address;
  }
  return tensor;
}

void OutputActor::UpdateOutputDeviceAddress() {
  // In the running end, when the device ptr of graph output node is set into host tensor, the graph output node
  // need be set new device ptr, to avoid that the device ptr context of host tensor be rewritten in the next
  // step or next loop. But the graph output nodes corresponding to device tensor store need to be skipped, because
  // they are fixed addresses and persistent.
  for (size_t i = 0; i < output_nodes_.size(); ++i) {
    auto &output_node = output_nodes_[i].first;
    if (i >= output_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid index:" << i << " current:" << output_device_tensors_.size();
    }
    auto device_tensor = output_device_tensors_[i];
    if (output_node == nullptr || device_tensor == nullptr) {
      MS_LOG(INFO) << "The output node or device tensor is nullptr, need check whether affect the result.";
      continue;
    }

    auto &tensor = outputs_[i];
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->base_shape_ptr() != nullptr && tensor->base_shape_ptr()->isa<abstract::SequenceShape>() &&
        tensor->base_shape_ptr()->cast<abstract::SequenceShapePtr>()->size() == 0) {
      continue;
    }
    auto tensor_device_address = std::dynamic_pointer_cast<DeviceTensor>(tensor->device_address());
    MS_EXCEPTION_IF_NULL(tensor_device_address);
    // Update tensor device address by device tensor of output node.
    tensor_device_address->set_original_ref_count(SIZE_MAX);
    tensor_device_address->ResetRefCount();
    tensor_device_address->set_dynamic_ref_count(INT32_MAX);
    auto node_with_index = device_tensor->GetNodeIndex();
    tensor_device_address->SetNodeIndex(node_with_index.first, node_with_index.second);
    tensor_device_address->set_from_persistent_mem(device_tensor->from_persistent_mem());
    tensor_device_address->set_host_shape(tensor->shape());
    // The outputs may have the same output node, so need skip when the node has been done.
    if (tensor_device_address->GetPtr() != nullptr) {
      continue;
    }

    auto device_context = device_contexts_[i];
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
    const auto &swap_manager = device_context->device_res_manager_->swap_manager();
    if (swap_manager != nullptr) {
      swap_manager->AddSwappableTensor(tensor_device_address);
    }
    // If the output node whose output address ptr can't be changed, then alloc the new device memory and copy the data:
    if (IsOutputAddressPersisted(device_tensor, output_nodes_[i])) {
      device::DynamicMemAllocatorDebugInfo::SetDebugInfo(GetAID().Name(), device::AllocatorType::kOther);
      if (!device_context->device_res_manager_->AllocateMemory(tensor_device_address.get())) {
        MS_LOG(EXCEPTION) << "Device(id:" << device_context->device_context_key().device_id_
                          << ") memory isn't enough and alloc failed, kernel name: "
                          << output_node->fullname_with_scope() << ", alloc size: " << tensor_device_address->GetSize()
                          << "B.";
      }
      if (!tensor_device_address->SyncDeviceToDevice(device_tensor)) {
        MS_LOG(EXCEPTION) << "Sync device to device failed, device type: " << tensor_device_address->GetDeviceType()
                          << ", output node: " << output_node->fullname_with_scope();
      }
    } else {
      MS_LOG(DEBUG) << "Swap ptr:" << device_tensor->GetPtr() << " from device tensor:" << device_tensor
                    << " device type:" << device_tensor->GetDeviceType() << " to :" << tensor_device_address
                    << " device type:" << tensor_device_address->GetDeviceType();
      // Move the device ptr from device_tensor to tensor_device_address.
      device_tensor->Swap(tensor_device_address.get());
      tensor_device_address->set_user_data(device_tensor->user_data());
    }
  }

  output_node_to_tensor_device_address_.clear();
  output_nodes_.clear();
  output_nodes_.resize(outputs_num_);
  output_device_tensors_.clear();
  output_device_tensors_.resize(outputs_num_);
}
}  // namespace runtime
}  // namespace mindspore
