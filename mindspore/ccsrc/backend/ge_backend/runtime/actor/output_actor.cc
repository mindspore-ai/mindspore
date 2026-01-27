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

#include "backend/ge_backend/runtime/actor/output_actor.h"
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include "utils/convert_utils.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "ir/map_tensor.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

#include "ir/tensor_new.h"
namespace mindspore {
namespace ge_backend {
namespace runtime {
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
    // The output_device_tensor may be repeated.
    if ((output_node == nullptr) || (output_device_tensor == nullptr) || (output_device_tensor->GetPtr() == nullptr)) {
      continue;
    }
    if (!IsOutputAddressPersisted(output_kernel_tensor, output_nodes_[i])) {
      FreeMemoryByDeviceContext(output_device_tensor.get());
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
    auto output_kernel_tensor = AnfAlgo::GetOutputKernelTensor(summary_node, index, false);
    if (output_kernel_tensor == nullptr) {
      continue;
    }
    auto output_device_addr = output_kernel_tensor->device_address();
    if ((output_device_addr == nullptr) || (output_device_addr->GetPtr() == nullptr)) {
      continue;
    }
    if (!IsOutputAddressPersisted(output_kernel_tensor, summary_nodes_[i])) {
      FreeMemoryByDeviceContext(output_device_addr.get());
    }
  }
}

void OutputActor::ClearOutputCache() {
  output_node_to_tensor_device_address_.clear();
  output_node_to_tensor_storage_info_.clear();
  outputs_.clear();
  outputs_.resize(outputs_num_);
  output_nodes_.clear();
  output_nodes_.resize(outputs_num_);
  output_kernel_tensors_.clear();
  output_kernel_tensors_.resize(outputs_num_);

  current_outputs_num_ = 0;
  current_count_ = 0;
}

void OutputActor::RunOpControl(AID *const, OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kOutputProcess, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);
  ++current_count_;
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op control and current count:" << current_count_;

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
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name()
                << ") receive the input op data and output position:" << input_data->index_
                << " device tensor:" << input_data->data_ << " ptr:" << input_data->data_->device_address()->GetPtr()
                << " ref count:" << input_data->data_->ref_count()
                << " origin ref count:" << input_data->data_->original_ref_count()
                << " dynamic ref count:" << input_data->data_->dynamic_ref_count()
                << " from memory pool:" << input_data->data_->device_address()->from_mem_pool() << " output node:"
                << (input_data->data_->device_address()->GetNodeIndex().first == nullptr
                      ? "null"
                      : input_data->data_->device_address()->GetNodeIndex().first->DebugString())
                << " index:" << input_data->data_->device_address()->GetNodeIndex().second;
  auto output_position = IntToSize(input_data->index_);
  if (output_position >= outputs_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The input index is of range.");
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

  auto tensor = CreateOutputTensor(node_with_index.first, node_with_index.second, output_position, context);
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
                                          OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output_node);
  bool is_dynamic_shape_output =
    common::AnfAlgo::IsDynamicShape(output_node) || common::AnfAlgo::IsDynamicSequence(output_node);
  auto output_kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, output_index, false);
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
    const auto &tensor = tensor::from_spec(type_id, shape, device::DeviceType::kNone);
    tensor->set_base_shape(output_shape);
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

  if (output_position >= outputs_num_) {
    MS_LOG(ERROR) << "The output position is of range: " << output_position;
    return nullptr;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  const auto &device_tensor = output_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  device_tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(output_node, output_index));
  if (device::GetDeviceTypeByName(device_name) != device_tensor->GetDeviceType()) {
    MS_LOG(EXCEPTION) << "GE backend only support Ascend, but got "
                      << device::GetDeviceNameByType(device_tensor->GetDeviceType());
  }

  // Create the device address and put it into host tensor.
  if (output_node_to_tensor_device_address_.count({output_node, output_index}) > 0) {
    tensor->set_device_address(output_node_to_tensor_device_address_[{output_node, output_index}]);
    tensor->set_storage_info(output_node_to_tensor_storage_info_[{output_node, output_index}]);
  } else {
    auto kernel_tensor = AnfAlgo::CreateKernelTensor(nullptr, device_tensor->GetSize(), output_kernel_tensor->format(),
                                                     output_kernel_tensor->dtype_id(),
                                                     output_kernel_tensor->GetShapeVector(), device_name, device_id);
    kernel_tensor->SetType(output_kernel_tensor->GetType());
    kernel_tensor->SetShape(output_kernel_tensor->GetShape());
    kernel_tensor->set_stream_id(device_tensor->stream_id());
    // SetShape will calculate a default size by host shape, need to set real device size for special format.
    kernel_tensor->set_size(device_tensor->GetSize());
    auto tensor_device_address = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(tensor_device_address);
    MS_LOG(DEBUG) << "Create device tensor:" << tensor_device_address->ToString()
                  << " output node:" << output_node->fullname_with_scope() << " output index:" << output_index
                  << " output position:" << output_position
                  << ", origin output kernel tensor: " << output_kernel_tensor->ToString();
    tensor->set_device_address(tensor_device_address);
    tensor->set_storage_info(kernel_tensor->tensor_storage_info());
    output_node_to_tensor_device_address_[{output_node, output_index}] = tensor_device_address;
    output_node_to_tensor_storage_info_[{output_node, output_index}] = kernel_tensor->tensor_storage_info();
  }

  return tensor;
}

void OutputActor::SetStubOutput(const stub::StubNodePtr &stub_output) {
  MS_EXCEPTION_IF_NULL(stub_output);
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kOutputProcess, "FlattenStubNode");
  stub::FlattenStubNode(stub_output, &flatten_stub_nodes_);
  if (flatten_stub_nodes_.size() != outputs_num()) {
    MS_LOG(EXCEPTION) << "Invalid output number for flatten stub output nodes, expect: " << outputs_num()
                      << ", but got: " << flatten_stub_nodes_.size();
  }
}

void OutputActor::UpdateOutputDeviceAddress() {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kOutputProcess, "UpdateOutputDeviceAddress");
  // In the running end, when the device ptr of graph output node is set into host tensor, the graph output node
  // need be set new device ptr, to avoid that the device ptr context of host tensor be rewritten in the next
  // step or next loop. But the graph output nodes corresponding to device tensor store need to be skipped, because
  // they are fixed addresses and persistent.
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), "UpdateOutputDeviceAddress", "");

  auto repeat_index = GetRepeatDeviceAddressIndexPair(output_kernel_tensors_);
  for (size_t i = 0; i < output_nodes_.size(); ++i) {
    auto &output_node = output_nodes_[i].first;
    if (i >= output_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid index:" << i << " current:" << output_kernel_tensors_.size();
    }
    if (output_kernel_tensors_[i] == nullptr) {
      MS_LOG(INFO) << "The kernel tensor is nullptr, need check whether affect the result.";
      continue;
    }
    auto device_tensor = output_kernel_tensors_[i]->device_address();
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
    if (repeat_index.find(i) != repeat_index.end() && i > repeat_index[i] && outputs_[i] != nullptr) {
      tensor->set_device_address(outputs_[repeat_index[i]]->device_address());
      tensor->set_storage_info(outputs_[repeat_index[i]]->storage_info());
      continue;
    }

    auto tensor_device_address = tensor->device_address();
    MS_EXCEPTION_IF_NULL(tensor_device_address);
    // Update tensor device address by device tensor of output node.
    output_kernel_tensors_[i]->set_original_ref_count(SIZE_MAX);
    output_kernel_tensors_[i]->ResetRefCount();
    output_kernel_tensors_[i]->set_dynamic_ref_count(INT32_MAX);
    auto node_with_index = device_tensor->GetNodeIndex();
    tensor_device_address->SetNodeIndex(node_with_index.first, node_with_index.second);
    tensor_device_address->set_from_persistent_mem(device_tensor->from_persistent_mem());
    // The outputs may have the same output node, so need skip when the node has been done.
    if (tensor_device_address->GetPtr() != nullptr) {
      continue;
    }

    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

    // If the output node whose output address ptr can't be changed, then alloc the new device memory and copy the data:
    if (IsOutputAddressPersisted(output_kernel_tensors_[i], output_nodes_[i])) {
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, GetAID().Name(), memory::mem_pool::MemType::kOther,
                                                     tensor_device_address->GetSize(), tensor_device_address.get());
      if (!host_context->device_res_manager_->AllocateMemory(tensor_device_address.get(), kDefaultStreamIndex)) {
        MS_LOG_WITH_NODE(EXCEPTION, output_node)
          << "Device(id:" << device_id << ") memory isn't enough and alloc failed in output actor, kernel name: "
          << output_node->fullname_with_scope() << ", alloc size: " << tensor_device_address->GetSize() << "B.";
      }
      if (mindspore::runtime::IsDisableRuntimeConfig(mindspore::runtime::kRuntimeCopyAsync)) {
        MS_LOG(DEBUG) << "Sync device data from device tensor: " << device_tensor
                      << ", to device tensor: " << tensor_device_address << ", size: " << device_tensor->GetSize();
        if (!SyncCopy(tensor, output_kernel_tensors_[i].get(), kDefaultStreamIndex) ||
            !host_context->device_res_manager_->SyncAllStreams()) {
          MS_LOG_WITH_NODE(EXCEPTION, output_node)
            << "Sync device to device failed, device type: " << tensor_device_address->GetDeviceType()
            << ", output node: " << output_node->fullname_with_scope();
        }
      } else {
        MS_LOG(DEBUG) << "Async device data from device tensor: " << device_tensor
                      << ", to device tensor: " << tensor_device_address << ", size: " << device_tensor->GetSize();
        if (!AsyncCopy(tensor, output_kernel_tensors_[i].get(), kDefaultStreamIndex)) {
          MS_LOG_WITH_NODE(EXCEPTION, output_node)
            << "Async device to device failed, device type: " << tensor_device_address->GetDeviceType()
            << ", output node: " << output_node->fullname_with_scope();
        }
      }

    } else {
      MS_LOG(DEBUG) << "Swap ptr:" << device_tensor->GetPtr() << " from device tensor:" << device_tensor
                    << " device type:" << device_tensor->GetDeviceType() << " to :" << tensor_device_address
                    << " device type:" << tensor_device_address->GetDeviceType();
      // Move the device ptr from device_tensor to tensor_device_address.
      device_tensor->Swap(tensor_device_address.get());
      if (output_kernel_tensors_[i]->user_data()) {
        tensor->CloneUserData(*(output_kernel_tensors_[i]->user_data()));
      }
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
      MarkTensorAsOutput, GetAID().Name(), device::GetDeviceNameByType(device_tensor->GetDeviceType()),
      device_tensor->GetPtr(), output_kernel_tensors_[i]->dtype_id(), output_kernel_tensors_[i]->GetShapeVector(),
      output_kernel_tensors_[i]->tensor_storage_info());
  }

  // output types used for construct outputs.
  RecordOutputTypes(&output_types_, output_kernel_tensors_, output_nodes_.size());
  output_node_to_tensor_device_address_.clear();
  output_node_to_tensor_storage_info_.clear();
  output_nodes_.clear();
  output_nodes_.resize(outputs_num_);
  output_kernel_tensors_.clear();
  output_kernel_tensors_.resize(outputs_num_);
  if (!flatten_stub_nodes_.empty()) {
    flatten_stub_nodes_.clear();
  }
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
