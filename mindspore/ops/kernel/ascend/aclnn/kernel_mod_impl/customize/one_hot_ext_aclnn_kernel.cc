/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/one_hot_ext_aclnn_kernel.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace one_hot_ext {
namespace {
constexpr int64_t kMinDepth = 1;
constexpr int64_t kAutoDepth = -1;
constexpr int64_t kDefaultAxis = -1;

int64_t GetValueFromTensorToInt64(KernelTensor *tensor) {
  auto data_type = tensor->dtype_id();
  switch (data_type) {
    case kNumberTypeInt32:
      return static_cast<int64_t>(tensor->GetValueWithCheck<int32_t>());
    case kNumberTypeInt64:
      return static_cast<int64_t>(tensor->GetValueWithCheck<int64_t>());
    default:
      MS_LOG(EXCEPTION) << "For 'OneHotExt', max result dtype must be int32 or int64, but got " << data_type;
  }
  return 0;
}

void SetKernelTensorShapeAndType(KernelTensor *tensor, const ShapeVector &shape_vector, const TypePtr &type_ptr) {
  tensor->SetType(type_ptr);
  tensor->SetShape(std::make_shared<abstract::TensorShape>(shape_vector));
}

void SetKernelTensorSize(KernelTensor *tensor) {
  size_t type_size = UnitSizeInBytes(tensor->dtype_id());
  size_t size = SizeOf(tensor->GetShapeVector()) * type_size;
  tensor->set_size(size);
}

std::shared_ptr<KernelTensor> CreateTempKernelTensor(const KernelTensor *base_tensor, const ShapeVector &shape_vector,
                                                     const TypePtr &type_ptr) {
  MS_EXCEPTION_IF_NULL(base_tensor);
  auto tensor = std::make_shared<KernelTensor>();
  tensor->SetType(type_ptr);
  tensor->SetShape(std::make_shared<abstract::TensorShape>(shape_vector));
  tensor->SetDeviceType(base_tensor->GetDeviceType());
  tensor->set_stream_id(base_tensor->stream_id());
  tensor->set_alloc_stream_id(base_tensor->alloc_stream_id());
  tensor->set_format(base_tensor->format());
  return tensor;
}

device::DeviceContext *GetAscendDeviceContext() {
  return device::DeviceContextManager::GetInstance().GetDeviceContext("Ascend").get();
}

// The temporary device memory used to hold the max value is released when the Launch scope exits, so the device
// pointer kept by the member kernel tensor must be cleared at the same time. Otherwise the member would keep a
// released device pointer, and any later reader would access released device memory.
class MaxOutputDevicePtrGuard {
 public:
  explicit MaxOutputDevicePtrGuard(KernelTensor *tensor) : tensor_(tensor) {}
  ~MaxOutputDevicePtrGuard() {
    if (tensor_ != nullptr) {
      tensor_->set_device_ptr(nullptr);
    }
  }
  MaxOutputDevicePtrGuard(const MaxOutputDevicePtrGuard &) = delete;
  MaxOutputDevicePtrGuard &operator=(const MaxOutputDevicePtrGuard &) = delete;

 private:
  KernelTensor *tensor_;
};

void EnsureOutputDeviceMemory(KernelTensor *tensor, device::DeviceContext *device_context, void *stream_ptr,
                              size_t previous_size) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_context);
  auto device_address = tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  auto current_size = device_address->GetSize();
  if (device_address->GetPtr() != nullptr && previous_size >= current_size) {
    return;
  }

  if (device_address->GetPtr() != nullptr) {
    // The output is allocated by the runtime with the placeholder depth, while the real depth is data dependent, so
    // this kernel mod has to reallocate the output in place. Freeing the device address is only safe when this kernel
    // mod owns it exclusively, otherwise it would release device memory that is still referenced by other tensors.
    auto device_pointer_holders = device_address->device_pointer().use_count();
    if (tensor->managed_by_somas() || tensor->is_ptr_persisted() || device_pointer_holders != 1) {
      MS_LOG(EXCEPTION) << "For 'OneHotExt', the output device address is shared and can not be reallocated in the "
                           "kernel mod, managed by somas: "
                        << tensor->managed_by_somas() << ", pointer persisted: " << tensor->is_ptr_persisted()
                        << ", device pointer holders: " << device_pointer_holders << ".";
    }
    device_context->device_res_manager_->FreeMemory(device_address.get());
  }
  auto stream_id = device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream_ptr);
  if (!device_context->device_res_manager_->AllocateMemory(device_address.get(), stream_id)) {
    MS_LOG(EXCEPTION) << "Allocate memory failed for OneHotExt output, size: " << current_size;
  }
}

void ResetMaxOutputTensor(KernelTensor *max_output_tensor, KernelTensor *input_tensor, KernelTensor *depth_tensor) {
  SetKernelTensorShapeAndType(max_output_tensor, depth_tensor->GetShapeVector(), input_tensor->GetType());
  SetKernelTensorSize(max_output_tensor);
}

ShapeVector BuildOutputShape(const ShapeVector &input_shape, int64_t depth, int64_t axis) {
  auto output_shape = input_shape;
  if (axis >= 0) {
    (void)output_shape.insert(output_shape.begin() + axis, depth);
    return output_shape;
  }
  output_shape.push_back(depth);
  return output_shape;
}
}  // namespace

int OneHotExtAscend::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  const auto depth = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  auto_depth_ = (depth == kAutoDepth);
  need_update_output_shape_ = auto_depth_;
  const auto output_depth = auto_depth_ ? kMinDepth : depth;
  const auto output_shape = BuildOutputShape(inputs[kIndex0]->GetShapeVector(), output_depth, kDefaultAxis);
  SetKernelTensorShapeAndType(outputs[kIndex0], output_shape, outputs[kIndex0]->GetType());
  SetKernelTensorSize(outputs[kIndex0]);
  return AclnnKernelMod::Resize(inputs, outputs);
}

void OneHotExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  auto_depth_ = false;
  need_update_output_shape_ = false;
  depth_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  axis_ = kDefaultAxis;
  origin_output_typeptr_ = outputs[kIndex0]->GetType();
  output_shape_[kIndex0] = inputs[kIndex0]->GetShapeVector();
  if (depth_ == kAutoDepth) {
    auto_depth_ = true;
    depth_ = kMinDepth;
  }
  need_update_output_shape_ = auto_depth_;
  output_shape_[kIndex0] = BuildOutputShape(output_shape_[kIndex0], depth_, axis_);
  SetKernelTensorShapeAndType(outputs[kIndex0], output_shape_[kIndex0], origin_output_typeptr_);
  SetKernelTensorSize(outputs[kIndex0]);
  GetWorkspaceForResizeDoOneHot(inputs[kIndex0], depth_, inputs[kIndex2], inputs[kIndex3], axis_, outputs[kIndex0]);
  if (auto_depth_) {
    max_output_tensor_ =
      CreateTempKernelTensor(outputs[kIndex0], inputs[kIndex1]->GetShapeVector(), inputs[kIndex0]->GetType());
    SetKernelTensorSize(max_output_tensor_.get());
    GetWorkspaceForResizeDoMax(inputs[kIndex0], max_output_tensor_.get());
  }
}

bool OneHotExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (!auto_depth_) {
    RunOpDoOneHot(stream_ptr, workspace, inputs[kIndex0], depth_, inputs[kIndex2], inputs[kIndex3], axis_,
                  outputs[kIndex0]);
    return true;
  }

  ResetMaxOutputTensor(max_output_tensor_.get(), inputs[kIndex0], inputs[kIndex1]);
  auto output_device_context = GetAscendDeviceContext();
  auto max_output_mem =
    std::make_shared<kernel::MemBlock>(output_device_context, max_output_tensor_->size(), stream_ptr);
  max_output_tensor_->set_device_ptr(max_output_mem->ptr_);
  MaxOutputDevicePtrGuard max_output_device_ptr_guard(max_output_tensor_.get());
  RunOpDoMax(stream_ptr, workspace, inputs[kIndex0], max_output_tensor_.get());
  const auto max_value = GetValueFromTensorToInt64(max_output_tensor_.get());
  depth_ = std::max(max_value + 1, kMinDepth);
  output_shape_[kIndex0] = BuildOutputShape(inputs[kIndex0]->GetShapeVector(), depth_, axis_);
  const auto previous_output_size = outputs[kIndex0]->size();
  SetKernelTensorShapeAndType(outputs[kIndex0], output_shape_[kIndex0], origin_output_typeptr_);
  SetKernelTensorSize(outputs[kIndex0]);
  outputs[kIndex0]->set_tensor_storage_info(nullptr);
  SetOutputSizeList({outputs[kIndex0]->size()});
  EnsureOutputDeviceMemory(outputs[kIndex0], output_device_context, stream_ptr, previous_output_size);
  hash_id_DoOneHot_ = 0;
  auto res = GEN_EXECUTOR_BOOST(op_type_DoOneHot_, hash_id_DoOneHot_, inputs[kIndex0], depth_, inputs[kIndex2],
                                inputs[kIndex3], axis_, outputs[kIndex0]);
  UpdateWorkspace(res);
  executor_DoOneHot_ = std::get<kIndex1>(res);
  auto release_func = std::get<kIndex2>(res);
  hash_id_DoOneHot_ = std::get<kIndex3>(res);
  auto ws_size = std::get<kIndex0>(res);
  const auto &iter = ops_workspace_size_map_.find("DoOneHot");
  if (iter == ops_workspace_size_map_.end() || ws_size == 0) {
    RUN_OP_API_ASYNC(op_type_DoOneHot_, nullptr, 0, executor_DoOneHot_, stream_ptr, release_func);
    return true;
  }

  auto workspace_size_idx = iter->second.first;
  auto workspace_size = iter->second.second;
  if (!workspace.empty() && workspace.size() > workspace_size_idx &&
      workspace[workspace_size_idx]->size() == workspace_size && workspace_size == ws_size) {
    RUN_OP_API_ASYNC(op_type_DoOneHot_, workspace[workspace_size_idx]->device_ptr(), workspace_size, executor_DoOneHot_,
                     stream_ptr, release_func);
    return true;
  }

  // The placeholder depth=1 used during Resize may not match the runtime depth=max(input)+1.
  // Rebuild the one-hot executor with the real depth and allocate a temporary workspace when needed.
  static device::DeviceContext *device_context =
    device::DeviceContextManager::GetInstance().GetDeviceContext("Ascend").get();
  auto ws_ptr = std::make_shared<kernel::MemBlock>(device_context, ws_size, stream_ptr);
  RUN_OP_API_ASYNC(op_type_DoOneHot_, ws_ptr->ptr_, ws_size, executor_DoOneHot_, stream_ptr, release_func);
  return true;
}

void OneHotExtAscend::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  if (!auto_depth_) {
    output_shape_[kIndex0] = BuildOutputShape(inputs[kIndex0]->GetShapeVector(), depth_, axis_);
  }
  outputs[kIndex0]->SetShapeVector(output_shape_[kIndex0]);
  SetKernelTensorSize(outputs[kIndex0]);
}

MS_ACLNN_KERNEL_FACTORY_REG(OneHotExt, OneHotExtAscend);
}  // namespace one_hot_ext
}  // namespace kernel
}  // namespace mindspore
