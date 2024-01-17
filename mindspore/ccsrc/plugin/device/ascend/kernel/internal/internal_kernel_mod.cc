/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"

#include "acl/acl_rt.h"

namespace mindspore {
namespace kernel {
InternalKernelMod::~InternalKernelMod() {
  for (auto t : inputs_) {
    delete t;
  }
  inputs_.clear();
  for (auto t : outputs_) {
    delete t;
  }
  outputs_.clear();
}

int InternalKernelMod::Build(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto param = CreateOpParam(inputs, outputs);
  impl_ = internal::CreateInternalKernelImpl(param);

  // abstract validation info from inputs
  internal::ValidateInfo info;
  info.input_num_ = inputsIdxMap_.size();
  info.output_num_ = outputsIdxMap_.size();
  for (auto iter = inputsIdxMap_.begin(); iter != inputsIdxMap_.end(); iter++) {
    info.input_dtype_.emplace_back(InternalKernelUtils::ToInternalDType(inputs[iter->first]->dtype_id()));
    info.input_format_.emplace_back(InternalKernelUtils::ToInternalFormat(inputs[iter->first]->GetFormat()));
  }
  for (auto iter = outputsIdxMap_.begin(); iter != outputsIdxMap_.end(); iter++) {
    info.output_dtype_.emplace_back(InternalKernelUtils::ToInternalDType(outputs[iter->first]->dtype_id()));
    info.output_format_.emplace_back(InternalKernelUtils::ToInternalFormat(outputs[iter->first]->GetFormat()));
  }
  impl_->Init(info);
  for (auto iter = inputsIdxMap_.begin(); iter != inputsIdxMap_.end(); iter++) {
    InternalKernelUtils::ToInternalTensor(inputs_[iter->second], inputs[iter->first]);
  }
  impl_->SetInputs(inputs_);

  return 0;
}

bool InternalKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  SetInOutIdx();
  inputs_.resize(inputsIdxMap_.size());
  std::generate(inputs_.begin(), inputs_.end(), []() { return new internal::Tensor(); });

  outputs_.resize(outputsIdxMap_.size());
  std::generate(outputs_.begin(), outputs_.end(), []() { return new internal::Tensor(); });

  device_tiling_buf_.size_ = 0;
  device_tiling_buf_.addr_ = nullptr;
  return true;
}

int InternalKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    MS_LOG(ERROR) << "op " << op_type_ << " invoke resize failed";
    return KRET_RESIZE_FAILED;
  }

  ret = Build(inputs, outputs);
  if (ret != 0) {
    MS_LOG(ERROR) << "op " << op_type_ << " build kernel failed";
    return KRET_RESIZE_FAILED;
  }

  // Invoke Tiling
  std::vector<internal::DIMS> input_shapes(inputs_.size()), output_shapes;
  for (size_t i = 0; i < inputs_.size(); ++i) {
    input_shapes[i] = inputs_[i]->desc.dims;
  }
  impl_->InferShape(input_shapes, output_shapes);

  auto tiling_size = impl_->GetTilingBufSize();
  internal::HostRawBuf host_tiling_buf;
  host_tiling_buf.addr_ = malloc(tiling_size);
  host_tiling_buf.size_ = tiling_size;

  ret = impl_->Tiling(host_tiling_buf);
  if (ret != 0) {
    MS_LOG(ERROR) << "op " << op_type_ << " tiling failed";
    return KRET_RESIZE_FAILED;
  }

  // allocate device tiling buf
  if (tiling_size != device_tiling_buf_.size_) {
    if (device_tiling_buf_.addr_ != nullptr) {
      ret = aclrtFree(device_tiling_buf_.addr_);
      if (ret != 0) {
        MS_LOG(ERROR) << "op " << op_type_ << " free old device tiling buf failed";
        return KRET_RESIZE_FAILED;
      }
      device_tiling_buf_.addr_ = nullptr;
    }
    device_tiling_buf_.size_ = tiling_size;
    ret = aclrtMalloc(&device_tiling_buf_.addr_, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0) {
      MS_LOG(ERROR) << "op " << op_type_ << " alloc device tiling buf failed";
      return KRET_RESIZE_FAILED;
    }
  }
  ret = aclrtMemcpy(device_tiling_buf_.addr_, device_tiling_buf_.size_, host_tiling_buf.addr_, host_tiling_buf.size_,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != 0) {
    MS_LOG(ERROR) << "op " << op_type_ << " copy tiling buf to device failed";
    return KRET_RESIZE_FAILED;
  }

  // update workspace_size list
  auto workspace_size_list = impl_->GetWorkSpaceSize();
  workspace_size_list_.resize(workspace_size_list.size());
  for (size_t i = 0; i < workspace_size_list.size(); ++i) {
    workspace_size_list_[i] = static_cast<size_t>(workspace_size_list[i]);
  }

  return 0;
}

bool InternalKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  for (auto iter = inputsIdxMap_.begin(); iter != inputsIdxMap_.end(); ++iter) {
    inputs_[iter->second]->data = inputs[iter->first]->device_ptr();
  }
  impl_->SetInputs(inputs_);
  impl_->SetStream(stream_ptr);
  impl_->SetDeviceTilingBuf(device_tiling_buf_);
  std::vector<internal::DeviceRawBuf> ws_raw_bufs(workspace.size());
  for (size_t i = 0; i < workspace.size(); ++i) {
    ws_raw_bufs[i] = InternalKernelUtils::ToDeviceRawBuf(workspace[i]);
  }
  impl_->SetWorkSpace(ws_raw_bufs);
  for (auto iter = outputsIdxMap_.begin(); iter != outputsIdxMap_.end(); ++iter) {
    InternalKernelUtils::ToInternalTensor(outputs_[iter->second], outputs[iter->first]);
  }
  impl_->SetOutputs(outputs_);
  impl_->Launch();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
