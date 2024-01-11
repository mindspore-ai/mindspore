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
#include "ms_kernels_internal/types.h"
internal::TensorFormat ToInternalFormat(Format format) { return internal::TensorFormat::TENSOR_FORMAT_ND; }

internal::TensorDType ToInternalDType(TypeId type) { return internal::TensorDType::TENSOR_DTYPE_FLOAT16; }

void ToInternalTensor(internal::Tensor *internal_tensor, const KernelTensor *kernel_tensor) {
  internal_tensor->desc.format = ToInternalFormat(kernel_tensor->format());
  internal_tensor->desc.dtype = ToInternalDType(kernel_tensor->dtype_id());
  internal_tensor->desc.dims = internal::VecToSVec<int64_t>(kernel_tensor->GetShapeVector());
  internal_tensor->data = kernel_tensor->device_ptr();
}

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
  info.input_num_ = inputsIdxMap.size();
  info.output_num_ = outputsIdxMap.size();
  info.input_dtype_ = ToInternalDType(inputs[0]->GetDtype());
  info.output_dtype_ = ToInternalDType(outputs[0]->GetDtype());
  info.input_format_ = ToInternalFormat(inputs[0]->GetFormat());
  info.output_format_ = ToInternalFormat(outputs[0]->GetFormat());
  impl_->Init(info);
  for (auto iter = inputsIdxMap.begin(); iter != inputsIdxMap.end(); iter++) {
    ToInternalTensor(inputs_[iter->second], inputs[iter->first]);
  }
  impl_->SetInputs(inputs_);
  return 0;
}
bool InternalKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  SetInOutIdx();
  inputs_.resize(inputsIdxMap_.size());
  std::fill(inputs_.begin(), inputs_.end(), new internal::Tensor());
  outputs_.resize(outputsIdxMap_.size());
  std::fill(outputs_.begin(), outputs_.end(), new internal::Tensor());
  Build(inputs, outputs);
  return true;
}
int InternalKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  Build(inputs, outputs);
  return 0;
}
bool InternalKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                               const DeviceRawBuf &tilingBuf, const std::vector<DeviceRawBuf> &workspace,
                               void *stream_ptr) {
  impl_->SetStream(stream_ptr);
  impl_->SetDeviceTilingBuf(tilingBuf);
  impl_->SetWorkSpace(workspace);
  for (auto iter = outputsIdxMap.begin(); iter != outputsIdxMap.end(); iter++) {
    ToInternalTensor(out[iter->second], [iter->first]);
  }
  impl_->SetOutputs(out);
  impl_->Launch();
  return true;
}
