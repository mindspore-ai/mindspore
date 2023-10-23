

/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/rl/tensor_array_stack_kernel.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::TensorArrayPtr;
TensorArrayStackKernelMod::TensorArrayStackKernelMod()
    : handle_(0), value_size_(0), ele_size_(0), stream_ptr_(nullptr), type_(nullptr), is_dynamic_(true) {
  ResetResource();
}

int TensorArrayStackKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  auto shape = GetValue<std::vector<int64_t>>(primitive_->GetAttr("element_shape"));
  auto max_element = GetValue<int64_t>(primitive_->GetAttr("max_element"));
  is_dynamic_ = GetValue<bool>(primitive_->GetAttr("is_dynamic_shape"));
  auto size = GetValue<int64_t>(primitive_->GetAttr("size"));
  for (auto i : shape) {
    shapes_.push_back(LongToSizeClipNeg(i));
  }
  type_ = GetValue<TypePtr>(primitive_->GetAttr("dtype"));
  ele_size_ = GetTypeByte(type_);
  for (auto i : shapes_) {
    ele_size_ *= i;
  }
  if (is_dynamic_) {
    value_size_ = ele_size_ * LongToSize(max_element);
    is_need_retrieve_output_shape_ = true;
  } else {
    if (size <= 0) {
      MS_LOG(EXCEPTION) << "Size should larger than 0 when is_dynamic_shape = false, but get " << size;
    }
    value_size_ = ele_size_ * LongToSize(size);
  }
  output_size_list_.clear();
  output_size_list_.push_back(value_size_);
  return KRET_OK;
}

bool TensorArrayStackKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  return true;
}

void TensorArrayStackKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &outputs) {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "TensorArrayStack cudaStreamSynchronized failed");
  TensorArrayPtr tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle_);
  MS_EXCEPTION_IF_NULL(tensors_);
  size_t tensor_size = tensors_->GetValidSize();
  auto shape = shapes_;
  shape.insert(shape.begin(), tensor_size);
  MS_LOG(DEBUG) << "After postexecute, the real shape of TensorArrayStack is " << shape;
  // common::AnfAlgo::SetOutputInferTypeAndShape({type_->type_id()}, {Convert2Long(shape)}, kernel_node_.lock().get());
  outputs_[0]->SetShapeVector(Convert2Long(shape));
  outputs_[0]->set_size(value_size_);
}

void TensorArrayStackKernelMod::ResetResource() noexcept {
  handle_ = 0;
  value_size_ = 0;
  ele_size_ = 0;
  stream_ptr_ = nullptr;
  shapes_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

bool TensorArrayStackKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  stream_ptr_ = stream_ptr;
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  auto out_value = GetDeviceAddress<unsigned char>(outputs, 0);
  MS_ERROR_IF_NULL(out_value);
  MS_ERROR_IF_NULL(handle_addr);

  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  MS_ERROR_IF_NULL(cuda_stream);

  // Set out_value to zeros when TensorArray in static size.
  if (!is_dynamic_) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(out_value, 0, outputs[0]->size(), cuda_stream),
                                       "For 'TensorArrayStack', Cudamemset output value failed");
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&handle_, handle_addr, sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream),
    "For 'TensorArrayStack', get handle to host failed");
  if (cudaStreamQuery(cuda_stream) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream), "cuda Stream Sync Failed");
  }
  TensorArrayPtr tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle_);
  MS_ERROR_IF_NULL(tensors_);
  if (tensors_->GetValidSize() > tensors_->GetRealSize()) {
    MS_LOG(EXCEPTION) << "Invalid TensorArray size, maybe should Clear() TensorArray before next usage.";
  }
  for (size_t i = 0; i < tensors_->GetValidSize(); i++) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(out_value + ele_size_ * i, tensors_->GetTensorAddr(i), ele_size_,
                                                       cudaMemcpyDeviceToDevice, cuda_stream),
                                       "For 'TensorArrayStack', stack value failed");
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
