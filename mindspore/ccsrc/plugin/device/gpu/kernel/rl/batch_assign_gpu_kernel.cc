/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <memory>
#include <shared_mutex>
#include "plugin/device/gpu/kernel/rl/batch_assign_gpu_kernel.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t kHalf = 2;
// Init shared_mutex in base.
std::shared_mutex BatchAssignBaseMod::rw_mutex_;

BatchAssignKernelMod::BatchAssignKernelMod() : elements_num_(0), lock_(false) {}

bool BatchAssignKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  size_t input_num = inputs.size();
  elements_num_ = (input_num - 1) / kHalf;
  auto lock_value = primitive_->GetAttr("lock");
  MS_EXCEPTION_IF_NULL(lock_value);
  lock_ = GetValue<bool>(lock_value);
  return true;
}

int BatchAssignKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just
    // return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  output_size_list_.clear();
  // Set an output for placeholder.
  output_size_list_.push_back(sizeof(float));
  return KRET_OK;
}

bool BatchAssignKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                  const std::vector<KernelTensor *> &, void *stream) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  // Using shared lock for reader so there can be more than one readers in the same time.
  // Using lock for writer to ensure there's only one writer at a time.
  if (lock_) {
    // Execute rw_mutex_.unlock() in lock's deconstruct.
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
  } else {
    // Execute rw_mutex_.unlock_shared() in lock's deconstruct.
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  }
  // Usually, we will get two inputs list, the first half are the weights to be updated, and the last half
  // are the sources. So we just copy the source to overwrite the dst.
  for (size_t i = 0; i < elements_num_; i++) {
    auto local_addr = GetDeviceAddress<int64_t>(inputs, i);
    auto source_addr = GetDeviceAddress<int64_t>(inputs, i + elements_num_);
    MS_ERROR_IF_NULL(local_addr);
    MS_ERROR_IF_NULL(source_addr);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(local_addr, source_addr, inputs[i]->size(), cudaMemcpyDeviceToDevice, cuda_stream),
      "Overwrite failed");
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream),
                                     "BatchAssignKernel cudaStreamSynchronized failed");
  return true;
}

std::vector<KernelAttr> BatchAssignKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BatchAssign, BatchAssignKernelMod);
}  // namespace kernel
}  // namespace mindspore
