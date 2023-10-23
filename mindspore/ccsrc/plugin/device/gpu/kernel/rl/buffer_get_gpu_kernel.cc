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
#include "plugin/device/gpu/kernel/rl/buffer_get_gpu_kernel.h"

#include <memory>
#include <string>

#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/rl/rl_buffer_impl.cuh"
#include "plugin/device/gpu/hal/device/gpu_common.h"

namespace mindspore {
namespace kernel {
constexpr size_t kSecondInputIndex = 2;
BufferGetKernelMod::BufferGetKernelMod() : element_nums_(0), capacity_(0) {}

BufferGetKernelMod::~BufferGetKernelMod() {}

int BufferGetKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  workspace_size_list_.clear();
  output_size_list_.clear();
  auto shapes = GetValue<std::vector<int64_t>>(primitive_->GetAttr("buffer_elements"));
  auto types = GetValue<std::vector<TypePtr>>(primitive_->GetAttr("buffer_dtype"));
  capacity_ = GetValue<int64_t>(primitive_->GetAttr("capacity"));
  element_nums_ = shapes.size();
  for (size_t i = 0; i < element_nums_; i++) {
    exp_element_list.push_back(shapes[i] * UnitSizeInBytes(types[i]->type_id()));
  }
  // buffer size
  for (auto i : exp_element_list) {
    output_size_list_.push_back(i);
  }
  workspace_size_list_.push_back(sizeof(int));
  return KRET_OK;
}

bool BufferGetKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                                const std::vector<KernelTensor *> &outputs, void *stream) {
  int *count_addr = GetDeviceAddress<int>(inputs, element_nums_);
  int *head_addr = GetDeviceAddress<int>(inputs, element_nums_ + 1);
  int *origin_index_addr = GetDeviceAddress<int>(inputs, element_nums_ + kSecondInputIndex);
  int *index_addr = GetDeviceAddress<int>(workspace, 0);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  auto status = ReMappingIndex(count_addr, head_addr, origin_index_addr, index_addr, cuda_stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  for (size_t i = 0; i < element_nums_; i++) {
    auto buffer_addr = GetDeviceAddress<unsigned char>(inputs, i);
    auto item_addr = GetDeviceAddress<unsigned char>(outputs, i);
    size_t one_exp_len = output_size_list_[i];
    status = BufferGetItem(one_exp_len, index_addr, one_exp_len, buffer_addr, item_addr, cuda_stream);
    CHECK_CUDA_STATUS(status, kernel_name_);
  }
  return true;
}
std::vector<KernelAttr> BufferGetKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BufferGetItem, BufferGetKernelMod);

}  // namespace kernel
}  // namespace mindspore
