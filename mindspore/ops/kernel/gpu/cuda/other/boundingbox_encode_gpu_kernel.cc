/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/other/boundingbox_encode_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool BoundingBoxEncodeGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 2;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);

  const size_t coordinate_size = 4;
  auto means = primitive_->GetAttr("means");
  MS_EXCEPTION_IF_NULL(means);
  if (means->isa<ValueSequence>()) {
    means_ = GetValue<std::vector<float>>(means);
  } else if (means->isa<FloatImm>()) {
    float mean = GetValue<float>(means);
    for (size_t i = 0; i < coordinate_size; i++) {
      (void)means_.emplace_back(mean);
    }
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the input 'means' must be a tuple or a list, and dtype must be float, but got is not.";
  }

  auto stds = primitive_->GetAttr("stds");
  MS_EXCEPTION_IF_NULL(stds);
  if (stds->isa<ValueSequence>()) {
    stds_ = GetValue<std::vector<float>>(stds);
  } else if (stds->isa<FloatImm>()) {
    float std = GetValue<float>(stds);
    for (size_t i = 0; i < coordinate_size; i++) {
      (void)stds_.emplace_back(std);
    }
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the input 'stds' must be a tuple or a list, and dtype must be float, but got is not.";
  }

  if (means_.size() < coordinate_size || stds_.size() < coordinate_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the length of input 'means' and 'stds' must be at least 4, "
                         "but got the length of 'means': "
                      << means_.size() << ", and the length of 'stds': " << stds_.size();
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int BoundingBoxEncodeGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool BoundingBoxEncodeGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &workspace,
                                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *anchor_addr = GetDeviceAddress<T>(inputs, 0);
  T *groundtruth_addr = GetDeviceAddress<T>(inputs, 1);
  T *deltas_addr = GetDeviceAddress<T>(outputs, 0);

  if (inputs[0]->size() != inputs[1]->size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', anchor box size must equal with groundtruth box size: " << inputs[1]->size() << ", but got "
                  << inputs[0]->size();
    return false;
  }

  const size_t coordinate = 4;
  const size_t block_size = inputs[0]->size() / sizeof(T);
  if ((block_size % coordinate) != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << ", the size of the box should be a multiple of 4.";
    return false;
  }

  auto status = BoundingBoxEncode(block_size / coordinate, anchor_addr, groundtruth_addr, deltas_addr, means_[0],
                                  means_[1], means_[2], means_[3], stds_[0], stds_[1], stds_[2], stds_[3],
                                  reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, BoundingBoxEncodeGpuKernelMod::BoundingBoxEncodeLaunchFunc>>
  BoundingBoxEncodeGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &BoundingBoxEncodeGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &BoundingBoxEncodeGpuKernelMod::LaunchKernel<half>}};

std::vector<KernelAttr> BoundingBoxEncodeGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, BoundingBoxEncodeGpuKernelMod::BoundingBoxEncodeLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BoundingBoxEncode, BoundingBoxEncodeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
