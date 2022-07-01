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

#include "plugin/device/gpu/kernel/other/check_valid_gpu_kernel.h"
#include <functional>
#include <utility>
#include <algorithm>
#include <memory>
#include "mindspore/core/ops/check_valid.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/check_valid_impl.cuh"

namespace mindspore {
namespace kernel {
bool CheckValidGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (kernel_name_ != prim::kPrimCheckValid->name()) {
    MS_LOG(ERROR) << "For 'CheckValid', the kernel name must be 'CheckValid', but got " << kernel_name_;
    return false;
  }
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'CheckValid', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int CheckValidGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto anchor_boxes_shape = inputs.at(kIndex0)->GetShapeVector();
  auto img_metas_shape = inputs.at(kIndex1)->GetShapeVector();
  auto valid_shape = outputs.at(kIndex0)->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(anchor_boxes_shape, kernel_name_, "bboxes") ||
                   CHECK_SHAPE_NULL(img_metas_shape, kernel_name_, "img_metas") ||
                   CHECK_SHAPE_NULL(valid_shape, kernel_name_, "output");
  return KRET_OK;
}

template <typename T, typename S>
bool CheckValidGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &outputs) {
  T *anchor_boxes_addr = GetDeviceAddress<T>(inputs, 0);
  T *img_metas_addr = GetDeviceAddress<T>(inputs, 1);
  S *valid_addr = GetDeviceAddress<S>(outputs, 0);

  constexpr size_t coordinate = 4;
  const size_t block_size = inputs[0]->size / sizeof(T);
  if ((block_size % coordinate) != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << ", the size of the box should be a multiple of 4.";
    return false;
  }
  const size_t size = block_size / coordinate;
  CheckValid(size, anchor_boxes_addr, img_metas_addr, valid_addr, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, CheckValidGpuKernelMod::CheckValidFunc>> CheckValidGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
   &CheckValidGpuKernelMod::LaunchKernel<float, bool>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
   &CheckValidGpuKernelMod::LaunchKernel<half, bool>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
   &CheckValidGpuKernelMod::LaunchKernel<int16_t, bool>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
   &CheckValidGpuKernelMod::LaunchKernel<uchar, bool>},
};

std::vector<KernelAttr> CheckValidGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CheckValidFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CheckValid, CheckValidGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
