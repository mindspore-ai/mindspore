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

#include "plugin/device/cpu/kernel/resize_nearest_neighbor_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/resize_nearest_neighbor.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kResizeNearestNeighborInputsNum = 1;
constexpr size_t kResizeNearestNeighborOutputNum = 1;
constexpr size_t kResizeNearestNeighborInputsShapeSize = 4;
constexpr size_t kResizeNearestNeighborAttrSize = 2;
}  // namespace

bool ResizeNearestNeighborCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeNearestNeighborInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeNearestNeighborOutputNum, kernel_name_);

  auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeNearestNeighbor>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  align_corners_ = kernel_ptr->get_align_corners();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int ResizeNearestNeighborCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  batch_size_ = input_shape[kIndex0];
  channel_ = input_shape[kIndex1];
  in_height_ = input_shape[kIndex2];
  in_width_ = input_shape[kIndex3];

  auto output_shape = LongVecToSizeVec(outputs[kIndex0]->GetShapeVector());
  out_height_ = output_shape[kIndex2];
  out_width_ = output_shape[kIndex3];

  height_scale_ = Scaling(in_height_, out_height_, align_corners_);
  width_scale_ = Scaling(in_width_, out_width_, align_corners_);
  output_size_ = batch_size_ * channel_ * out_height_ * out_width_;
  return KRET_OK;
}

template <typename T>
bool ResizeNearestNeighborCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &,
                                                     const std::vector<AddressPtr> &outputs) {
  auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  if (out_height_ == in_height_ && out_width_ == in_width_) {
    for (size_t i = 0; i < output_size_; ++i) {
      output_addr[i] = input_addr[i];
    }
  }

  for (size_t i = 0; i < output_size_; ++i) {
    size_t pos0 = i / (channel_ * out_height_ * out_width_) % batch_size_;
    size_t pos1 = i / (out_height_ * out_width_) % channel_;
    size_t pos2 = i / (out_width_) % out_height_;
    size_t pos3 = i % out_width_;
    const size_t in_y = std::min((align_corners_) ? static_cast<size_t>(roundf(pos2 * height_scale_))
                                                  : static_cast<size_t>(floorf(pos2 * height_scale_)),
                                 in_height_ - 1);
    const size_t in_x = std::min((align_corners_) ? static_cast<size_t>(roundf(pos3 * width_scale_))
                                                  : static_cast<size_t>(floorf(pos3 * width_scale_)),
                                 in_width_ - 1);
    size_t input_pos =
      pos0 * channel_ * in_height_ * in_width_ + pos1 * in_height_ * in_width_ + in_y * in_width_ + in_x;
    output_addr[i] = input_addr[input_pos];
  }
  return true;
}

const std::vector<std::pair<KernelAttr, ResizeNearestNeighborCpuKernelMod::KernelRunFunc>>
  &ResizeNearestNeighborCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ResizeNearestNeighborCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &ResizeNearestNeighborCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ResizeNearestNeighborCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &ResizeNearestNeighborCpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &ResizeNearestNeighborCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &ResizeNearestNeighborCpuKernelMod::LaunchKernel<int64_t>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeNearestNeighbor, ResizeNearestNeighborCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
