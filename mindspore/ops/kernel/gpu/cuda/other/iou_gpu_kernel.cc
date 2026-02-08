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

#include <map>
#include <utility>
#include <algorithm>

#include "kernel/gpu/cuda/other/iou_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIOUInputsNum = 2;
constexpr size_t kIOUOutputsNum = 1;
constexpr size_t kBoxCoordinateLen = 4;
constexpr auto kIou = "iou";
constexpr auto kIof = "iof";
};  // namespace

bool IOUGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIOUInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIOUOutputsNum, kernel_name_);

  auto mode_value_ptr = primitive_->GetAttr(kAttrMode);
  MS_EXCEPTION_IF_NULL(mode_value_ptr);
  auto mode = GetValue<std::string>(mode_value_ptr);
  if (mode == kIou) {
    mode_ = IOU_MODE;
  } else if (mode == kIof) {
    mode_ = IOF_MODE;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', mode only support 'iou' or 'iof'.";
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int IOUGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIOUInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIOUOutputsNum, kernel_name_);
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  size_t type_size = GetTypeByte(TypeIdToType(inputs[ANCHOR_BOXES]->dtype_id()));
  const size_t anchor_boxes_size_ = inputs[ANCHOR_BOXES]->size() / type_size;
  const size_t gt_boxes_size_ = inputs[GT_BOXES]->size() / type_size;
  if ((anchor_boxes_size_ % kBoxCoordinateLen) != 0 || (gt_boxes_size_ % kBoxCoordinateLen) != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << ", the size of the box should be a multiple of 4.";
    return false;
  }
  anchor_boxes_len_ = anchor_boxes_size_ / kBoxCoordinateLen;
  gt_boxes_len_ = gt_boxes_size_ / kBoxCoordinateLen;
  return KRET_OK;
}

template <typename T>
bool IOUGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  auto *anchor_boxes_addr = GetDeviceAddress<T>(inputs, ANCHOR_BOXES);
  auto *gt_boxes_addr = GetDeviceAddress<T>(inputs, GT_BOXES);
  auto *iou_addr = GetDeviceAddress<T>(outputs, IOU_VALUE);

  auto status = IOU(anchor_boxes_len_ * gt_boxes_len_, anchor_boxes_addr, gt_boxes_addr, iou_addr, mode_,
                    anchor_boxes_len_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, IOUGpuKernelMod::IOULaunchFunc>> IOUGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &IOUGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &IOUGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &IOUGpuKernelMod::LaunchKernel<double>},
};

std::vector<KernelAttr> IOUGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, IOUGpuKernelMod::IOULaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, IOU, IOUGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
