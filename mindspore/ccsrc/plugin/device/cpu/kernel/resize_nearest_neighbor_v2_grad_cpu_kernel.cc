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

#include "plugin/device/cpu/kernel/resize_nearest_neighbor_v2_grad_cpu_kernel.h"
#include <string>
#include "kernel/ops_utils.h"
#include "mindspore/core/ops/grad/resize_nearest_neighbor_v2_grad.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kResizeNearestNeighborV2GradInputsNum = 2;
constexpr size_t kResizeNearestNeighborV2GradOutputNum = 1;
}  // namespace

bool ResizeNearestNeighborV2GradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                   const std::vector<KernelTensorPtr> &inputs,
                                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeNearestNeighborV2GradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeNearestNeighborV2GradOutputNum, kernel_name_);
  auto op_prim = std::dynamic_pointer_cast<ops::ResizeNearestNeighborV2Grad>(base_operator);
  MS_ERROR_IF_NULL(op_prim);
  align_corners_ = op_prim->get_align_corners();
  half_pixel_centers_ = op_prim->get_half_pixel_centers();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
  y_type_ = outputs[kIndex0]->GetDtype();
  return true;
}

int ResizeNearestNeighborV2GradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                    const std::vector<KernelTensorPtr> &inputs,
                                                    const std::vector<KernelTensorPtr> &outputs,
                                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeNearestNeighborV2GradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeNearestNeighborV2GradOutputNum, kernel_name_);
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  y_shape_ = outputs[kIndex0]->GetDeviceShapeAdaptively();
  grads_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  y_size_ = SizeOf(y_shape_);
  if (y_type_ == kNumberTypeFloat16) {
    workspace_size_list_.push_back(y_size_ * sizeof(float));
  }
  return KRET_OK;
}

template <typename T, typename S>
void ResizeNearestNeighborV2GradCpuKernelMod::RealCompute(T *const input, S *const output) {
  const int64_t batch_size = grads_shape_[kIndex0];
  const int64_t channels = grads_shape_[kIndex1];
  const int64_t in_height = grads_shape_[kIndex2];
  const int64_t in_width = grads_shape_[kIndex3];

  const int64_t out_height = y_shape_[kIndex2];
  const int64_t out_width = y_shape_[kIndex3];

  const float height_scale = Scaling(out_height, in_height, align_corners_);
  const float width_scale = Scaling(out_width, in_width, align_corners_);

  auto grads_4d = EigenTensor(grads_shape_, static_cast<void *>(input)).tensor<T, kDim4>();
  auto y_4d = EigenTensor(y_shape_, static_cast<void *>(output)).tensor<S, kDim4>();
  y_4d.setZero();

  for (int64_t y = 0; y < in_height; ++y) {
    int64_t out_y =
      std::min((align_corners_) ? static_cast<int64_t>(roundf(Scaler(y, height_scale, half_pixel_centers_)))
                                : static_cast<int64_t>(floorf(Scaler(y, height_scale, half_pixel_centers_))),
               out_height - 1);
    for (int64_t x = 0; x < in_width; ++x) {
      int64_t out_x =
        std::min((align_corners_) ? static_cast<int64_t>(roundf(Scaler(x, width_scale, half_pixel_centers_)))
                                  : static_cast<int64_t>(floorf(Scaler(x, width_scale, half_pixel_centers_))),
                 out_width - 1);
      for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
          y_4d(b, c, out_y, out_x) += static_cast<S>(grads_4d(b, c, y, x));
        }
      }
    }
  }
}

template <typename T>
bool ResizeNearestNeighborV2GradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                           const std::vector<AddressPtr> &workspace,
                                                           const std::vector<kernel::AddressPtr> &outputs) {
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  MS_EXCEPTION_IF_NULL(input);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(output);
  if (y_type_ == kNumberTypeFloat16) {
    auto work_fp32 = GetDeviceAddress<float>(workspace, kIndex0);
    MS_EXCEPTION_IF_NULL(work_fp32);
    RealCompute<T, float>(input, work_fp32);
    auto task = [work_fp32, output](const size_t start, const size_t end) {
      for (size_t i = start; i < end; i++) {
        output[i] = static_cast<T>(work_fp32[i]);
      }
    };
    ParallelLaunchAutoSearch(task, y_size_, this, &parallel_search_info_);
  } else {
    RealCompute<T, T>(input, output);
  }
  return true;
}

#define RESIZE_NEAREST_NEIGHBOR_V2_GRAD_CPU_REG(MS_T, MS_S, T)            \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_S).AddOutputAttr(MS_T), \
    &ResizeNearestNeighborV2GradCpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, ResizeNearestNeighborV2GradCpuKernelMod::ResizeNearestNeighborV2GradLaunchFunc>>
  ResizeNearestNeighborV2GradCpuKernelMod::func_list_ = {
    {RESIZE_NEAREST_NEIGHBOR_V2_GRAD_CPU_REG(kNumberTypeFloat16, kNumberTypeInt32, float16)},
    {RESIZE_NEAREST_NEIGHBOR_V2_GRAD_CPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float)},
    {RESIZE_NEAREST_NEIGHBOR_V2_GRAD_CPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double)},
    {RESIZE_NEAREST_NEIGHBOR_V2_GRAD_CPU_REG(kNumberTypeFloat16, kNumberTypeInt64, float16)},
    {RESIZE_NEAREST_NEIGHBOR_V2_GRAD_CPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float)},
    {RESIZE_NEAREST_NEIGHBOR_V2_GRAD_CPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double)}};

std::vector<KernelAttr> ResizeNearestNeighborV2GradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, ResizeNearestNeighborV2GradCpuKernelMod::ResizeNearestNeighborV2GradLaunchFunc>
         &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeNearestNeighborV2Grad, ResizeNearestNeighborV2GradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
