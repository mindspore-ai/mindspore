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

#include "plugin/device/cpu/kernel/upsample_nearest_3d_cpu_kernel.h"
#include <string>
#include <utility>
#include "kernel/kernel_get_value.h"
#include "kernel/ops_utils.h"
#include "ops/upsample_nearest_3d.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kUpsampleNearest3DInputsNum = 2;
constexpr auto kUpsampleNearest3DOutputNum = 1;
const double kValueZero = 0.;
}  // namespace

void UpsampleNearest3DCpuKernelMod::ComputeNearestIndex(int64_t *const indices, const int64_t stride,
                                                        const int64_t input_szie, const int64_t output_size,
                                                        const double scale) const {
  auto loop = [&](int64_t begin, int64_t end) {
    for (int64_t out_idx = begin; out_idx < end; ++out_idx) {
      size_t in_idx = NearestIndex(static_cast<size_t>(out_idx), static_cast<size_t>(input_szie),
                                   static_cast<size_t>(output_size), scale);
      indices[out_idx] = static_cast<int64_t>(in_idx) * stride;
    }
  };
  float block_size = 64.0;
  ParallelLaunch(loop, static_cast<size_t>(output_size), block_size);
}

bool UpsampleNearest3DCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = sizeof(int64_t);
  return true;
}

int UpsampleNearest3DCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  // shape
  x_shape_ = inputs.at(kIndex0)->GetShapeVector();
  y_shape_ = outputs.at(kIndex0)->GetShapeVector();
  // apply workspace
  workspace_size_list_.push_back(unit_size_ * LongToSize(y_shape_[kIndex2]));
  workspace_size_list_.push_back(unit_size_ * LongToSize(y_shape_[kIndex3]));
  workspace_size_list_.push_back(unit_size_ * LongToSize(y_shape_[kIndex4]));
  // none_list
  MS_EXCEPTION_IF_NULL(base_operator);
  none_list_ = GetValue<std::vector<int64_t>>(base_operator->GetAttr(kAttrNoneList));
  if (none_list_.size() != kIndex1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', only one of output_size or scales should be specified.";
  }
  if (none_list_[kIndex0] == static_cast<int64_t>(kIndex2)) {
    scales_ = std::vector<double>(kIndex3, kValueZero);
  } else {
    if (!TryGetFloatValue(inputs, kIndex1, kernel_name_, &scales_, false)) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << " can't get scales input! ";
    }
  }
  return KRET_OK;
}

template <typename T>
bool UpsampleNearest3DCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  auto x_ptr = GetDeviceAddress<T>(inputs, kIndex0);
  MS_EXCEPTION_IF_NULL(x_ptr);
  auto y_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(y_ptr);

  int64_t channels = x_shape_[kIndex0] * x_shape_[kIndex1];
  int64_t input_depth = x_shape_[kIndex2];
  int64_t input_height = x_shape_[kIndex3];
  int64_t input_width = x_shape_[kIndex4];
  int64_t input_slice_size = input_depth * input_height * input_width;

  int64_t output_depth = y_shape_[kIndex2];
  int64_t output_height = y_shape_[kIndex3];
  int64_t output_width = y_shape_[kIndex4];
  int64_t output_slice_size = output_depth * output_height * output_width;

  if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
    auto cpy_ret = memcpy_s(y_ptr, outputs[kIndex0]->size, x_ptr, outputs[kIndex0]->size);
    if (cpy_ret != EOK) {
      MS_EXCEPTION(MemoryError) << "For " << kernel_name_ << ", memcpy_s to output failed.";
    }
    return true;
  }

  int64_t *const d_helper = GetDeviceAddress<int64_t>(workspace, kIndex0);
  int64_t *const h_helper = GetDeviceAddress<int64_t>(workspace, kIndex1);
  int64_t *const w_helper = GetDeviceAddress<int64_t>(workspace, kIndex2);
  MS_EXCEPTION_IF_NULL(d_helper);
  MS_EXCEPTION_IF_NULL(d_helper);
  MS_EXCEPTION_IF_NULL(d_helper);
  ComputeNearestIndex(d_helper, input_height * input_width, input_depth, output_depth, scales_[kIndex0]);
  ComputeNearestIndex(h_helper, input_width, input_height, output_height, scales_[kIndex1]);
  ComputeNearestIndex(w_helper, 1, input_width, output_width, scales_[kIndex2]);

  auto loop3d = [&](int64_t begin, int64_t end) {
    int64_t n{0};
    int64_t od{0};
    int64_t oh{0};

    (void)DataIndexInit(&begin, &n, &channels, &od, &output_depth, &oh, &output_height);
    for (int64_t i = begin; i < end; ++i) {
      int64_t id = d_helper[od];
      int64_t ih = h_helper[oh];
      T *dst_ptr = y_ptr + n * output_slice_size + od * output_height * output_width + oh * output_width;
      T *src_ptr = x_ptr + n * input_slice_size + id + ih;
      for (int64_t ow = 0; ow < output_width; ++ow) {
        dst_ptr[ow] = src_ptr[w_helper[ow]];
      }

      (void)DataIndexStep(&n, &channels, &od, &output_depth, &oh, &output_height);
    }
  };
  float block_size = 1.0;
  ParallelLaunch(loop3d, static_cast<size_t>(channels * output_depth * output_height), block_size);

  return true;
}
#define UpsampleNearest3D_CPU_KERNEL_REG(M_S, M_T, T) \
  KernelAttr().AddInputAttr(M_S).AddInputAttr(M_T).AddOutputAttr(M_S), &UpsampleNearest3DCpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, UpsampleNearest3DCpuKernelMod::KernelRunFunc>>
  UpsampleNearest3DCpuKernelMod::func_list_ = {
    {UpsampleNearest3D_CPU_KERNEL_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t)},
    {UpsampleNearest3D_CPU_KERNEL_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t)},
    {UpsampleNearest3D_CPU_KERNEL_REG(kNumberTypeUInt8, kNumberTypeFloat32, uint8_t)},
    {UpsampleNearest3D_CPU_KERNEL_REG(kNumberTypeFloat16, kNumberTypeInt32, float16)},
    {UpsampleNearest3D_CPU_KERNEL_REG(kNumberTypeFloat16, kNumberTypeInt64, float16)},
    {UpsampleNearest3D_CPU_KERNEL_REG(kNumberTypeFloat16, kNumberTypeFloat32, float16)},
    {UpsampleNearest3D_CPU_KERNEL_REG(kNumberTypeFloat32, kNumberTypeInt32, float)},
    {UpsampleNearest3D_CPU_KERNEL_REG(kNumberTypeFloat32, kNumberTypeInt64, float)},
    {UpsampleNearest3D_CPU_KERNEL_REG(kNumberTypeFloat32, kNumberTypeFloat32, float)},
    {UpsampleNearest3D_CPU_KERNEL_REG(kNumberTypeFloat64, kNumberTypeInt32, double)},
    {UpsampleNearest3D_CPU_KERNEL_REG(kNumberTypeFloat64, kNumberTypeInt64, double)},
    {UpsampleNearest3D_CPU_KERNEL_REG(kNumberTypeFloat64, kNumberTypeFloat32, double)},
};

std::vector<KernelAttr> UpsampleNearest3DCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, UpsampleNearest3DCpuKernelMod::KernelRunFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UpsampleNearest3D, UpsampleNearest3DCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
