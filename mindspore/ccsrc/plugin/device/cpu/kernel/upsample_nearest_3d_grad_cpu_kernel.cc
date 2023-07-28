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

#include "plugin/device/cpu/kernel/upsample_nearest_3d_grad_cpu_kernel.h"
#include <string>
#include <utility>
#include "kernel/kernel_get_value.h"
#include "kernel/ops_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const double kValueZero = 0.;
constexpr auto kUpsampleNearest3DGradInputsNum = 3;
constexpr auto kUpsampleNearest3DGradOutputNum = 1;
// GRAIN_SIZE for Parallel
constexpr size_t kGrainSize = 32768;
}  // namespace
void UpsampleNearest3DGradCpuKernelMod::ComputeNearestIndex(int64_t *const indices, const int64_t stride,
                                                            const int64_t input_szie, const int64_t output_size,
                                                            const double scale) const {
  auto loop = [&](int64_t begin, int64_t end) {
    for (int64_t out_idx = begin; out_idx < end; ++out_idx) {
      auto in_idx = NearestIndex(static_cast<size_t>(out_idx), static_cast<size_t>(input_szie),
                                 static_cast<size_t>(output_size), scale);
      indices[out_idx] = static_cast<int64_t>(in_idx) * stride;
    }
  };
  float block_size = 64.0;
  ParallelLaunch(loop, static_cast<size_t>(output_size), block_size);
}

bool UpsampleNearest3DGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int UpsampleNearest3DGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  output_shape_ = inputs.at(kIndex0)->GetDeviceShapeAdaptively();
  input_shape_ = outputs.at(kIndex0)->GetDeviceShapeAdaptively();
  // workspace
  size_t unit_size = sizeof(int64_t);
  workspace_size_list_.push_back(unit_size * static_cast<size_t>(output_shape_[kIndex2]));
  workspace_size_list_.push_back(unit_size * static_cast<size_t>(output_shape_[kIndex3]));
  workspace_size_list_.push_back(unit_size * static_cast<size_t>(output_shape_[kIndex4]));
  // none_list
  MS_EXCEPTION_IF_NULL(base_operator);
  none_list_ = GetValue<std::vector<int64_t>>(base_operator->GetAttr(kAttrNoneList));
  if (none_list_.size() != kIndex1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', only one of output_size or scales should be specified.";
  }
  if (none_list_[kIndex0] == static_cast<int64_t>(kIndex3)) {
    scales_ = std::vector<double>(kIndex3, kValueZero);
  } else {
    if (!TryGetFloatValue(inputs, kIndex2, kernel_name_, &scales_, false)) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << " can't get scales input! ";
    }
  }
  return KRET_OK;
}

template <typename T, typename S>
bool UpsampleNearest3DGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &workspace,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  // the input grad of backward process is the output of forward process
  auto grad_output_ptr = GetDeviceAddress<T>(inputs, kIndex0);
  MS_EXCEPTION_IF_NULL(grad_output_ptr);
  const int64_t total = CPUKernelUtils::CalcElementNum(input_shape_);
  S *grad_input_ptr = nullptr;
  bool is_fp16 = std::is_same<T, float16>::value;
  // define for fp16
  std::vector<S> grad_input_copy(1, 0);
  if (is_fp16) {
    grad_input_copy.resize(total, 0);
    grad_input_ptr = grad_input_copy.data();
    MS_EXCEPTION_IF_NULL(grad_input_ptr);
  } else {
    grad_input_ptr = GetDeviceAddress<S>(outputs, kIndex0);
    MS_EXCEPTION_IF_NULL(grad_input_ptr);
    int ret = memset_s(outputs[kIndex0]->addr, outputs[kIndex0]->size, 0, outputs[kIndex0]->size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s error. Error no: " << ret;
    }
  }

  // treat nbatch and channels as one dimension
  int64_t channels = input_shape_[kIndex0] * input_shape_[kIndex1];
  int64_t input_depth = input_shape_[kIndex2];
  int64_t input_height = input_shape_[kIndex3];
  int64_t input_width = input_shape_[kIndex4];

  int64_t output_depth = output_shape_[kIndex2];
  int64_t output_height = output_shape_[kIndex3];
  int64_t output_width = output_shape_[kIndex4];

  int64_t output_slice_size = output_depth * output_height * output_width;
  int64_t input_slice_size = input_depth * input_height * input_width;

  int64_t *const d_helper = GetDeviceAddress<int64_t>(workspace, kIndex0);
  MS_EXCEPTION_IF_NULL(d_helper);
  int64_t *const h_helper = GetDeviceAddress<int64_t>(workspace, kIndex1);
  MS_EXCEPTION_IF_NULL(h_helper);
  int64_t *const w_helper = GetDeviceAddress<int64_t>(workspace, kIndex2);
  MS_EXCEPTION_IF_NULL(w_helper);
  (void)ComputeNearestIndex(d_helper, input_height * input_width, input_depth, output_depth,
                            static_cast<double>(scales_[kIndex0]));
  (void)ComputeNearestIndex(h_helper, input_width, input_height, output_height, static_cast<double>(scales_[kIndex1]));
  (void)ComputeNearestIndex(w_helper, 1, input_width, output_width, static_cast<double>(scales_[kIndex2]));

  auto loop3d = [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; ++c) {
      for (int64_t od = 0; od < output_depth; ++od) {
        int64_t id = d_helper[od];

        for (int64_t oh = 0; oh < output_height; ++oh) {
          int64_t ih = h_helper[oh];

          for (int64_t ow = 0; ow < output_width; ++ow) {
            int64_t iw = w_helper[ow];

            int64_t output_offset = c * output_slice_size + od * output_height * output_width + oh * output_width + ow;
            int64_t input_offset = c * input_slice_size + id + ih + iw;
            grad_input_ptr[input_offset] += static_cast<S>(grad_output_ptr[output_offset]);
          }
        }
      }
    }
  };
  float block_size = static_cast<float>(kGrainSize) / output_slice_size;
  ParallelLaunch(loop3d, static_cast<size_t>(channels), block_size);
  // memcopy and cast for fp16
  if (is_fp16) {
    T *real_input_ptr = GetDeviceAddress<T>(outputs, kIndex0);
    MS_EXCEPTION_IF_NULL(real_input_ptr);
    auto task_fp16 = [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        real_input_ptr[idx] = static_cast<T>(grad_input_ptr[idx]);
      }
    };
    ParallelLaunch(task_fp16, static_cast<size_t>(total), block_size);
  }
  return true;
}

#define UpsampleNearest3D_GRAD_CPU_KERNEL_INT32_REG(M_S, M_T, T, S)                                   \
  KernelAttr().AddInputAttr(M_S).AddInputAttr(kNumberTypeInt32).AddInputAttr(M_T).AddOutputAttr(M_S), \
    &UpsampleNearest3DGradCpuKernelMod::LaunchKernel<T, S>
#define UpsampleNearest3D_GRAD_CPU_KERNEL_INT64_REG(M_S, M_T, T, S)                                   \
  KernelAttr().AddInputAttr(M_S).AddInputAttr(kNumberTypeInt64).AddInputAttr(M_T).AddOutputAttr(M_S), \
    &UpsampleNearest3DGradCpuKernelMod::LaunchKernel<T, S>

std::vector<std::pair<KernelAttr, UpsampleNearest3DGradCpuKernelMod::KernelRunFunc>>
  UpsampleNearest3DGradCpuKernelMod::func_list_ = {
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat16, kNumberTypeInt32, float16, float)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat32, kNumberTypeInt32, float, float)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat64, kNumberTypeInt32, double, double)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat16, kNumberTypeInt64, float16, float)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat32, kNumberTypeInt64, float, float)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat64, kNumberTypeInt64, double, double)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat16, kNumberTypeFloat32, float16, float)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat32, kNumberTypeFloat32, float, float)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat64, kNumberTypeFloat32, double, double)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat16, kNumberTypeInt32, float16, float)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat32, kNumberTypeInt32, float, float)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat64, kNumberTypeInt32, double, double)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat16, kNumberTypeInt64, float16, float)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat32, kNumberTypeInt64, float, float)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat64, kNumberTypeInt64, double, double)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat16, kNumberTypeFloat32, float16, float)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat32, kNumberTypeFloat32, float, float)},
    {UpsampleNearest3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat64, kNumberTypeFloat32, double, double)}};

std::vector<KernelAttr> UpsampleNearest3DGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, UpsampleNearest3DGradCpuKernelMod::KernelRunFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UpsampleNearest3DGrad, UpsampleNearest3DGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
