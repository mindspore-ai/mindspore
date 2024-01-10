/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/softmax_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <unordered_map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/softmax_fp32.h"

namespace mindspore {
namespace kernel {
namespace {
#define SOFTMAX_CPU_REG(M_T, T)                                                                       \
  KernelAttr().AddInputAttr(M_T).AddInputAttr(kObjectTypeTuple, kNumberTypeInt64).AddOutputAttr(M_T), \
    &SoftmaxCpuKernelMod::LaunchKernel<T>

constexpr size_t kSoftmaxAxisNum = 1;

template <typename T>
struct AccType {
  using type = T;
};

template <>
struct AccType<float16> {
  using type = float;
};

template <typename T, typename acc_T>
void SoftmaxFunc(const T *input_ptr, T *output_ptr, acc_T *sum_data, int start, int end, int dim_axis, int inner_size) {
  for (int i = start; i < end; i++) {
    int outter_offset = i * dim_axis * inner_size;
    int sum_outter_offset = i * inner_size;

    for (int k = 0; k < inner_size; k++) {
      int inner_offset = outter_offset + k;
      T max_data = input_ptr[inner_offset];
      sum_data[k + sum_outter_offset] = static_cast<acc_T>(0);
      for (int j = 0; j < dim_axis; j++) {
        int axis_offset = inner_offset + j * inner_size;
        max_data = max_data > input_ptr[axis_offset] ? max_data : input_ptr[axis_offset];
      }
      for (int j = 0; j < dim_axis; j++) {
        int axis_offset = inner_offset + j * inner_size;
        output_ptr[axis_offset] = exp(input_ptr[axis_offset] - max_data);
        sum_data[k + sum_outter_offset] += static_cast<acc_T>(output_ptr[axis_offset]);
      }
    }

    for (int j = 0; j < dim_axis; j++) {
      int axis_offset = outter_offset + j * inner_size;
      for (int k = 0; k < inner_size; k++) {
        int inner_offset = axis_offset + k;
        output_ptr[inner_offset] = output_ptr[inner_offset] / static_cast<T>(sum_data[k + sum_outter_offset]);
      }
    }
  }
}
}  // namespace

bool SoftmaxCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Softmax does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;

  auto dtype = inputs[0]->dtype_id();
  unit_size_ = abstract::TypeIdSize(dtype);
  // for fp16, use acc_T in workspace
  unit_size_ = dtype == kNumberTypeFloat16 ? unit_size_ * 2 : unit_size_;

  return true;
}

void SoftmaxCpuKernelMod::CheckAndRectifyAxis(KernelTensor *axis_kernel_tensor) noexcept {
  auto axis_list = axis_kernel_tensor->GetValueWithCheck<std::vector<int64_t>>();
  if (axis_list.size() != kSoftmaxAxisNum) {
    MS_LOG(EXCEPTION) << "For Softmax, the parameter 'axis' only support int type on CPU, but got tuple.";
  }
  axis_ = axis_list[0];
  if (axis_ < -input_dims_ || axis_ >= input_dims_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be in range [" << -input_dims_ << ", "
                      << input_dims_ << "), but got " << axis_;
  }
  if (axis_ < 0) {
    axis_ += input_dims_;
  }
  last_axis_ = axis_ == input_dims_ - 1;
}

int SoftmaxCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs[kIndex0]->GetShapeVector();
  input_dims_ = SizeToLong(input_shape_.size());
  if (input_dims_ == 0) {
    MS_LOG(ERROR) << "Input shape size is 0.";
    return KRET_RESIZE_FAILED;
  }

  input_elements_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());
  if (!input_elements_) {
    return KRET_OK;
  }

  CheckAndRectifyAxis(inputs[kIndex1]);
  dim_axis_ = input_shape_[axis_];
  output_elements_ = input_elements_ / static_cast<size_t>(dim_axis_);
  inner_size_ =
    std::accumulate(input_shape_.begin() + axis_ + 1, input_shape_.end(), size_t(1), std::multiplies<size_t>());

  // workspace
  workspace_size_list_ = {output_elements_ * unit_size_};

  return KRET_OK;
}

template <typename T>
bool SoftmaxCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs) noexcept {
  if (!input_elements_) {
    return true;
  }

  auto input_data = GetDeviceAddress<T>(inputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(input_data, false);
  auto output_data = GetDeviceAddress<T>(outputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(output_data, false);

  using acc_T = typename AccType<T>::type;
  auto sum_data = GetDeviceAddress<acc_T>(workspace, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(sum_data, false);

  if constexpr (std::is_same_v<T, float>) {
    if (last_axis_) {
      auto task = [this, input_data, output_data](size_t start, size_t end) {
        int batch = SizeToInt(end - start);
        size_t offset = start * IntToSize(dim_axis_);
        (void)SoftmaxLastAxis(input_data + offset, output_data + offset, batch, dim_axis_);
      };
      ParallelLaunchAutoSearch(task, output_elements_, this, &parallel_search_info_);
      return true;
    }
  }

  auto outter_size = output_elements_ / inner_size_;
  auto task = [this, input_data, output_data, sum_data](int start, int end) {
    SoftmaxFunc<T, acc_T>(input_data, output_data, sum_data, start, end, static_cast<int>(dim_axis_),
                          static_cast<int>(inner_size_));
  };
  ParallelLaunchAutoSearch(task, outter_size, this, &parallel_search_info_);

  return true;
}

std::vector<std::pair<KernelAttr, SoftmaxCpuKernelMod::LaunchFunc>> SoftmaxCpuKernelMod::func_list_ = {
  {SOFTMAX_CPU_REG(kNumberTypeFloat16, float16)},
  {SOFTMAX_CPU_REG(kNumberTypeFloat32, float)},
  {SOFTMAX_CPU_REG(kNumberTypeFloat64, double)}};

std::vector<KernelAttr> SoftmaxCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  if (support_list.empty()) {
    (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, SoftmaxCpuKernelMod::LaunchFunc> &pair) { return pair.first; });
  }
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Softmax, SoftmaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
