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

#include "plugin/device/cpu/kernel/sparse_segment_mean_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseSegmentMeanGradInputsNum = 4;
constexpr size_t kSparseSegmentMeanGradOutputsNum = 1;

#define ADD_KERNEL(t1, t2, t3, t4, t5) \
  KernelAttr()                         \
    .AddInputAttr(kNumberType##t1)     \
    .AddInputAttr(kNumberType##t2)     \
    .AddInputAttr(kNumberType##t3)     \
    .AddInputAttr(kNumberType##t4)     \
    .AddOutputAttr(kNumberType##t5)
}  // namespace

bool SparseSegmentMeanGradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SparseSegmentMeanGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  x_shape_ = inputs.at(kIndex0)->GetDeviceShapeVector();
  segment_ids_shape_ = inputs.at(kIndex2)->GetDeviceShapeVector();
  y_shape_ = outputs.at(kIndex0)->GetDeviceShapeVector();
  return KRET_OK;
}

template <typename T>
bool SparseSegmentMeanGradCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                     const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseSegmentMeanGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseSegmentMeanGradOutputsNum, kernel_name_);
  constexpr size_t kMultiply = 1;
  size_t n = std::accumulate(x_shape_.begin(), x_shape_.end(), kMultiply, std::multiplies<int>()) / x_shape_[kIndex0];
  size_t m = std::accumulate(segment_ids_shape_.begin(), segment_ids_shape_.end(), kMultiply, std::multiplies<int>());
  size_t num_elements = std::accumulate(y_shape_.begin(), y_shape_.end(), kMultiply, std::multiplies<int>());
  int32_t k = *reinterpret_cast<int32_t *>(inputs[kIndex3]->device_ptr());
  auto x_addr = reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  auto indices_addr = reinterpret_cast<int32_t *>(inputs[kIndex1]->device_ptr());
  auto segment_ids_addr = reinterpret_cast<int32_t *>(inputs[kIndex2]->device_ptr());
  auto y_addr = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());

  for (size_t i = 0; i < num_elements; i++) {
    y_addr[i] = (T)0;
  }
  for (size_t i = 1; i < m; i++) {
    if (segment_ids_addr[i] < segment_ids_addr[i - 1]) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', segment_ids should be sorted.";
    }
  }
  for (size_t i = 0; i < m; i++) {
    if (indices_addr[i] >= k) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices is out of range of output_dim0.";
    }
    if (segment_ids_addr[i] >= x_shape_[0]) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', segment_ids is out of range of grad_dim0.";
    }
  }
  int beginindex = segment_ids_addr[0];
  size_t countnum = 1;
  for (size_t i = 1; i < m; i++) {
    if (segment_ids_addr[i] != beginindex) {
      for (size_t j = 1; j <= countnum; j++) {
        for (size_t l = 0; l < n; l++) {
          y_addr[IntToSize(indices_addr[i - j]) * n + l] += x_addr[IntToSize(beginindex) * n + l] / (T)(countnum);
        }
      }
      beginindex = segment_ids_addr[i];
      countnum = 1;
    } else {
      countnum++;
    }
  }

  int i = SizeToInt(m);
  for (size_t j = 1; j <= countnum; j++) {
    for (size_t l = 0; l < n; l++) {
      y_addr[IntToSize(indices_addr[IntToSize(i) - j]) * n + l] +=
        x_addr[IntToSize(beginindex) * n + l] / (T)(countnum);
    }
  }

  return true;
}

std::vector<std::pair<KernelAttr, SparseSegmentMeanGradCpuKernelMod::SparseSegmentMeanGradLaunchFunc>>
  SparseSegmentMeanGradCpuKernelMod::func_list_ = {
    {ADD_KERNEL(Float32, Int32, Int32, Int32, Float32), &SparseSegmentMeanGradCpuKernelMod::LaunchKernel<float>},
    {ADD_KERNEL(Float64, Int32, Int32, Int32, Float64), &SparseSegmentMeanGradCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> SparseSegmentMeanGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SparseSegmentMeanGradCpuKernelMod::SparseSegmentMeanGradLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSegmentMeanGrad, SparseSegmentMeanGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
