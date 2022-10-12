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

#include "plugin/device/cpu/kernel/sparse_segment_sqrt_n_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseSegmentSqrtNGradInputsNum = 4;
constexpr size_t kSparseSegmentSqrtNGradOutputsNum = 1;

#define ADD_KERNEL(t1, t2, t3, t4, t5) \
  KernelAttr()                         \
    .AddInputAttr(kNumberType##t1)     \
    .AddInputAttr(kNumberType##t2)     \
    .AddInputAttr(kNumberType##t3)     \
    .AddInputAttr(kNumberType##t4)     \
    .AddOutputAttr(kNumberType##t5)
}  // namespace

void SparseSegmentSqrtNGradCpuKernelMod::CheckParam(const CNodePtr &kernel_node) const {
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kSparseSegmentSqrtNGradInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kSparseSegmentSqrtNGradOutputsNum, kernel_name_);
}

void SparseSegmentSqrtNGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  x_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex0);
  x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex0);
  indices_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex1);
  segment_ids_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex2);
  output_dim0_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex3);
  y_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, kIndex0);
}

bool SparseSegmentSqrtNGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &workspace,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  if (x_dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (x_dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (x_dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', data type of x is " << TypeIdLabel(x_dtype_)
                            << " which is not supported.";
  }
  return true;
}

template <typename T>
void SparseSegmentSqrtNGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                      const std::vector<kernel::AddressPtr> &outputs) {
  size_t n = static_cast<size_t>(
    std::accumulate(x_shape_.begin(), x_shape_.end(), kIndex1, std::multiplies<int64_t>()) / x_shape_[kIndex0]);
  size_t m = static_cast<size_t>(
    std::accumulate(segment_ids_shape_.begin(), segment_ids_shape_.end(), kIndex1, std::multiplies<int64_t>()));
  size_t num_elements =
    static_cast<size_t>(std::accumulate(y_shape_.begin(), y_shape_.end(), kIndex1, std::multiplies<int64_t>()));
  int32_t k = *static_cast<int32_t *>(inputs[kIndex3]->addr);
  auto x_addr = static_cast<T *>(inputs[kIndex0]->addr);
  auto indices_addr = static_cast<int32_t *>(inputs[kIndex1]->addr);
  auto segment_ids_addr = static_cast<int32_t *>(inputs[kIndex2]->addr);
  auto y_addr = static_cast<T *>(outputs[kIndex0]->addr);

  for (size_t i = 0; i < num_elements; i++) {
    y_addr[i] = static_cast<T>(0);
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
    if (segment_ids_addr[i] >= k) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', segment_ids is out of range of output_dim0.";
    }
  }
  int beginindex = segment_ids_addr[0];
  size_t countnum = 1;
  for (size_t i = 1; i < m; i++) {
    if (segment_ids_addr[i] != beginindex) {
      for (size_t j = 1; j <= countnum; j++) {
        for (size_t l = 0; l < n; l++) {
          y_addr[IntToSize(indices_addr[i - j]) * n + l] +=
            x_addr[static_cast<size_t>(beginindex) * n + l] / static_cast<T>(sqrt(countnum));
        }
      }
      beginindex = segment_ids_addr[i];
      countnum = 1;
    } else {
      countnum++;
    }
  }

  size_t i = m;
  for (size_t j = 1; j <= countnum; j++) {
    for (size_t l = 0; l < n; l++) {
      y_addr[IntToSize(indices_addr[i - j]) * n + l] +=
        x_addr[static_cast<size_t>(beginindex) * n + l] / static_cast<T>(sqrt(countnum));
    }
  }
}

std::vector<KernelAttr> SparseSegmentSqrtNGradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {ADD_KERNEL(Float16, Int32, Int32, Int32, Float16),
                                                     ADD_KERNEL(Float32, Int32, Int32, Int32, Float32),
                                                     ADD_KERNEL(Float64, Int32, Int32, Int32, Float64)};

  return kernel_attr_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSegmentSqrtNGrad, SparseSegmentSqrtNGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
