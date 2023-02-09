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

#include "plugin/device/cpu/kernel/sparse_segment_sqrt_n_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseSegmentSqrtNInputsNum = 3;
constexpr size_t kSparseSegmentSqrtNOutputsNum = 1;

#define ADD_KERNEL(t1, t2, t3, t4) \
  KernelAttr()                     \
    .AddInputAttr(kNumberType##t1) \
    .AddInputAttr(kNumberType##t2) \
    .AddInputAttr(kNumberType##t3) \
    .AddOutputAttr(kNumberType##t4)
}  // namespace

bool SparseSegmentSqrtNCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseSegmentSqrtNInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseSegmentSqrtNOutputsNum, kernel_name_);
  dtype_ = inputs.at(kIndex0)->GetDtype();
  dtype1_ = inputs.at(kIndex1)->GetDtype();
  dtype2_ = inputs.at(kIndex2)->GetDtype();
  return true;
}

int SparseSegmentSqrtNCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  x_shape_ = inputs.at(kIndex0)->GetDeviceShapeAdaptively();
  indices_shape_ = inputs.at(kIndex1)->GetDeviceShapeAdaptively();
  segment_ids_shape_ = inputs.at(kIndex2)->GetDeviceShapeAdaptively();
  y_shape_ = outputs.at(kIndex0)->GetDeviceShapeAdaptively();

  is_null_input_ = CHECK_SHAPE_NULL(x_shape_, kernel_name_, "x_shape_") ||
                   CHECK_SHAPE_NULL(indices_shape_, kernel_name_, "indices_shape_") ||
                   CHECK_SHAPE_NULL(segment_ids_shape_, kernel_name_, "segment_ids_shape_");
  if (is_null_input_) {
    return KRET_OK;
  }
  return KRET_OK;
}

bool SparseSegmentSqrtNCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &workspace,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  if (dtype_ == kNumberTypeFloat16) {
    if (dtype1_ == kNumberTypeInt32) {
      if (dtype2_ == kNumberTypeInt32) {
        LaunchKernel<float16, int32_t, int32_t>(inputs, outputs);
      } else {
        LaunchKernel<float16, int32_t, int64_t>(inputs, outputs);
      }
    } else {
      if (dtype2_ == kNumberTypeInt32) {
        LaunchKernel<float16, int64_t, int32_t>(inputs, outputs);
      } else {
        LaunchKernel<float16, int64_t, int64_t>(inputs, outputs);
      }
    }
  } else if (dtype_ == kNumberTypeFloat32) {
    if (dtype1_ == kNumberTypeInt32) {
      if (dtype2_ == kNumberTypeInt32) {
        LaunchKernel<float, int32_t, int32_t>(inputs, outputs);
      } else {
        LaunchKernel<float, int32_t, int64_t>(inputs, outputs);
      }
    } else {
      if (dtype2_ == kNumberTypeInt32) {
        LaunchKernel<float, int64_t, int32_t>(inputs, outputs);
      } else {
        LaunchKernel<float, int64_t, int64_t>(inputs, outputs);
      }
    }
  } else if (dtype_ == kNumberTypeFloat64) {
    if (dtype1_ == kNumberTypeInt32) {
      if (dtype2_ == kNumberTypeInt32) {
        LaunchKernel<double, int32_t, int32_t>(inputs, outputs);
      } else {
        LaunchKernel<double, int32_t, int64_t>(inputs, outputs);
      }
    } else {
      if (dtype2_ == kNumberTypeInt32) {
        LaunchKernel<double, int64_t, int32_t>(inputs, outputs);
      } else {
        LaunchKernel<double, int64_t, int64_t>(inputs, outputs);
      }
    }
  } else {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', data type of x is " << TypeIdLabel(dtype_)
                            << " which is not supported.";
  }
  return true;
}

template <typename T1, typename T2, typename T3>
void SparseSegmentSqrtNCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                  const std::vector<kernel::AddressPtr> &outputs) {
  size_t n = static_cast<size_t>(
    std::accumulate(x_shape_.begin(), x_shape_.end(), kIndex1, std::multiplies<int64_t>()) / x_shape_[kIndex0]);
  size_t m = static_cast<size_t>(
    std::accumulate(segment_ids_shape_.begin(), segment_ids_shape_.end(), kIndex1, std::multiplies<int64_t>()));
  size_t k =
    static_cast<size_t>(std::accumulate(y_shape_.begin(), y_shape_.end(), kIndex1, std::multiplies<int64_t>()));
  auto x_shape_0 = static_cast<T2>(x_shape_[kIndex0]);
  auto x_addr = static_cast<T1 *>(inputs[kIndex0]->addr);
  auto indices_addr = static_cast<T2 *>(inputs[kIndex1]->addr);
  auto segment_ids_addr = static_cast<T3 *>(inputs[kIndex2]->addr);
  auto y_addr = static_cast<T1 *>(outputs[kIndex0]->addr);

  for (size_t i = 0; i < k; i++) {
    y_addr[i] = static_cast<T1>(0);
  }
  if (segment_ids_addr[0] != 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', indices in 'segment_ids' should be contiguous and start from 0.";
  }
  for (size_t i = 1; i < m; i++) {
    if (segment_ids_addr[i] < segment_ids_addr[i - 1]) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', segment_ids should be sorted.";
    }
    if (segment_ids_addr[i] - segment_ids_addr[i - 1] > 1) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', indices in 'segment_ids' should be contiguous and start from 0.";
    }
  }
  for (size_t i = 0; i < m; i++) {
    if (indices_addr[i] >= x_shape_0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices is out of range of x's first dimension.";
    }
  }

  int oldindex = -1;
  int countnum = 0;
  for (size_t i = 0; i < m; i++) {
    if (oldindex == static_cast<int>(segment_ids_addr[i])) {
      countnum++;
    } else {
      if (countnum != 0) {
        for (size_t j = 0; j < n; j++) {
          y_addr[j + static_cast<size_t>(oldindex) * n] /= static_cast<T1>(sqrt(countnum));
        }
      }
      countnum = 1;
      oldindex = static_cast<int>(segment_ids_addr[i]);
      for (size_t j = 0; j < n; j++) {
        y_addr[j + static_cast<size_t>(oldindex) * n] = static_cast<T1>(0);
      }
    }
    for (size_t j = 0; j < n; j++) {
      y_addr[j + static_cast<size_t>(oldindex) * n] += x_addr[j + static_cast<size_t>(indices_addr[i]) * n];
    }
  }
  if (countnum != 0) {
    for (size_t j = 0; j < n; j++) {
      y_addr[j + static_cast<size_t>(oldindex) * n] /= static_cast<T1>(sqrt(countnum));
    }
  }
}

std::vector<KernelAttr> SparseSegmentSqrtNCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Float16, Int32, Int32, Float16), ADD_KERNEL(Float16, Int32, Int64, Float16),
    ADD_KERNEL(Float16, Int64, Int32, Float16), ADD_KERNEL(Float16, Int64, Int64, Float16),
    ADD_KERNEL(Float32, Int32, Int32, Float32), ADD_KERNEL(Float32, Int32, Int64, Float32),
    ADD_KERNEL(Float32, Int64, Int32, Float32), ADD_KERNEL(Float32, Int64, Int64, Float16),
    ADD_KERNEL(Float64, Int32, Int32, Float64), ADD_KERNEL(Float64, Int32, Int64, Float64),
    ADD_KERNEL(Float64, Int64, Int32, Float64), ADD_KERNEL(Float64, Int64, Int64, Float64)};

  return kernel_attr_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSegmentSqrtN, SparseSegmentSqrtNCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
