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
  x_num_ = SizeOf(inputs[kIndex0]->GetShapeVector());
  indices_num_ = SizeOf(inputs[kIndex1]->GetShapeVector());
  segment_ids_num_ = SizeOf(inputs[kIndex2]->GetShapeVector());
  y_num_ = SizeOf(outputs[kIndex0]->GetShapeVector());
  x_shape_0_val_ = inputs[kIndex0]->GetShapeVector()[0];
  inner_size_ = x_num_ / x_shape_0_val_;
  return KRET_OK;
}

bool SparseSegmentSqrtNCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &workspace,
                                            const std::vector<kernel::AddressPtr> &outputs) {
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
  auto x_addr = static_cast<T1 *>(inputs[kIndex0]->addr);
  auto indices_addr = static_cast<T2 *>(inputs[kIndex1]->addr);
  auto segment_ids_addr = static_cast<T3 *>(inputs[kIndex2]->addr);
  auto y_addr = static_cast<T1 *>(outputs[kIndex0]->addr);

  if (memset_s(y_addr, y_num_ * sizeof(T1), 0, y_num_ * sizeof(T1)) != EOK) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', failed to memset_s y_addr.";
  }
  if (segment_ids_num_ > 0) {
    std::vector<int64_t> start_end_point(1, 0);
    if (segment_ids_addr[0] != 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices in 'segment_ids' should be start from 0.";
    }
    if (indices_addr[0] >= x_shape_0_val_ || indices_addr[0] < 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices is out of range of x's first dimension.";
    }
    for (size_t idx = 1; idx < segment_ids_num_; ++idx) {
      if (segment_ids_addr[idx] == segment_ids_addr[idx - 1] + 1) {
        start_end_point.emplace_back(idx);
      } else if (segment_ids_addr[idx] != segment_ids_addr[idx - 1]) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', segment_ids should be sorted and contiguous.";
      }
      if (indices_addr[idx] >= x_shape_0_val_ || indices_addr[idx] < 0) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices is out of range of x's first dimension.";
      }
    }
    start_end_point.emplace_back(segment_ids_num_);
    for (int64_t idx_inner = 0; idx_inner < inner_size_; ++idx_inner) {
      for (size_t idx_seg_node = 0; idx_seg_node < start_end_point.size() - 1; ++idx_seg_node) {
        int64_t start = start_end_point[idx_seg_node];
        int64_t end = start_end_point[idx_seg_node + 1];
        float sum_val = static_cast<float>(0);
        for (int64_t idx_indices = start; idx_indices < end; ++idx_indices) {
          sum_val +=
            static_cast<float>(x_addr[idx_inner + static_cast<int64_t>(indices_addr[idx_indices]) * inner_size_]);
        }
        y_addr[idx_inner + idx_seg_node * inner_size_] =
          static_cast<T1>(sum_val / static_cast<float>(sqrt(end - start)));
      }
    }
  }
}

std::vector<KernelAttr> SparseSegmentSqrtNCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Float16, Int32, Int32, Float16), ADD_KERNEL(Float16, Int32, Int64, Float16),
    ADD_KERNEL(Float16, Int64, Int32, Float16), ADD_KERNEL(Float16, Int64, Int64, Float16),
    ADD_KERNEL(Float32, Int32, Int32, Float32), ADD_KERNEL(Float32, Int32, Int64, Float32),
    ADD_KERNEL(Float32, Int64, Int32, Float32), ADD_KERNEL(Float32, Int64, Int64, Float32),
    ADD_KERNEL(Float64, Int32, Int32, Float64), ADD_KERNEL(Float64, Int32, Int64, Float64),
    ADD_KERNEL(Float64, Int64, Int32, Float64), ADD_KERNEL(Float64, Int64, Int64, Float64)};

  return kernel_attr_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSegmentSqrtN, SparseSegmentSqrtNCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
