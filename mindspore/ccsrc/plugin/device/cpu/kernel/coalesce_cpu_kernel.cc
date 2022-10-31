/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/coalesce_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCoalesceInputsNum = 3;
constexpr size_t kCoalesceOutputsNum = 3;
constexpr char kKernelName[] = "Coalesce";
}  // namespace

bool CoalesceCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << " which is not supported.";
  }

  return true;
}

void CoalesceCpuKernelMod::SyncData() {
  ShapeVector dims;
  (void)dims.emplace_back(SizeToLong(shape_size_));
  (void)dims.emplace_back(SizeToLong(jump) + 1);
  ShapeVector dim;
  (void)dim.emplace_back(SizeToLong(jump) + 1);
  outputs_[kIndex0]->SetShapeVector(dims);
  outputs_[kIndex1]->SetShapeVector(dim);
  outputs_[kIndex2]->SetShapeVector(y_shape_shape_);
}

bool CoalesceCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCoalesceInputsNum, kKernelName);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCoalesceOutputsNum, kKernelName);
  dtype_ = inputs.at(kIndex1)->GetDtype();
  is_need_retrieve_output_shape_ = true;
  return true;
}

int CoalesceCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_UNKNOWN_OUT_SHAPE && ret != KRET_OK) {
    return ret;
  }

  outputs_ = outputs;

  auto indices_shape = inputs.at(kIndex0)->GetShapeVector();
  y_shape_shape_ = inputs.at(kIndex2)->GetShapeVector();

  values_size_ = IntToSize(indices_shape[1]);
  shape_size_ = IntToSize(indices_shape[0]);

  return KRET_OK;
}

void CoalesceCpuKernelMod::Check(const std::vector<kernel::AddressPtr> &inputs) const {
  auto x_indices_addr = reinterpret_cast<int64_t *>(inputs[0]->addr);
  auto x_shape_addr = reinterpret_cast<int64_t *>(inputs[2]->addr);
  for (size_t i = 0; i < values_size_; i++) {
    for (size_t j = 0; j < shape_size_; j++) {
      if (x_indices_addr[j * values_size_ + i] < 0) {
        MS_EXCEPTION(ValueError) << "For Coalesce, values of elements of x_indices must be non-negative"
                                 << ", but got x_indices[" << j << "][" << i
                                 << "] = " << x_indices_addr[j * values_size_ + i];
      }
      if (x_indices_addr[j * values_size_ + i] >= x_shape_addr[j]) {
        MS_EXCEPTION(ValueError)
          << "For Coalesce, values of elements of x_indices can not exceed the limit set by x_shape"
          << ", but got x_indices[" << j << "][" << i << "] = " << x_indices_addr[j * values_size_ + i]
          << ", got x_shape[" << j << "] = " << x_shape_addr[j];
      }
    }
  }
}

template <typename T>
void CoalesceCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  auto x_indices_addr = reinterpret_cast<int64_t *>(inputs[0]->addr);
  auto x_values_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto x_shape_addr = reinterpret_cast<int64_t *>(inputs[2]->addr);
  auto y_indices_addr = reinterpret_cast<int64_t *>(outputs[0]->addr);
  auto y_values_addr = reinterpret_cast<T *>(outputs[1]->addr);
  auto y_shape_addr = reinterpret_cast<int64_t *>(outputs[2]->addr);
  Check(inputs);

  std::vector<size_t> reorder(values_size_);
  std::iota(reorder.begin(), reorder.end(), 0);

  size_t shape_size = shape_size_;
  size_t values_size = values_size_;
  auto sorter = [x_indices_addr, shape_size, values_size](size_t i, size_t j) -> bool {
    for (size_t n = 0; n < shape_size; n++) {
      if (x_indices_addr[n * values_size + i] < x_indices_addr[n * values_size + j]) {
        return true;
      }
      if (x_indices_addr[n * values_size + i] > x_indices_addr[n * values_size + j]) {
        return false;
      }
    }
    return true;
  };
  std::sort(reorder.begin(), reorder.end(), sorter);

  std::vector<bool> del(values_size_);
  del[0] = false;
  y_values_addr[0] = x_values_addr[reorder[0]];
  for (size_t i = 1; i < values_size_; i++) {
    del[i] = true;
    for (size_t j = 0; j < shape_size_; j++) {
      if (x_indices_addr[j * values_size_ + reorder[i]] != x_indices_addr[j * values_size_ + reorder[i - 1]]) {
        del[i] = false;
        break;
      }
    }
    if (del[i]) {
      y_values_addr[jump] += x_values_addr[reorder[i]];
    } else {
      jump++;
      y_values_addr[jump] = x_values_addr[reorder[i]];
    }
  }

  size_t up = 0;
  for (size_t i = 0; i < values_size_; i++) {
    if (!del[i]) {
      for (size_t j = 0; j < shape_size_; j++) {
        y_indices_addr[j * (jump + 1) + up] = x_indices_addr[j * values_size_ + reorder[i]];
      }
      up++;
    }
  }

  for (size_t i = 0; i < shape_size_; i++) {
    y_shape_addr[i] = x_shape_addr[i];
  }
}

std::vector<KernelAttr> CoalesceCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeInt64)};

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Coalesce, CoalesceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
