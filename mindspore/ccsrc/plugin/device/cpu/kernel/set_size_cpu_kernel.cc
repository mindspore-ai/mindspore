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

#include "plugin/device/cpu/kernel/set_size_cpu_kernel.h"

#include <algorithm>
#include <unordered_set>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIndicesShapeSize = 2;
constexpr size_t kSetSizeInputsNum = 3;
constexpr size_t kSetSizeOutputsNum = 1;
}  // namespace

void SetSizeCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  validate_indices_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "validate_indices");
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  val_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  auto indices_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (indices_shape.size() != kIndicesShapeSize) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', it requires 'set_indices' should be a "
                             << kIndicesShapeSize << "-D Tensor, but got " << indices_shape.size() << "-D.";
  }
  auto values_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (values_shape.size() != 1 || values_shape[0] != indices_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', it requires 'set_values' should be a 1-D Tensor "
                                "and the first dimension length "
                                "should be equal to the first dimension length of "
                                "'set_indices', but got 'set_values' shape: "
                             << Vector2Str(values_shape) << " and 'set_indices' shape: " << Vector2Str(indices_shape)
                             << ".";
  }
  auto shape_index = 2;
  shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, shape_index);
  if (shape_.size() != 1 || shape_[0] != indices_shape[1]) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', it requires 'set_shape' should be a 1-D Tensor "
                                "and the first dimension length "
                                "should be equal to the second dimension length of "
                                "'set_indices', but got 'set_shape' shape: "
                             << Vector2Str(shape_) << " and 'set_indices' shape: " << Vector2Str(indices_shape) << ".";
  }
  values_size_ = SizeToLong(values_shape[0]);
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  dims_ = shape_[0];
}

bool SetSizeCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSetSizeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSetSizeOutputsNum, kernel_name_);
  if (outputs[0]->size == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', output memory size should be greater than 0, but got 0.";
    return true;
  }
  auto ret = memset_s(outputs[0]->addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output failed. Error no: " << ret << ".";
  }
  for (unsigned int i = 0; i < values_size_ && validate_indices_; ++i) {
    if (!IndicesValid(i, inputs)) {
      return false;
    }
  }
  switch (val_dtype_) {
    case kNumberTypeInt8:
      (void)SetSizeCompute<int8_t>(inputs, outputs);
      break;
    case kNumberTypeInt16:
      (void)SetSizeCompute<int16_t>(inputs, outputs);
      break;
    case kNumberTypeInt32:
      (void)SetSizeCompute<int32_t>(inputs, outputs);
      break;
    case kNumberTypeInt64:
      (void)SetSizeCompute<int64_t>(inputs, outputs);
      break;
    case kNumberTypeUInt8:
      (void)SetSizeCompute<uint8_t>(inputs, outputs);
      break;
    case kNumberTypeUInt16:
      (void)SetSizeCompute<uint16_t>(inputs, outputs);
      break;
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', set_values type error.";
      break;
  }
  return true;
}

bool SetSizeCpuKernelMod::IndicesValid(int64_t n, const std::vector<kernel::AddressPtr> &inputs) const {
  bool valid = true;
  bool different = false;
  bool increasing = true;
  const auto *indices_t = static_cast<int64_t *>(inputs[0]->addr);
  const auto *shape_t = static_cast<int64_t *>(inputs[2]->addr);
  for (int64_t di = 0; di < dims_; ++di) {
    if (indices_t[(n * dims_) + di] < 0 || indices_t[(n * dims_) + di] >= shape_t[di]) {
      valid = false;
      break;
    }
    if (n != 0) {
      int64_t diff = indices_t[(n * dims_) + di] - indices_t[((n - 1) * dims_) + di];
      if (diff > 0) {
        different = true;
      }
      if (!different && diff < 0) {
        increasing = false;
        break;
      }
    }
  }
  if (n == 0) {
    different = true;
  }
  if (!valid) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices is out of bounds, index=" << n << ".";
    return false;
  }
  if (!increasing) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices is out of order, index=" << n << ".";
    return false;
  }
  if (!different) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices is repeated, index=" << n << ".";
    return false;
  }
  return true;
}

template <typename T>
bool SetSizeCpuKernelMod::SetSizeCompute(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) const {
  auto output_t = static_cast<int32_t *>(outputs[0]->addr);
  auto indices_t = static_cast<int64_t *>(inputs[0]->addr);
  auto vals_t = static_cast<T *>(inputs[1]->addr);
  auto vals_num = values_size_;
  std::vector<int64_t> strides(dims_ - 1);
  auto shape_t = static_cast<int64_t *>(inputs[2]->addr);
  if (dims_ > 1) {
    int t = 2;
    strides[dims_ - t] = 1;
  }
  for (int32_t d = dims_ - 3; d >= 0; --d) {
    strides[d] = strides[d + 1] * shape_t[d + 1];
  }

  int32_t output_size = 1;
  for (int32_t d = 0; d < dims_ - 1; ++d) {
    output_size = output_size * shape_t[d];
  }
  std::vector<std::unordered_set<T>> all_values(output_size);
  for (unsigned int n = 0; n < vals_num; ++n) {
    int64_t ix = 0;
    for (int d = 0; d < dims_ - 1; ++d) {
      const int64_t ix_n_d = indices_t[n * dims_ + d];
      ix += strides[d] * ix_n_d;
    }
    all_values[ix].insert(*(vals_t + n));
  }
  for (int i = 0; i < output_size; ++i) {
    output_t[i] = SizeToLong(all_values[i].size());
  }
  return true;
}

std::vector<KernelAttr> SetSizeCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt32)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SetSize, SetSizeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
