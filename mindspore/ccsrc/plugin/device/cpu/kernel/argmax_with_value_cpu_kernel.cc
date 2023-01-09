/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/argmax_with_value.h"
#include "plugin/device/cpu/kernel/argmax_with_value_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr char kKernelName[] = "ArgMaxWithValue";

size_t get_element_num(const std::vector<size_t> &shape) {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  return size;
}

template <typename T>
bool check_validation(const std::vector<size_t> &shape, const size_t num_before_axis, const size_t num_after_axis,
                      const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the number of inputs must be 1, but got " << inputs.size();
  }
  if (outputs.size() != 2) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the number of outputs must be 2, but got " << outputs.size();
  }
  size_t data_size = sizeof(T);
  size_t input_size = get_element_num(shape) * data_size;
  size_t output_num = num_before_axis * num_after_axis;
  size_t out0_size = output_num * sizeof(int);
  size_t out1_size = output_num * data_size;
  if (inputs[0]->size != input_size) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the memory size of 'x' must be " << input_size
                      << ", but got the memory size is " << inputs[0]->size;
  }
  if (outputs[0]->size != out0_size) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the memory size of the 1st output must be " << out0_size
                      << ", but got the memory size is " << outputs[0]->size;
  }
  if (outputs[1]->size != out1_size) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the memory size of the 2nd output must be " << out1_size
                      << ", but got the memory size is " << outputs[1]->size;
  }
  return true;
}
}  // namespace

template <typename T>
bool ArgMaxWithValueCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  (void)check_validation<T>(shape_, num_before_axis_, num_after_axis_, inputs, outputs);

  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output0 = reinterpret_cast<int32_t *>(outputs[0]->addr);
  auto output1 = reinterpret_cast<T *>(outputs[1]->addr);

  auto task = [&](size_t start, size_t end) {
    for (size_t pos = start; pos < end; pos++) {
      size_t i = pos / num_after_axis_;
      size_t j = pos % num_after_axis_;
      size_t src_index_j = i * dim_axis_ * num_after_axis_ + j;

      T max_value = input[src_index_j];
      int32_t max_index = 0;
      for (size_t k = 0; k < dim_axis_; k++) {
        size_t src_index_k = k * num_after_axis_ + src_index_j;
        if (input[src_index_k] > max_value) {
          max_value = input[src_index_k];
          max_index = SizeToInt(k);
        }
      }
      auto dst_index = i * num_after_axis_ + j;
      output0[dst_index] = max_index;
      output1[dst_index] = max_value;
    }
  };
  ParallelLaunchAutoSearch(task, num_before_axis_ * num_after_axis_, this, &parallel_search_info_);

  return true;
}

bool ArgMaxWithValueCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto input_type_id = inputs.at(kIndex0)->GetDtype();
  switch (input_type_id) {
    case kNumberTypeFloat64:
      kernel_func_ = &ArgMaxWithValueCpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &ArgMaxWithValueCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &ArgMaxWithValueCpuKernelMod::LaunchKernel<float16>;
      break;
    case kNumberTypeInt64:
      kernel_func_ = &ArgMaxWithValueCpuKernelMod::LaunchKernel<int64_t>;
      break;
    case kNumberTypeInt32:
      kernel_func_ = &ArgMaxWithValueCpuKernelMod::LaunchKernel<int32_t>;
      break;
    case kNumberTypeInt16:
      kernel_func_ = &ArgMaxWithValueCpuKernelMod::LaunchKernel<int16_t>;
      break;
    case kNumberTypeInt8:
      kernel_func_ = &ArgMaxWithValueCpuKernelMod::LaunchKernel<int8_t>;
      break;
    case kNumberTypeUInt64:
      kernel_func_ = &ArgMaxWithValueCpuKernelMod::LaunchKernel<uint64_t>;
      break;
    case kNumberTypeUInt32:
      kernel_func_ = &ArgMaxWithValueCpuKernelMod::LaunchKernel<uint32_t>;
      break;
    case kNumberTypeUInt16:
      kernel_func_ = &ArgMaxWithValueCpuKernelMod::LaunchKernel<uint16_t>;
      break;
    case kNumberTypeUInt8:
      kernel_func_ = &ArgMaxWithValueCpuKernelMod::LaunchKernel<uint8_t>;
      break;
    default:
      MS_LOG(EXCEPTION) << "Argmax kernel does not support " << TypeIdToString(input_type_id);
  }
  return true;
}

int ArgMaxWithValueCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  shape_ = Convert2SizeTClipNeg(inputs.at(kIndex0)->GetDeviceShapeAdaptively());
  size_t shape_len = shape_.size();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ArgMaxWithValue>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  int64_t axis = kernel_ptr->axis();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  if (CheckNullInput(input_shape)) {
    kernel_name_ = base_operator->name();
    MS_LOG(EXCEPTION) << kernel_name_ << " cannot deal with empty input. Please try other inputs.";
  }
  axis += static_cast<int64_t>(shape_len);
  if (shape_len == 0 && axis != -1 && axis != 0) {
    MS_LOG(EXCEPTION) << "For ArgMaxWithValue with 0d input tensor, axis must be one of 0 or -1, but got " << axis
                      << ".";
    axis = 0;
  }
  num_before_axis_ = 1;
  num_after_axis_ = 1;
  dim_axis_ = 0;
  if (shape_len > 0) {
    axis = axis % SizeToLong(shape_len);
    for (size_t i = 0; i < shape_len; i++) {
      if (SizeToLong(i) < axis) {
        num_before_axis_ *= shape_[i];
      } else if (SizeToLong(i) > axis) {
        num_after_axis_ *= shape_[i];
      }
    }
    dim_axis_ = shape_[LongToSize(axis)];
  }
  return KRET_OK;
}

std::vector<KernelAttr> ArgMaxWithValueCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
  };
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ArgMaxWithValue, ArgMaxWithValueCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
