/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/argmax_cpu_kernel.h"
#include <string>
#include "mindspore/core/ops/arg_max.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kArgMaxInputsNum = 1;
constexpr size_t kArgMaxOutputsNum = 1;
constexpr char kKernelName[] = "Argmax";

int64_t get_element_num(const std::vector<int64_t> &shape) { return SizeToLong(SizeOf(shape)); }

template <typename T>
bool check_validation(const std::vector<int64_t> &shape, const int64_t num_before_axis, const int64_t num_after_axis,
                      const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kArgMaxInputsNum, kKernelName);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kArgMaxOutputsNum, kKernelName);
  auto data_size = sizeof(T);
  int64_t input_size = get_element_num(shape) * static_cast<int64_t>(data_size);
  int64_t output_num = num_before_axis * num_after_axis;
  int64_t output_size = output_num * static_cast<int64_t>(sizeof(int));
  if (static_cast<int64_t>(inputs[0]->size) != input_size) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the memory size of 'input_x' must be equal to " << input_size
                      << ", but got the memory size is " << inputs[0]->size;
  }
  if (static_cast<int64_t>(outputs[0]->size) != output_size) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the memory size of output must be equal to " << output_size
                      << ", but got the memory size is " << outputs[0]->size;
  }
  return true;
}
}  // namespace

template <typename T>
bool ArgmaxCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  if (!check_validation<T>(shape_, num_before_axis_, num_after_axis_, inputs, outputs)) {
    return false;
  }

  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<int32_t *>(outputs[0]->addr);

  std::vector<float> array_axis(dim_axis_);
  for (int64_t i = 0; i < num_before_axis_; i++) {
    int64_t src_index_i = i * dim_axis_ * num_after_axis_;
    for (int64_t j = 0; j < num_after_axis_; j++) {
      int64_t src_index_j = src_index_i + j;
      for (int64_t k = 0; k < dim_axis_; k++) {
        auto src_index_k = LongToSize(k * num_after_axis_ + src_index_j);
        array_axis[LongToSize(k)] = static_cast<float>(input[src_index_k]);
      }
      auto max_ops = std::max_element(array_axis.begin(), array_axis.end());
      auto max_index = static_cast<int32_t>(std::distance(array_axis.begin(), max_ops));
      auto dst_index = static_cast<size_t>(i * num_after_axis_ + j);
      output[dst_index] = max_index;
    }
  }
  return true;
}

bool ArgmaxCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Argmax>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast Argmax ops failed!";
    return false;
  }
  if (inputs.size() < 1) {
    MS_LOG(ERROR) << "Argmax input size can not less than 1!";
    return false;
  }

  kernel_name_ = kernel_ptr->name();
  axis_ = kernel_ptr->get_axis();
  auto input_type_id = inputs[0]->GetDtype();
  switch (input_type_id) {
    case kNumberTypeFloat32:
      kernel_func_ = &ArgmaxCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &ArgmaxCpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "Argmax kernel does not support " << TypeIdToString(input_type_id);
      return false;
  }
  return true;
}

int ArgmaxCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }
  shape_ = inputs[0]->GetShapeVector();
  size_t shape_len = shape_.size();
  if (shape_len == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', the dimension of 'input_x' must be at least 1, but got 0.";
    return KRET_RESIZE_FAILED;
  }
  axis_ += SizeToLong(shape_len);
  if (axis_ < 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', the 'axis' must be in range [-1, " << (shape_len - 1)
                    << "], but got " << axis_;
    return KRET_RESIZE_FAILED;
  }
  axis_ = axis_ % SizeToLong(shape_len);
  num_before_axis_ = 1;
  num_after_axis_ = 1;
  for (size_t i = 0; i < shape_len; i++) {
    if (SizeToLong(i) < axis_) {
      num_before_axis_ *= shape_[i];
    } else if (SizeToLong(i) > axis_) {
      num_after_axis_ *= shape_[i];
    }
  }
  dim_axis_ = shape_[LongToSize(axis_)];
  return 0;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Argmax, ArgmaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
