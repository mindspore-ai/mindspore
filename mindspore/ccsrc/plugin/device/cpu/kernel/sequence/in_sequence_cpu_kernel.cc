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

#include "plugin/device/cpu/kernel/sequence/in_sequence_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 2;
constexpr int kOutputsNum = 1;
}  // namespace
bool InSequenceCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int InSequenceCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  ele_shape_ = inputs[0]->GetShapeVector();
  tuple_shape_ = inputs[1]->GetShapeVector();
  ele_type_ = inputs[0]->GetDtype();
  input_type_ = inputs[1]->GetDtype();
  if (tuple_shape_.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the input tuple size must greater 0";
  }
  return KRET_OK;
}

template <typename T, typename S>
bool InSequenceCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs) {
  const auto ele_addr = GetDeviceAddress<T>(inputs, 0);
  const auto input_addr = GetDeviceAddress<S>(inputs, 1);
  auto output_addr = GetDeviceAddress<bool>(outputs, 0);
  auto len_seq = tuple_shape_[0];

  if (len_seq == 0 || ele_type_ != input_type_) {
    *output_addr = false;
    return true;
  }

  size_t element_index_size = std::accumulate(ele_shape_.begin(), ele_shape_.end(), 1, std::multiplies<int64_t>());
  for (size_t i = 0; i < static_cast<size_t>(len_seq); ++i) {
    bool res = true;
    for (size_t j = 0; j < element_index_size; ++j) {
      auto ele_eq = ele_addr[j] == input_addr[i * element_index_size + j];
      if (!ele_eq) {
        res = false;
        break;
      }
    }
    if (res) {
      *output_addr = true;
      return true;
    }
  }
  *output_addr = false;
  return true;
}

#define ADD_TENSOR_KERNEL(x_dtype, y_dtype, x_type, y_type) \
  {                                                         \
    KernelAttr()                                            \
      .AddInputAttr(x_dtype)                                \
      .AddInputAttr(kObjectTypeTuple, y_dtype)              \
      .AddOutputAttr(kObjectTypeNumber, kNumberTypeBool),   \
      &InSequenceCpuKernelMod::LaunchKernel<x_type, y_type> \
  }

#define ADD_KERNEL(x_dtype, y_dtype, x_type, y_type)        \
  {                                                         \
    KernelAttr()                                            \
      .AddInputAttr(kObjectTypeNumber, x_dtype)             \
      .AddInputAttr(kObjectTypeTuple, y_dtype)              \
      .AddOutputAttr(kObjectTypeNumber, kNumberTypeBool),   \
      &InSequenceCpuKernelMod::LaunchKernel<x_type, y_type> \
  }

const std::vector<std::pair<KernelAttr, InSequenceCpuKernelMod::KernelRunFunc>> &InSequenceCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, InSequenceCpuKernelMod::KernelRunFunc>> func_list = {
    ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeFloat32, float, float),
    ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeFloat64, float, double),
    ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeInt32, float, int),
    ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t),
    ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeFloat32, double, float),
    ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeFloat64, double, double),
    ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeInt32, double, int),
    ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t),
    ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeFloat32, int, float),
    ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeFloat64, int, double),
    ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeInt32, int, int),
    ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeInt64, int, int64_t),
    ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeFloat32, int64_t, float),
    ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeFloat64, int64_t, double),
    ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeInt32, int64_t, int),
    ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t),
    ADD_KERNEL(kNumberTypeFloat32, kNumberTypeFloat32, float, float),
    ADD_KERNEL(kNumberTypeFloat32, kNumberTypeFloat64, float, double),
    ADD_KERNEL(kNumberTypeFloat32, kNumberTypeInt32, float, int),
    ADD_KERNEL(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t),
    ADD_KERNEL(kNumberTypeFloat64, kNumberTypeFloat32, double, float),
    ADD_KERNEL(kNumberTypeFloat64, kNumberTypeFloat64, double, double),
    ADD_KERNEL(kNumberTypeFloat64, kNumberTypeInt32, double, int),
    ADD_KERNEL(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t),
    ADD_KERNEL(kNumberTypeInt32, kNumberTypeFloat32, int, float),
    ADD_KERNEL(kNumberTypeInt32, kNumberTypeFloat64, int, double),
    ADD_KERNEL(kNumberTypeInt32, kNumberTypeInt32, int, int),
    ADD_KERNEL(kNumberTypeInt32, kNumberTypeInt64, int, int64_t),
    ADD_KERNEL(kNumberTypeInt64, kNumberTypeFloat32, int64_t, float),
    ADD_KERNEL(kNumberTypeInt64, kNumberTypeFloat64, int64_t, double),
    ADD_KERNEL(kNumberTypeInt64, kNumberTypeInt32, int64_t, int),
    ADD_KERNEL(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, InSequence, InSequenceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
