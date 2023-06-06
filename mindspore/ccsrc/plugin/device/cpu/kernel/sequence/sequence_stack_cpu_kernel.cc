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

#include "plugin/device/cpu/kernel/sequence/sequence_stack_cpu_kernel.h"
#include <thread>
#include <algorithm>
#include <string>
#include <map>
#include <complex>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/add_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"
#include "mindspore/core/ops/sequence_stack.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSequenceStackOutputsNum = 1;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool SequenceStackFwdCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSequenceStackOutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int SequenceStackFwdCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  tuple_shape_ = inputs[0]->GetShapeVector();
  if (tuple_shape_.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the input tuple size must greater 0";
  }
  std::vector<size_t> shape_vec_item;
  std::copy(tuple_shape_.begin() + 1, tuple_shape_.end(), std::back_inserter(shape_vec_item));

  input_num_ = tuple_shape_[0];
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SequenceStack>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  axis_ = kernel_ptr->get_axis();
  if (axis_ < 0) {
    axis_ += (SizeToInt(shape_vec_item.size()) + 1);
  }
  dims_behind_axis_ = 1;
  // calculate elements while dim >= axis
  for (size_t i = IntToSize(axis_); i < shape_vec_item.size(); i++) {
    dims_behind_axis_ *= static_cast<size_t>(shape_vec_item[i]);
  }
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  output_size_ = 1;
  for (size_t i = 0; i < output_shape.size(); i++) {
    output_size_ *= static_cast<size_t>(output_shape[i]);
  }
  return KRET_OK;
}

template <typename T>
bool SequenceStackFwdCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs) {
  const auto input_addr = GetDeviceAddress<T>(inputs, 0);
  auto output_addr = GetDeviceAddress<T>(outputs, 0);

  size_t element_index_size =
    static_cast<size_t>(std::accumulate(tuple_shape_.begin() + 1, tuple_shape_.end(), 1, std::multiplies<int64_t>()));

  // multi-threading
  size_t input_size = output_size_;
  size_t dims_behind_axis = dims_behind_axis_;
  size_t copy_time = input_size / dims_behind_axis;
  size_t single_copy_size = dims_behind_axis * sizeof(T);
  auto task = [&](size_t start, size_t end) {
    for (size_t pos = start; pos < end; ++pos) {
      size_t cur_input_index = pos % this->input_num_;
      size_t local_idx = pos / this->input_num_;
      (void)memcpy_s(output_addr + dims_behind_axis * pos, single_copy_size,
                     input_addr + cur_input_index * element_index_size + dims_behind_axis * local_idx,
                     single_copy_size);
    }
  };
  ParallelLaunchAutoSearch(task, copy_time, this, &parallel_search_info_);
  return true;
}

#define SEQUENCE_STACK_REG(ms_type, builtin_type)                                \
  {                                                                              \
    KernelAttr().AddInputAttr(kObjectTypeTuple, ms_type).AddOutputAttr(ms_type), \
      &SequenceStackFwdCpuKernelMod::LaunchKernel<builtin_type>                  \
  }

const SequenceStackFwdCpuKernelMod::FuncList &SequenceStackFwdCpuKernelMod::GetFuncList() const {
  static const FuncList func_list = {
    SEQUENCE_STACK_REG(kNumberTypeInt8, int8_t),           SEQUENCE_STACK_REG(kNumberTypeInt16, int16_t),
    SEQUENCE_STACK_REG(kNumberTypeInt32, int32_t),         SEQUENCE_STACK_REG(kNumberTypeInt64, int64_t),
    SEQUENCE_STACK_REG(kNumberTypeUInt8, uint8_t),         SEQUENCE_STACK_REG(kNumberTypeUInt16, uint16_t),
    SEQUENCE_STACK_REG(kNumberTypeUInt32, uint32_t),       SEQUENCE_STACK_REG(kNumberTypeUInt64, uint64_t),
    SEQUENCE_STACK_REG(kNumberTypeFloat16, float16),       SEQUENCE_STACK_REG(kNumberTypeFloat32, float),
    SEQUENCE_STACK_REG(kNumberTypeFloat64, double),        SEQUENCE_STACK_REG(kNumberTypeComplex64, complex64),
    SEQUENCE_STACK_REG(kNumberTypeComplex128, complex128), SEQUENCE_STACK_REG(kNumberTypeBool, bool)};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SequenceStack, SequenceStackFwdCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
