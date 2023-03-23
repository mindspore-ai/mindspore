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

#include "plugin/device/cpu/kernel/sequence/sequence_addn_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/add_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 1;
constexpr int kOutputsNum = 1;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

template <typename T>
void Add(const T *in0, const T *in1, T *out, int start, int end) {
  for (int index = start; index < end; index++) {
    out[index] = in0[index] + in1[index];
  }
}

template <>
void Add(const int *in_0, const int *in_1, int *out, int start, int end) {
  int ret = ElementAddInt(in_0 + start, in_1 + start, out + start, end - start);
  if (ret != NNACL_OK) {
    MS_LOG(EXCEPTION) << "For 'AddN', AddInt failed.";
  }
}

template <>
void Add(const float *in_0, const float *in_1, float *out, int start, int end) {
  int ret = ElementAdd(in_0 + start, in_1 + start, out + start, end - start);
  if (ret != NNACL_OK) {
    MS_LOG(EXCEPTION) << "For 'AddN', AddFloat failed.";
  }
}
}  // namespace

bool SequenceAddNCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int SequenceAddNCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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

  return KRET_OK;
}

template <typename T>
bool SequenceAddNCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                            const std::vector<AddressPtr> &outputs) {
  size_t elements_num = outputs[0]->size / sizeof(T);
  const auto input_0 = reinterpret_cast<T *>(inputs[0]->addr);
  auto input_1 = input_0 + elements_num;

  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  auto task_0 = std::bind(Add<T>, input_0, input_1, output, std::placeholders::_1, std::placeholders::_2);
  ParallelLaunchAutoSearch(task_0, elements_num, this, &parallel_search_info_);

  for (int64_t index = 2; index < tuple_shape_[0]; ++index) {
    input_1 += elements_num;
    auto task = std::bind(Add<T>, input_1, output, output, std::placeholders::_1, std::placeholders::_2);
    ParallelLaunchAutoSearch(task, elements_num, this, &parallel_search_info_);
  }
  return true;
}

#define SEQUENCE_ADDN_REG(ms_type, builtin_type)                                                   \
  {KernelAttr().AddInputAttr(kObjectTypeTuple, ms_type).AddOutputAttr(kObjectTypeNumber, ms_type), \
   &SequenceAddNCpuKernelMod::LaunchKernel<builtin_type>},                                         \
  {                                                                                                \
    KernelAttr().AddInputAttr(kObjectTypeTuple, ms_type).AddOutputAttr(ms_type),                   \
      &SequenceAddNCpuKernelMod::LaunchKernel<builtin_type>                                        \
  }

const SequenceAddNCpuKernelMod::FuncList &SequenceAddNCpuKernelMod::GetFuncList() const {
  static const FuncList func_list = {
    SEQUENCE_ADDN_REG(kNumberTypeInt8, int8_t),          SEQUENCE_ADDN_REG(kNumberTypeInt16, int16_t),
    SEQUENCE_ADDN_REG(kNumberTypeInt32, int32_t),        SEQUENCE_ADDN_REG(kNumberTypeInt64, int64_t),
    SEQUENCE_ADDN_REG(kNumberTypeUInt8, uint8_t),        SEQUENCE_ADDN_REG(kNumberTypeUInt16, uint16_t),
    SEQUENCE_ADDN_REG(kNumberTypeUInt32, uint32_t),      SEQUENCE_ADDN_REG(kNumberTypeUInt64, uint64_t),
    SEQUENCE_ADDN_REG(kNumberTypeFloat16, float16),      SEQUENCE_ADDN_REG(kNumberTypeFloat32, float),
    SEQUENCE_ADDN_REG(kNumberTypeFloat64, double),       SEQUENCE_ADDN_REG(kNumberTypeComplex64, complex64),
    SEQUENCE_ADDN_REG(kNumberTypeComplex128, complex128)};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SequenceAddN, SequenceAddNCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
