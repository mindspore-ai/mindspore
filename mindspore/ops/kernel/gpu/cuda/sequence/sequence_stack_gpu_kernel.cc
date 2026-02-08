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

#include "kernel/gpu/cuda/sequence/sequence_stack_gpu_kernel.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <algorithm>
#include <utility>
#include <complex>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/pack.cuh"
#include "mindspore/ops/infer/sequence_stack.h"
#include "ir/format_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 1;
constexpr int kOutputsNum = 1;
}  // namespace

bool SequenceStackGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int SequenceStackGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  workspace_size_list_.clear();
  tuple_shape_ = inputs[0]->GetShapeVector();
  if (tuple_shape_.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the input tuple size must greater 0";
  }
  std::vector<int64_t> shape_vec_item;
  std::copy(tuple_shape_.begin() + 1, tuple_shape_.end(), std::back_inserter(shape_vec_item));
  axis_ = GetValue<int64_t>(primitive_->GetAttr(ops::kAxis));
  if (axis_ < 0) {
    axis_ += (SizeToInt(shape_vec_item.size()) + 1);
  }
  auto origin_data_format = kOpFormat_DEFAULT;
  auto input_format = GetFormatFromEnumToStr(inputs[0]->format());
  axis_ = AxisTransform(origin_data_format, input_format, axis_);
  input_num_ = tuple_shape_[0];
  inputs_host_.resize(input_num_);
  dims_behind_axis_ = 1;
  for (size_t i = IntToSize(axis_); i < shape_vec_item.size(); i++) {
    dims_behind_axis_ *= static_cast<size_t>(shape_vec_item[i]);
  }
  workspace_size_list_.push_back(sizeof(void *) * input_num_);
  auto output_shape = outputs[0]->GetShapeVector();
  output_size_ = 1;
  for (size_t i = 0; i < output_shape.size(); i++) {
    output_size_ *= static_cast<size_t>(output_shape[i]);
  }
  return KRET_OK;
}

template <typename T>
bool SequenceStackGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs) {
  const auto input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);
  T **inputs_array = GetDeviceAddress<T *>(workspace, 0);
  size_t element_num = outputs[0]->size() / sizeof(T) / input_num_;
  for (int i = 0; i < input_num_; i++) {
    T *tmp_addr = input_addr + i * element_num;
    inputs_host_[i] = tmp_addr;
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(inputs_array, inputs_host_.data(), sizeof(T *) * input_num_, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "SequenceStack opt cudaMemcpyAsync inputs failed");
  auto status = PackKernel(output_size_, input_num_, dims_behind_axis_, inputs_array, output,
                           reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define SEQUENCE_STACK_KERNEL_REG(ms_type, builtin_type)                                              \
  {                                                                                                   \
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kObjectTypeTuple, ms_type).AddOutputAttr(ms_type), \
      &SequenceStackGpuKernelMod::LaunchKernel<builtin_type>                                          \
  }

const SequenceStackGpuKernelMod::FuncList &SequenceStackGpuKernelMod::GetFuncList() const {
  static const FuncList func_list = {SEQUENCE_STACK_KERNEL_REG(kNumberTypeInt8, int8_t),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeInt16, int16_t),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeInt32, int32_t),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeInt64, int64_t),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeUInt8, uint8_t),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeUInt16, uint16_t),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeUInt32, uint32_t),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeUInt64, uint64_t),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeFloat16, half),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeFloat32, float),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeFloat64, double),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeComplex64, Complex<float>),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeComplex128, Complex<double>),
                                     SEQUENCE_STACK_KERNEL_REG(kNumberTypeBool, bool)};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SequenceStack, SequenceStackGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
