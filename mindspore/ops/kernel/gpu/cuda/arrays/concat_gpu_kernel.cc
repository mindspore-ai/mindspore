/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/concat_gpu_kernel.h"
#include <utility>
#include <map>
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "ops_utils/op_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/check_convert_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/concat_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

#define CONCAT_GPU_KERNEL_ATTR(input_type, real_type)    \
  {                                                      \
    KernelAttr()                                         \
      .AddAllSameAttr(true, 1)                           \
      .AddInputAttr(input_type)                          \
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
      .AddOutputAttr(input_type),                        \
      &ConcatGpuKernelMod::LaunchKernel<real_type>       \
  }

const std::vector<std::pair<KernelAttr, ConcatGpuKernelMod::KernelRunFunc>> &ConcatGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ConcatGpuKernelMod::KernelRunFunc>> func_list = {
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeComplex128, Complex<double>),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeComplex64, Complex<float>),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeFloat64, double),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeFloat32, float),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeFloat16, half),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeInt64, int64_t),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeInt32, int32_t),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeInt16, int16_t),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeInt8, int8_t),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeUInt64, uint64_t),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeUInt32, uint),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeUInt16, uint16_t),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeUInt8, uint8_t),
    CONCAT_GPU_KERNEL_ATTR(kNumberTypeBool, bool)};
  return func_list;
}

bool ConcatGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  stream_ptr_ = stream_ptr;
  return kernel_func_(this, inputs, workspace, outputs);
}

template <typename T>
bool ConcatGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &workspace,
                                      const std::vector<KernelTensor *> &outputs) {
  if (input_tensor_num_ == 0) {
    return true;
  }
  T *output = GetDeviceAddress<T>(outputs, 0);
  T **inputs_device = GetDeviceAddress<T *>(workspace, 0);
  int *len_axis_device = GetDeviceAddress<int>(workspace, 1);
  for (size_t i = 0; i < input_tensor_num_; i++) {
    auto input_index = not_null_input_index_[i];
    inputs_host_[i] = GetDeviceAddress<T>(inputs, input_index);
  }
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(inputs_device, inputs_host_.data(), sizeof(T *) * input_tensor_num_, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "ConcatV2 opt cudaMemcpyAsync inputs failed");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(len_axis_device, len_axis_.data(), sizeof(int) * input_tensor_num_, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "ConcatV2 opt cudaMemcpyAsync length on axis failed");
  output_size_ = output_size_list_[0] / sizeof(T);
  auto status = ConcatKernel(output_size_, SizeToInt(input_tensor_num_), all_size_before_axis_, all_size_axis_,
                             len_axis_device, inputs_device, output, reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

bool ConcatGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return true;
}

int ConcatGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  MS_CHECK_VALUE(inputs.size() > 1, CheckAndConvertUtils::FormatCheckIntegerMsg(kernel_name_, inputs.size(),
                                                                                kGreaterThan, 1, primitive()));
  input_tensor_num_ = inputs.size() - 1;
  len_axis_.resize(input_tensor_num_);
  auto input_0_shape = inputs[0]->GetDeviceShapeVector();
  int dims = SizeToInt(input_0_shape.size());
  axis_ = LongToInt(inputs[input_tensor_num_]->GetValueWithCheck<int64_t>());
  MS_CHECK_VALUE(-dims <= axis_ && axis_ < dims,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis_, kIncludeLeft, {-dims, dims}, primitive()));
  if (axis_ < 0) {
    axis_ += dims;
  }

  auto input_format = mindspore::FormatEnumToString(inputs[0]->format());
  std::string origin_data_format = kOpFormat_DEFAULT;
  if (primitive_->HasAttr(kAttrFormat)) {
    origin_data_format = GetValue<std::string>(primitive_->GetAttr(kAttrFormat));
  }
  axis_ = AxisTransform(origin_data_format, input_format, axis_);

  not_null_input_index_.clear();
  len_axis_.clear();
  for (size_t i = 0; i < input_tensor_num_; i++) {
    auto input_shape = inputs[i]->GetDeviceShapeVector();
    auto is_null_input = CHECK_NULL_INPUT(input_shape);
    if (!is_null_input) {
      not_null_input_index_.push_back(i);
      len_axis_.push_back(LongToInt(input_shape[axis_]));
    }
  }
  input_tensor_num_ = not_null_input_index_.size();
  workspace_size_list_.push_back(sizeof(void *) * input_tensor_num_);
  workspace_size_list_.push_back(sizeof(int) * input_tensor_num_);
  inputs_host_.resize(input_tensor_num_);

  auto output_shape = outputs[0]->GetDeviceShapeVector();
  all_size_before_axis_ = 1;
  all_size_axis_ = 1;
  for (int i = 0; i < SizeToInt(output_shape.size()); i++) {
    if (i > axis_) {
      all_size_before_axis_ *= LongToInt(output_shape[i]);
      all_size_axis_ *= LongToInt(output_shape[i]);
    }
    if (i == axis_) {
      all_size_before_axis_ *= LongToInt(output_shape[i]);
    }
  }
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Concat, ConcatGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
