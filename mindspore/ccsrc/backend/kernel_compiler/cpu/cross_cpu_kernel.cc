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
#include "backend/kernel_compiler/cpu/cross_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace {
const size_t kDataSizeThreshold = 4 * 1024;

#define CROSS_COMPUTE_CASE(DTYPE, TYPE)        \
  case (DTYPE): {                              \
    ret = LaunchKernel<TYPE>(inputs, outputs); \
    break;                                     \
  }
}  // namespace

namespace mindspore {
namespace kernel {
void CrossCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  input1_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input2_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  input1_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  dim_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "dim");
  int64_t default_dim = -65530;
  if (dim_ == default_dim) {
    size_t dim_size_value = 3;
    for (size_t i = 0; i < input1_shape_.size(); i++) {
      if (input1_shape_[i] == dim_size_value) {
        dim_ = i;
        break;
      }
      if (i == input1_shape_.size() - 1 && input1_shape_[i] != dim_size_value) {
        MS_EXCEPTION(ValueError) << "The size of inputs dim should be 3,but got" << input1_shape_[i];
      }
    }
  }
  if (dim_ < 0) {
    dim_ = static_cast<int64_t>(input1_shape_.size()) + dim_;
  }
}

bool CrossCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                               const std::vector<kernel::AddressPtr> &outputs) {
  bool ret = true;
  switch (input1_dtype_) {
    CROSS_COMPUTE_CASE(kNumberTypeInt8, int8_t)
    CROSS_COMPUTE_CASE(kNumberTypeInt16, int16_t)
    CROSS_COMPUTE_CASE(kNumberTypeInt32, int32_t)
    CROSS_COMPUTE_CASE(kNumberTypeInt64, int64_t)
    CROSS_COMPUTE_CASE(kNumberTypeUInt8, uint8_t)
    CROSS_COMPUTE_CASE(kNumberTypeUInt16, uint16_t)
    CROSS_COMPUTE_CASE(kNumberTypeUInt32, uint32_t)
    CROSS_COMPUTE_CASE(kNumberTypeUInt64, uint64_t)
    CROSS_COMPUTE_CASE(kNumberTypeFloat16, float16)
    CROSS_COMPUTE_CASE(kNumberTypeFloat32, float)
    CROSS_COMPUTE_CASE(kNumberTypeFloat64, double)
    CROSS_COMPUTE_CASE(kNumberTypeComplex64, std::complex<float>)
    CROSS_COMPUTE_CASE(kNumberTypeComplex128, std::complex<double>)
    default:
      MS_EXCEPTION(TypeError) << "Unsupported input data type: " << input1_dtype_;
      ret = false;
  }
  return ret;
}

template <typename T>
bool CrossCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  auto input1_data_addr = reinterpret_cast<T *>(inputs[0]->addr);
  size_t tmp = 1;
  for (size_t i = 0; i < input1_shape_.size(); i++) {
    tmp = tmp * input1_shape_[i];
  }
  size_t input1_data_num = tmp;
  auto input2_data_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto output_data_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t total = input1_data_num / 3;
  const size_t n = input1_shape_.size();
  std::vector<size_t> a_stride(n);
  size_t stride_tmp = 1;
  for (int64_t i = static_cast<int64_t>(n - 1); i >= 0; i--) {
    a_stride[LongToSize(i)] = stride_tmp;
    stride_tmp *= input1_shape_[i];
  }
  size_t input1_data_stride = a_stride[dim_];
  std::vector<size_t> b_stride(n);
  stride_tmp = 1;
  for (int64_t i = static_cast<int64_t>(n - 1); i >= 0; i--) {
    b_stride[LongToSize(i)] = stride_tmp;
    stride_tmp *= input2_shape_[i];
  }
  size_t input2_data_stride = b_stride[dim_];
  std::vector<size_t> r_stride(n);
  stride_tmp = 1;
  for (int64_t i = static_cast<int64_t>(n - 1); i >= 0; i--) {
    r_stride[LongToSize(i)] = stride_tmp;
    stride_tmp *= output_shape_[i];
  }
  size_t output_data_stride = r_stride[dim_];
  const int64_t pos = 2;
  auto cross_shard = [this, &a_stride, &b_stride, &r_stride, &output_data_addr, &input1_data_addr, &input2_data_addr,
                      &output_data_stride, &input1_data_stride, &input2_data_stride](size_t start, size_t end) {
    const size_t input1_data_dim = input1_shape_.size();
    std::vector<size_t> position_in_dims(input1_data_dim);
    size_t index_in_curr_dim = start;
    size_t input1_data_start = 0;
    size_t input2_data_start = 0;
    size_t output_data_start = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(input1_data_dim); i++) {
      if (i == static_cast<int64_t>(dim_)) continue;
      position_in_dims[i] = index_in_curr_dim % input1_shape_[i];
      input1_data_start += (index_in_curr_dim % input1_shape_[i]) * a_stride[i];
      input2_data_start += (index_in_curr_dim % input2_shape_[i]) * b_stride[i];
      output_data_start += (index_in_curr_dim % output_shape_[i]) * r_stride[i];
      index_in_curr_dim = index_in_curr_dim / input1_shape_[i];
    }
    while (start < end) {
      output_data_addr[output_data_start + 0 * output_data_stride] =
        input1_data_addr[input1_data_start + 1 * input1_data_stride] *
          input2_data_addr[input2_data_start + pos * input2_data_stride] -
        input1_data_addr[input1_data_start + pos * input1_data_stride] *
          input2_data_addr[input2_data_start + 1 * input2_data_stride];
      output_data_addr[output_data_start + 1 * output_data_stride] =
        input1_data_addr[input1_data_start + pos * input1_data_stride] *
          input2_data_addr[input2_data_start + 0 * input2_data_stride] -
        input1_data_addr[input1_data_start + 0 * input1_data_stride] *
          input2_data_addr[input2_data_start + pos * input2_data_stride];
      output_data_addr[output_data_start + pos * output_data_stride] =
        input1_data_addr[input1_data_start + 0 * input1_data_stride] *
          input2_data_addr[input2_data_start + 1 * input2_data_stride] -
        input1_data_addr[input1_data_start + 1 * input1_data_stride] *
          input2_data_addr[input2_data_start + 0 * input2_data_stride];
      start++;
      for (int64_t i = 0; i < static_cast<int64_t>(input1_data_dim); i++) {
        if (i == static_cast<int64_t>(dim_)) {
          continue;
        }
        position_in_dims[i]++;
        input1_data_start += a_stride[i];
        input2_data_start += b_stride[i];
        output_data_start += r_stride[i];
        if (position_in_dims[i] == input1_shape_[i] && i != static_cast<int64_t>(input1_shape_.size()) - 1) {
          input1_data_start -= position_in_dims[i] * a_stride[i];
          input2_data_start -= position_in_dims[i] * b_stride[i];
          output_data_start -= position_in_dims[i] * r_stride[i];
          position_in_dims[i] = 0;
        } else {
          break;
        }
      }
    }
  };
  if (total * sizeof(T) < kDataSizeThreshold) {
    cross_shard(0, total);
  } else {
    CPUKernelUtils::ParallelFor(cross_shard, total);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
