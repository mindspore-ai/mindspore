/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/cumsum_cpu_kernel.h"

#include <thread>

#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCumSumInputsNum = 1;
constexpr size_t kCumSumOutputsNum = 1;
}  // namespace

void CumSumCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  axis_ = LongToInt(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS));
  dst_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  exclusive_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, EXCLUSIVE);
  reverse_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, REVERSE);
  int input_dim_length = SizeToInt(shape_.size());
  if (axis_ >= input_dim_length) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", 'axis' should be less than the length of 'input' dimension, but got 'axis': " << axis_
                      << " and the length of 'input' dimension: " << input_dim_length;
  }
  while (axis_ < 0) {
    axis_ += input_dim_length;
  }
}

template <typename T>
void CumSumCPUKernel::InitWorkspaceSize() {
  input_size_0_ = sizeof(T);
  for (size_t i = 0; i < shape_.size(); i++) {
    input_size_0_ *= shape_[i];
  }
  (void)workspace_size_list_.emplace_back(input_size_0_);
}

void CumSumCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  if (dtype_ == kNumberTypeFloat32) {
    InitWorkspaceSize<float_t>();
  } else if (dtype_ == kNumberTypeFloat16) {
    InitWorkspaceSize<float16>();
  } else if (dtype_ == kNumberTypeInt32) {
    InitWorkspaceSize<int32_t>();
  } else if (dtype_ == kNumberTypeInt8) {
    InitWorkspaceSize<int8_t>();
  } else if (dtype_ == kNumberTypeUInt8) {
    InitWorkspaceSize<uint8_t>();
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", the dtype of input should be in (float16, float32, uint8, int8, int32) on CPU, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
}

bool CumSumCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> &workspace,
                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCumSumInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCumSumOutputsNum, kernel_name_);
  Reshape();
  if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int32_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeInt8) {
    LaunchKernel<int8_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeUInt8) {
    LaunchKernel<uint8_t>(inputs, workspace, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", the dtype of input should be in (float16, float32, uint8, int8, int32) on CPU, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

void CumSumCPUKernel::Reshape() {
  dims_[0] = 1;
  dims_[1] = shape_[IntToSize(axis_)];
  dims_[2] = 1;
  for (size_t i = 0; i < IntToSize(axis_); i++) {
    dims_[0] *= shape_[i];
  }
  for (size_t i = IntToSize(axis_) + 1; i < shape_.size(); i++) {
    dims_[2] *= shape_[i];
  }
  stride_ = dims_[1] * dims_[2];
  stride2_ = dims_[2];
}

template <typename T>
void CumSumCPUKernel::LeftMove(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                               size_t stride2, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    size_t k1 = i / dim2 % dim0;
    size_t k2 = i % dim2;
    size_t offset = k1 * stride + k2;
    for (size_t j = 0; j < dim1; ++j) {
      size_t read_index = j * stride2 + offset;
      if (j == 0) {
        output[read_index] = (T)0;
      } else {
        size_t read_index2 = (j - 1) * stride2 + offset;
        output[read_index] = input[read_index2];
      }
    }
  }
}

template <typename T>
void CumSumCPUKernel::RightMove(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                                size_t stride2, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    size_t k1 = i / dim2 % dim0;
    size_t k2 = i % dim2;
    size_t offset = k1 * stride + k2;
    for (int j = SizeToInt(dim1 - 1); j >= 0; --j) {
      size_t read_index = j * stride2 + offset;
      if (j == SizeToInt(dim1 - 1)) {
        output[read_index] = (T)0;
      } else {
        size_t read_index2 = (j + 1) * stride2 + offset;
        output[read_index] = input[read_index2];
      }
    }
  }
}

template <typename T>
void CumSumCPUKernel::Copy(T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                           size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    size_t k1 = i / dim2 % dim0;
    size_t k2 = i % dim2;
    size_t offset = k1 * stride + k2;
    for (size_t j = 0; j < dim1; ++j) {
      size_t read_index = j * stride2 + offset;
      input[read_index] = output[read_index];
    }
  }
}

template <typename T>
void CumSumCPUKernel::CumSumKernelReverse(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2,
                                          size_t stride, size_t stride2, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    size_t k1 = i / dim2 % dim0;
    size_t k2 = i % dim2;
    size_t offset = k1 * stride + k2;
    for (int j = SizeToInt(dim1 - 1); j >= 0; --j) {
      size_t read_index = j * stride2 + offset;
      if (j == SizeToInt(dim1 - 1)) {
        output[read_index] = input[read_index];
      } else {
        size_t read_index2 = (j + 1) * stride2 + offset;
        output[read_index] = output[read_index2] + input[read_index];
      }
    }
  }
}

template <typename T>
void CumSumCPUKernel::CumSumKernel(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                                   size_t stride2, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    size_t k1 = i / dim2 % dim0;
    size_t k2 = i % dim2;
    size_t offset = k1 * stride + k2;
    for (size_t j = 0; j < dim1; ++j) {
      size_t read_index = j * stride2 + offset;
      if (j == 0) {
        output[read_index] = input[read_index];
      } else {
        size_t read_index2 = (j - 1) * stride2 + offset;
        output[read_index] = output[read_index2] + input[read_index];
      }
    }
  }
}

template <typename T>
void CumSumCPUKernel::LaunchCumSum(const T *input, T *output, T *workspace, size_t start, size_t end) const {
  start = start / dims_[1];
  end = end / dims_[1];
  if (exclusive_) {
    if (reverse_) {
      RightMove(input, output, dims_[0], dims_[1], dims_[2], stride_, stride2_, start, end);
      Copy(workspace, output, dims_[0], dims_[1], dims_[2], stride_, stride2_, start, end);
      CumSumKernelReverse(workspace, output, dims_[0], dims_[1], dims_[2], stride_, stride2_, start, end);
    } else {
      LeftMove(input, output, dims_[0], dims_[1], dims_[2], stride_, stride2_, start, end);
      Copy(workspace, output, dims_[0], dims_[1], dims_[2], stride_, stride2_, start, end);
      CumSumKernel(workspace, output, dims_[0], dims_[1], dims_[2], stride_, stride2_, start, end);
    }
  } else {
    if (reverse_) {
      CumSumKernelReverse(input, output, dims_[0], dims_[1], dims_[2], stride_, stride2_, start, end);
    } else {
      CumSumKernel(input, output, dims_[0], dims_[1], dims_[2], stride_, stride2_, start, end);
    }
  }
}

template <typename T>
void CumSumCPUKernel::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &workspace,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *ws = reinterpret_cast<T *>(workspace[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  // multithreading
  size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(T)) : 1;
  auto task = [this, &input, &output, &ws](size_t start, size_t end) {
    LaunchCumSum<T>(input, output, ws, start, end);
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
}
}  // namespace kernel
}  // namespace mindspore
