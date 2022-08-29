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

#include "plugin/device/cpu/kernel/cumprod_cpu_kernel.h"

#include <thread>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCumProdInputsNum = 1;
constexpr size_t kCumProdOutputsNum = 1;
constexpr size_t kDimSize2 = 2;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

void CumProdCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  axis_ = LongToInt(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS));
  dst_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  exclusive_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, EXCLUSIVE);
  reverse_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, REVERSE);
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
void CumProdCpuKernelMod::InitWorkspaceSize() {
  input_size_0_ = sizeof(T);
  for (size_t i = 0; i < shape_.size(); i++) {
    input_size_0_ *= static_cast<size_t>(shape_[i]);
  }
  (void)workspace_size_list_.emplace_back(input_size_0_);
}

void CumProdCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  DeprecatedNativeCpuKernelMod::InitInputOutputSize(kernel_node);
  if (dtype_ == kNumberTypeFloat32) {
    InitWorkspaceSize<float_t>();
  } else if (dtype_ == kNumberTypeFloat16) {
    InitWorkspaceSize<float16>();
  } else if (dtype_ == kNumberTypeFloat64) {
    InitWorkspaceSize<double>();
  } else if (dtype_ == kNumberTypeInt32) {
    InitWorkspaceSize<int32_t>();
  } else if (dtype_ == kNumberTypeInt8) {
    InitWorkspaceSize<int8_t>();
  } else if (dtype_ == kNumberTypeUInt8) {
    InitWorkspaceSize<uint8_t>();
  } else if (dtype_ == kNumberTypeUInt16) {
    InitWorkspaceSize<uint16_t>();
  } else if (dtype_ == kNumberTypeUInt32) {
    InitWorkspaceSize<uint32_t>();
  } else if (dtype_ == kNumberTypeUInt64) {
    InitWorkspaceSize<uint64_t>();
  } else if (dtype_ == kNumberTypeInt16) {
    InitWorkspaceSize<int16_t>();
  } else if (dtype_ == kNumberTypeInt64) {
    InitWorkspaceSize<int64_t>();
  } else if (dtype_ == kNumberTypeComplex64) {
    InitWorkspaceSize<std::complex<float>>();
  } else if (dtype_ == kNumberTypeComplex128) {
    InitWorkspaceSize<std::complex<double>>();
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", the dtype of input should be in (int, uint, float, double, complex) on CPU, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
}

bool CumProdCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                 const std::vector<kernel::AddressPtr> &workspace,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCumProdInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCumProdOutputsNum, kernel_name_);
  Reshape();
  if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int32_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeInt8) {
    LaunchKernel<int8_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeUInt8) {
    LaunchKernel<uint8_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeUInt16) {
    LaunchKernel<uint16_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeUInt32) {
    LaunchKernel<uint32_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeUInt64) {
    LaunchKernel<uint64_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeInt16) {
    LaunchKernel<int16_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeComplex64) {
    LaunchKernel<std::complex<float>>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeComplex128) {
    LaunchKernel<std::complex<double>>(inputs, workspace, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", the dtype of input should be in (int, uint, float, double, complex) on CPU, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

void CumProdCpuKernelMod::Reshape() {
  dims_[0] = 1;
  dims_[1] = static_cast<size_t>(shape_[IntToSize(axis_)]);
  dims_[kDimSize2] = 1;
  for (size_t i = 0; i < IntToSize(axis_); i++) {
    dims_[0] *= static_cast<size_t>(shape_[i]);
  }
  for (size_t i = IntToSize(axis_) + 1; i < shape_.size(); i++) {
    dims_[kDimSize2] *= static_cast<size_t>(shape_[i]);
  }
  stride_ = dims_[1] * dims_[kDimSize2];
  stride2_ = dims_[kDimSize2];
}

template <typename T>
void CumProdCpuKernelMod::LeftMove(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                                   size_t stride2, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    size_t k1 = 0;
    size_t k2 = 0;
    if (dim2 != 0 && dim0 != 0) {
      k1 = i / dim2 % dim0;
      k2 = i % dim2;
    }
    size_t offset = k1 * stride + k2;
    for (size_t j = 0; j < dim1; ++j) {
      size_t read_index = static_cast<size_t>(j * stride2 + offset);
      if (j == 0) {
        output[read_index] = (T)1;
      } else {
        size_t read_index2 = (j - 1) * stride2 + offset;
        output[read_index] = input[read_index2];
      }
    }
  }
}

template <typename T>
void CumProdCpuKernelMod::RightMove(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                                    size_t stride2, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    size_t k1 = 0;
    size_t k2 = 0;
    if (dim2 != 0 && dim0 != 0) {
      k1 = i / dim2 % dim0;
      k2 = i % dim2;
    }
    size_t offset = k1 * stride + k2;
    for (int j = SizeToInt(dim1 - 1); j >= 0; --j) {
      size_t read_index = static_cast<size_t>(j * stride2 + offset);
      if (j == SizeToInt(dim1 - 1)) {
        output[read_index] = (T)1;
      } else {
        size_t read_index2 = static_cast<size_t>((j + 1) * stride2 + offset);
        output[read_index] = input[read_index2];
      }
    }
  }
}

template <typename T>
void CumProdCpuKernelMod::Copy(T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                               size_t stride2, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    size_t k1 = 0;
    size_t k2 = 0;
    if (dim2 != 0 && dim0 != 0) {
      k1 = i / dim2 % dim0;
      k2 = i % dim2;
    }
    size_t offset = k1 * stride + k2;
    for (size_t j = 0; j < dim1; ++j) {
      size_t read_index = static_cast<size_t>(j * stride2 + offset);
      input[read_index] = output[read_index];
    }
  }
}

template <typename T>
void CumProdCpuKernelMod::CumProdKernelReverse(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2,
                                               size_t stride, size_t stride2, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    size_t k1 = 0;
    size_t k2 = 0;
    if (dim2 != 0 && dim0 != 0) {
      k1 = i / dim2 % dim0;
      k2 = i % dim2;
    }
    size_t offset = k1 * stride + k2;
    for (int j = SizeToInt(dim1 - 1); j >= 0; --j) {
      size_t read_index = static_cast<size_t>(j * stride2 + offset);
      if (j == SizeToInt(dim1 - 1)) {
        output[read_index] = input[read_index];
      } else {
        size_t read_index2 = static_cast<size_t>((j + 1) * stride2 + offset);
        output[read_index] = output[read_index2] * static_cast<T>(input[read_index]);
      }
    }
  }
}

template <typename T>
void CumProdCpuKernelMod::CumProdKernel(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                                        size_t stride2, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    size_t k1 = 0;
    size_t k2 = 0;
    if (dim2 != 0 && dim0 != 0) {
      k1 = i / dim2 % dim0;
      k2 = i % dim2;
    }
    size_t offset = k1 * stride + k2;
    for (size_t j = 0; j < dim1; ++j) {
      size_t read_index = static_cast<size_t>(j * stride2 + offset);
      if (j == 0) {
        output[read_index] = input[read_index];
      } else {
        size_t read_index2 = (j - 1) * stride2 + offset;
        output[read_index] = output[read_index2] * static_cast<T>(input[read_index]);
      }
    }
  }
}

template <typename T>
void CumProdCpuKernelMod::LaunchCumProd(const T *input, T *output, T *workspace, size_t start, size_t end) const {
  start = start / dims_[1];
  end = end / dims_[1];
  if (exclusive_) {
    if (reverse_) {
      RightMove(input, output, dims_[0], dims_[1], dims_[kDimSize2], stride_, stride2_, start, end);
      Copy(workspace, output, dims_[0], dims_[1], dims_[kDimSize2], stride_, stride2_, start, end);
      CumProdKernelReverse(workspace, output, dims_[0], dims_[1], dims_[kDimSize2], stride_, stride2_, start, end);
    } else {
      LeftMove(input, output, dims_[0], dims_[1], dims_[kDimSize2], stride_, stride2_, start, end);
      Copy(workspace, output, dims_[0], dims_[1], dims_[kDimSize2], stride_, stride2_, start, end);
      CumProdKernel(workspace, output, dims_[0], dims_[1], dims_[kDimSize2], stride_, stride2_, start, end);
    }
  } else {
    if (reverse_) {
      CumProdKernelReverse(input, output, dims_[0], dims_[1], dims_[kDimSize2], stride_, stride2_, start, end);
    } else {
      CumProdKernel(input, output, dims_[0], dims_[1], dims_[kDimSize2], stride_, stride2_, start, end);
    }
  }
}

template <typename T>
void CumProdCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &workspace,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input = static_cast<T *>(inputs[0]->addr);
  auto *ws = static_cast<T *>(workspace[0]->addr);
  auto output = static_cast<T *>(outputs[0]->addr);
  // multithreading
  size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(T)) : 1;
  auto task = [this, &input, &output, &ws](size_t start, size_t end) {
    LaunchCumProd<T>(input, output, ws, start, end);
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
}

std::vector<KernelAttr> CumProdCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64)},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32)},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16)},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8)},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16)},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32)},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64)},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64)},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128)},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8)}};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CumProd, CumProdCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
