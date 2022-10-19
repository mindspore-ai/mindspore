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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/cum_prod_kernel.h"
#include <Eigen/Dense>
#include <unordered_set>
#include "proto/aicpu_tensor.pb.h"

namespace aicpu {
namespace {
constexpr auto kExclusive = "exclusive";
constexpr auto kReverse = "reverse";
constexpr size_t kDim0 = 0;
constexpr size_t kDim1 = 1;
constexpr size_t kDim2 = 2;
constexpr size_t kDim3 = 3;
constexpr size_t kCumProdAddressSize = 3;
}  // namespace

// Init Kernel
uint32_t CumProdKernel::ParseKernelParam() {
  aicpuops::Tensor x = node_def_.inputs(kDim0);
  dtype_ = static_cast<aicpuops::DataType>(x.tensor_type());
  shape_.clear();
  const auto &x_shape = x.tensor_shape();
  for (int i = 0; i < x_shape.dim_size(); ++i) {
    shape_.push_back(SizeToInt(x_shape.dim(i).size()));
  }
  aicpuops::Tensor out = node_def_.outputs(kDim0);
  dst_shape_.clear();
  const auto &out_shape = out.tensor_shape();
  for (int i = 0; i < out_shape.dim_size(); ++i) {
    dst_shape_.push_back(SizeToInt(out_shape.dim(i).size()));
  }
  auto node_def_attrs = node_def_.attrs();
  exclusive_ = node_def_attrs[kExclusive].i();
  reverse_ = node_def_attrs[kReverse].i();
  return kAicpuKernelStateSucess;
}

// Get Support Type
uint32_t CumProdKernel::DoCompute() {
  switch (dtype_) {
    case aicpuops::DataType::MS_UINT8:
      return CumProdTask<uint8_t>();
    case aicpuops::DataType::MS_UINT16:
      return CumProdTask<uint16_t>();
    case aicpuops::DataType::MS_UINT32:
      return CumProdTask<uint32_t>();
    case aicpuops::DataType::MS_UINT64:
      return CumProdTask<uint64_t>();
    case aicpuops::DataType::MS_INT8:
      return CumProdTask<int8_t>();
    case aicpuops::DataType::MS_INT16:
      return CumProdTask<int16_t>();
    case aicpuops::DataType::MS_INT32:
      return CumProdTask<int>();
    case aicpuops::DataType::MS_INT64:
      return CumProdTask<int64_t>();
    case aicpuops::DataType::MS_FLOAT16:
      return CumProdTask<Eigen::half>();
    case aicpuops::DataType::MS_FLOAT32:
      return CumProdTask<float>();
    case aicpuops::DataType::MS_FLOAT64:
      return CumProdTask<double>();
    default:
      return kAicpuKernelStateInvalid;
  }
}

// Launch Kernel
template <typename T>
uint32_t CumProdKernel::CumProdTask() {
  if (io_addrs_.size() != kCumProdAddressSize) {
    AICPU_LOGE("CumProdKernel's address is invalid");
    return kAicpuKernelStateFailed;
  }
  auto x_addr = reinterpret_cast<T *>(io_addrs_[kDim0]);
  auto axis_addr = reinterpret_cast<int64_t *>(io_addrs_[kDim1]);
  if (axis_addr == nullptr) {
    AICPU_LOGE("CumProdKernel's axis must be integer.");
    return kAicpuKernelStateFailed;
  }
  axis_ = IntToSize(*axis_addr);
  Reshape();
  auto out_addr = reinterpret_cast<T *>(io_addrs_[kDim2]);
  LaunchCumProd<T>(x_addr, out_addr);
  return kAicpuKernelStateSucess;
}

void CumProdKernel::Reshape() {
  while (axis_ < 0) {
    axis_ += shape_.size();
  }
  dims_[kDim1] = IntToSize(shape_[axis_]);
  for (size_t i = 0; i < axis_; i++) {
    dims_[kDim0] *= IntToSize(shape_[i]);
  }
  for (size_t i = axis_ + 1; i < shape_.size(); i++) {
    dims_[kDim2] *= IntToSize(shape_[i]);
  }
  stride_ = dims_[kDim1] * dims_[kDim2];
  stride2_ = dims_[kDim2];
  lens_ = shape_[kDim0] > 0 ? IntToSize(shape_[kDim0]) : 1;
}

template <typename T>
void CumProdKernel::LaunchCumProd(const T *input, T *output) const {
  if (exclusive_) {
    if (reverse_) {
      RightMove(input, output);
      CumProdKernelReverse(input, output);
    } else {
      LeftMove(input, output);
      CumProd(input, output);
    }
  } else {
    if (reverse_) {
      CumProdKernelReverse(input, output);
    } else {
      CumProd(input, output);
    }
  }
}

template <typename T>
void CumProdKernel::CumProd(const T *input, T *output) const {
  for (size_t i = 0; i < lens_; ++i) {
    size_t k1 = 0;
    size_t k2 = 0;
    if (dims_[kDim2] != 0 && dims_[kDim0] != 0) {
      k1 = i / dims_[kDim2] % dims_[kDim0];
      k2 = i % dims_[kDim2];
    }
    size_t offset = k1 * stride_ + k2;
    for (size_t j = 0; j < dims_[kDim1]; ++j) {
      size_t read_index = j * stride2_ + offset;
      if (j == 0) {
        output[read_index] = input[read_index];
      } else {
        size_t read_index2 = (j - 1) * stride2_ + offset;
        output[read_index] = output[read_index2] * static_cast<T>(input[read_index]);
      }
    }
  }
}

template <typename T>
void CumProdKernel::CumProdKernelReverse(const T *input, T *output) const {
  for (size_t i = 0; i < lens_; ++i) {
    size_t k1 = 0;
    size_t k2 = 0;
    if (dims_[kDim2] != 0 && dims_[kDim0] != 0) {
      k1 = i / dims_[kDim2] % dims_[kDim0];
      k2 = i % dims_[kDim2];
    }
    size_t offset = k1 * stride_ + k2;
    for (int j = SizeToInt(dims_[kDim1] - 1); j >= 0; --j) {
      size_t read_index = IntToSize(j * stride2_ + offset);
      if (j == SizeToInt(dims_[kDim1] - 1)) {
        output[read_index] = input[read_index];
      } else {
        size_t read_index2 = IntToSize((j + 1) * stride2_ + offset);
        output[read_index] = output[read_index2] * static_cast<T>(input[read_index]);
      }
    }
  }
}

template <typename T>
void CumProdKernel::LeftMove(const T *input, T *output) const {
  for (size_t i = 0; i < lens_; ++i) {
    size_t k1 = 0;
    size_t k2 = 0;
    if (dims_[kDim2] != 0 && dims_[kDim0] != 0) {
      k1 = i / dims_[kDim2] % dims_[kDim0];
      k2 = i % dims_[kDim2];
    }
    size_t offset = k1 * stride_ + k2;
    for (size_t j = 0; j < dims_[kDim1]; ++j) {
      size_t read_index = j * stride2_ + offset;
      if (j == 0) {
        output[read_index] = static_cast<T>(1);
      } else {
        size_t read_index2 = (j - 1) * stride2_ + offset;
        output[read_index] = input[read_index2];
      }
    }
  }
}

template <typename T>
void CumProdKernel::RightMove(const T *input, T *output) const {
  for (size_t i = 0; i < lens_; ++i) {
    size_t k1 = 0;
    size_t k2 = 0;
    if (dims_[kDim2] != 0 && dims_[kDim0] != 0) {
      k1 = i / dims_[kDim2] % dims_[kDim0];
      k2 = i % dims_[kDim2];
    }
    size_t offset = k1 * stride_ + k2;
    for (int j = SizeToInt(dims_[kDim1] - 1); j >= 0; --j) {
      size_t read_index = j * stride2_ + offset;
      if (j == SizeToInt(dims_[kDim1] - 1)) {
        output[read_index] = static_cast<T>(1);
      } else {
        size_t read_index2 = IntToSize((j + 1) * stride2_ + offset);
        output[read_index] = input[read_index2];
      }
    }
  }
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t CumProd(void *param) {
  aicpu::CumProdKernel cum_prod_kernel;
  return cum_prod_kernel.Compute(param);
}
}
