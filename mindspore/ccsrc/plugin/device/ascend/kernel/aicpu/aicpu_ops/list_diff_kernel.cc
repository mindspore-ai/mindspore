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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/list_diff_kernel.h"
#include <unordered_set>
#include "proto/aicpu_tensor.pb.h"

namespace aicpu {
namespace {
constexpr size_t kDim0 = 0;
constexpr size_t kDim1 = 1;
constexpr size_t kDim2 = 2;
constexpr size_t kDim3 = 3;
constexpr size_t kListDiffAddressSize = 4;

#define REG_LIST_DIFF_TYPE(data_type, type)          \
  case (data_type): {                                \
    if (idx_type_ == aicpuops::DataType::MS_INT64) { \
      return ListDiffTask<type, int64_t>();          \
    } else {                                         \
      return ListDiffTask<type, int32_t>();          \
    }                                                \
    break;                                           \
  }
}  // namespace

// Launch Kernel
template <typename T, typename Tidx>
uint32_t ListDiffKernel::ListDiffTask() {
  if (io_addrs_.size() != kListDiffAddressSize) {
    AICPU_LOGE("ListDiffKernel's address is invalid");
    return kAicpuKernelStateFailed;
  }

  auto x_addr = reinterpret_cast<T *>(io_addrs_[kDim0]);
  auto y_addr = reinterpret_cast<T *>(io_addrs_[kDim1]);
  auto out_addr = reinterpret_cast<T *>(io_addrs_[kDim2]);
  auto idx_addr = reinterpret_cast<Tidx *>(io_addrs_[kDim3]);

  std::unordered_set<T> y_set;
  y_set.reserve(y_size_);
  for (int64_t i = 0; i < y_size_; ++i) {
    (void)y_set.insert(y_addr[i]);
  }

  // calculate results
  out_size_ = 0;
  for (Tidx i = 0; i < static_cast<Tidx>(x_size_); ++i) {
    if (y_set.count(x_addr[i]) == 0) {
      out_addr[out_size_] = x_addr[i];
      idx_addr[out_size_] = i;
      ++out_size_;
    }
  }

  // update out
  if (output_shape_and_type_.size() < kDim2) {
    AICPU_LOGE("ListDiffKernel's output size is invalid");
    return kAicpuKernelStateFailed;
  }

  output_shape_and_type_[kDim0]->dims[kDim0] = out_size_;  // output
  output_shape_and_type_[kDim1]->dims[kDim0] = out_size_;  // out_idx
  return kAicpuKernelStateSucess;
}

// Init Kernel
uint32_t ListDiffKernel::ParseKernelParam() {
  idx_type_ = static_cast<aicpuops::DataType>(output_shape_and_type_[1]->type);

  aicpuops::Tensor x = node_def_.inputs(kDim0);
  input_type_ = static_cast<aicpuops::DataType>(x.tensor_type());
  x_size_ = 0;
  const auto &x_shape = x.tensor_shape();
  for (int i = 0; i < x_shape.dim_size(); ++i) {
    x_size_ += static_cast<int64_t>(x_shape.dim(i).size());
  }

  aicpuops::Tensor y = node_def_.inputs(kDim1);
  const auto &y_shape = y.tensor_shape();
  y_size_ = 0;
  for (int i = 0; i < y_shape.dim_size(); ++i) {
    y_size_ += static_cast<int64_t>(y_shape.dim(i).size());
  }

  return kAicpuKernelStateSucess;
}

// Get Support Type
uint32_t ListDiffKernel::DoCompute() {
  switch (input_type_) {
    REG_LIST_DIFF_TYPE(aicpuops::DataType::MS_INT32, int32_t)
    REG_LIST_DIFF_TYPE(aicpuops::DataType::MS_INT64, int64_t)
    REG_LIST_DIFF_TYPE(aicpuops::DataType::MS_FLOAT32, float)
    REG_LIST_DIFF_TYPE(aicpuops::DataType::MS_FLOAT64, double)
    default:
      return kAicpuKernelStateInvalid;
  }
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t ListDiff(void *param) {
  aicpu::ListDiffKernel list_diff_kernel;
  return list_diff_kernel.Compute(param);
}
}
