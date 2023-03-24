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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/slice_grad_kernel.h"
#include <vector>
#include <numeric>
#include <unordered_map>
#include <thread>
#include <functional>
#include <algorithm>
#include "Eigen/Core"
#include "proto/aicpu_tensor.pb.h"
#include "aicpu_sharder/aicpu_sharder.h"

namespace aicpu {
namespace {
constexpr size_t kSliceGradInputNum = 4;
constexpr size_t kDim2 = 2;
constexpr size_t kDim3 = 3;

std::vector<int64_t> GetShape(const ::aicpuops::TensorShape &shape) {
  std::vector<int64_t> res;
  for (int i = 0; i < shape.dim_size(); ++i) {
    res.push_back(shape.dim(i).size());
  }
  return res;
}
}  // namespace

bool SliceGradKernel::CheckParams() const {
  if (io_addrs_.size() != kSliceGradInputNum + 1) {
    AICPU_LOGE("For 'SliceGrad', input and output address list's size must be 5, but got %lu", io_addrs_.size());
    return false;
  }

  // check dy shape, N-D
  auto n = dy_shape_.size();
  if (n == 0) {
    AICPU_LOGE("For 'SliceGrad', 'dy' shape can not be empty.");
    return false;
  }

  // check begin shape, 1-D with shape [N]
  if (begin_shape_.size() != 1) {
    AICPU_LOGE("For 'SliceGrad', 'begin' shape rank must be 1, but got %lu", begin_shape_.size());
    return false;
  }
  if (LongToSize(begin_shape_[0]) != n) {
    AICPU_LOGE("For 'SliceGrad', 'begin' shape must be [%lu], but got [%ld]", n, begin_shape_[0]);
    return false;
  }

  // check size shape, 1-D with shape [N]
  if (size_shape_.size() != 1) {
    AICPU_LOGE("For 'SliceGrad', 'size' shape rank must be 1, but got %lu", size_shape_.size());
    return false;
  }
  if (LongToSize(size_shape_[0]) != n) {
    AICPU_LOGE("For 'SliceGrad', 'size' shape must be [%lu], but got [%ld]", n, size_shape_[0]);
    return false;
  }

  // check output shape, N-D
  if (output_shape_.size() != n) {
    AICPU_LOGE("For 'SliceGrad', 'dy' shape and output tensor shape must have same rank, but got %lu vs %lu", n,
               output_shape_.size());
    return false;
  }

  for (size_t i = 0; i < n; ++i) {
    if (dy_shape_[i] <= 0 || dy_shape_[i] > output_shape_[i]) {
      AICPU_LOGE(
        "For 'SliceGrad', it is required that 0 < 'dy' shape[%lu] <= output tensor shape[%lu], but got %ld vs %ld", i,
        i, dy_shape_[i], output_shape_[i]);
      return false;
    }
  }
  return true;
}

bool SliceGradKernel::CheckBeginSizeValue() {
  for (size_t i = 0; i < begin_value_.size(); ++i) {
    if (begin_value_[i] < 0 || begin_value_[i] >= output_shape_[i]) {
      AICPU_LOGE("For 'SliceGrad', 'begin' [%lu] must be in range [0, %ld), but got %ld", i, output_shape_[i],
                 begin_value_[i]);
      return false;
    }
    if (size_value_[i] < 0) {
      size_value_[i] = output_shape_[i] - begin_value_[i];
    }
    if (size_value_[i] != dy_shape_[i]) {
      AICPU_LOGE("For 'SliceGrad', 'size' [%lu] must be equal to 'dy' shape[%lu], but got %ld vs %ld", i, i,
                 size_value_[i], dy_shape_[i]);
      return false;
    }
    if (begin_value_[i] + size_value_[i] > output_shape_[i]) {
      AICPU_LOGE(
        "For 'SliceGrad', 'begin' [%lu] + 'size' [%lu] must be <= output tensor shape[%lu], but got %ld, %ld, %ld", i,
        i, i, begin_value_[i], size_value_[i], output_shape_[i]);
      return false;
    }
  }
  return true;
}

template <typename T, typename S>
uint32_t SliceGradKernel::SliceGradTask() {
  if (!CheckParams()) {
    return kAicpuKernelStateFailed;
  }

  S *dy_addr = reinterpret_cast<S *>(io_addrs_[0]);
  T *begin_addr = reinterpret_cast<T *>(io_addrs_[kDim2]);
  T *size_addr = reinterpret_cast<T *>(io_addrs_[kDim3]);
  S *out_addr = reinterpret_cast<S *>(io_addrs_[kSliceGradInputNum]);

  for (size_t i = 0; i < dy_shape_.size(); ++i) {
    begin_value_.push_back(static_cast<int64_t>(begin_addr[i]));
    size_value_.push_back(static_cast<int64_t>(size_addr[i]));
  }
  if (!CheckBeginSizeValue()) {
    return kAicpuKernelStateFailed;
  }

  // Calc process
  // 1. fill output address with 0
  int64_t output_num = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<int64_t>());
  size_t output_byte = LongToSize(output_num) * sizeof(S);
  if (memset_s(out_addr, output_byte, 0, output_byte) != EOK) {
    AICPU_LOGE("For 'SliceGrad', memset_s on output tensor address failed!");
    return kAicpuKernelStateFailed;
  }

  // 2. copy dy_addr to out_addr
  // case: 1D
  if (dy_shape_.size() == 1) {
    size_t block_byte = dy_shape_[0] * sizeof(S);
    S *out_start_addr = out_addr + begin_value_[0];
    if (memcpy_s(out_start_addr, block_byte, dy_addr, block_byte) != EOK) {
      AICPU_LOGE("For 'SliceGrad', memcpy_s failed!");
      return kAicpuKernelStateFailed;
    }
    return kAicpuKernelStateSucess;
  }
  // case: > 1D (0D already checked inside CheckParams), the last dim will be scheduled as a block
  std::vector<size_t> dy_block_shape;
  (void)std::transform(dy_shape_.begin(), dy_shape_.end() - 1, std::back_inserter(dy_block_shape), LongToSize);

  std::vector<size_t> out_block_shape_acc{1};
  size_t acc = 1;
  for (size_t i = output_shape_.size() - kDim2; i > 0; --i) {
    acc *= LongToSize(output_shape_[i]);
    (void)out_block_shape_acc.insert(out_block_shape_acc.begin(), acc);
  }

  auto block_task = [this, &dy_addr, &out_addr, &dy_block_shape, &out_block_shape_acc](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      size_t k = 0;
      auto a = i;
      for (size_t j = 0; j < dy_block_shape.size(); ++j) {
        size_t m = dy_block_shape.size() - 1 - j;
        auto idx = a % dy_block_shape[m] + begin_value_[m];
        a /= dy_block_shape[m];
        k += idx * out_block_shape_acc[m];
      }
      auto block_sz = LongToSize(dy_shape_.back());
      size_t block_byte = block_sz * sizeof(S);
      S *dy_start_addr = dy_addr + i * block_sz;
      S *out_start_addr = out_addr + k * LongToSize(output_shape_.back()) + begin_value_.back();
      if (memcpy_s(out_start_addr, block_byte, dy_start_addr, block_byte) != EOK) {
        AICPU_LOGE("For 'SliceGrad', memcpy_s failed! Current block index is %lu", i);
        return;
      }
    }
  };

  int64_t block_num = 1;
  for (size_t i = 0; i < dy_shape_.size() - 1; ++i) {
    block_num *= dy_shape_[i];
  }
  const int64_t per_unit_size = block_num / std::thread::hardware_concurrency();
  ParallelFor(block_num, per_unit_size, block_task);

  return kAicpuKernelStateSucess;
}

uint32_t SliceGradKernel::ParseKernelParam() {
  // check input tensor number
  if (IntToSize(node_def_.inputs_size()) != kSliceGradInputNum) {
    AICPU_LOGE("For 'SliceGrad', input tensor number must be 4, but got %d", node_def_.inputs_size());
    return kAicpuKernelStateInvalid;
  }

  // dy
  aicpuops::Tensor dy_tensor = node_def_.inputs(0);
  dy_shape_ = GetShape(dy_tensor.tensor_shape());
  dy_type_ = static_cast<aicpuops::DataType>(dy_tensor.tensor_type());

  // begin
  aicpuops::Tensor begin_tensor = node_def_.inputs(SizeToInt(kDim2));
  begin_shape_ = GetShape(begin_tensor.tensor_shape());
  begin_type_ = static_cast<aicpuops::DataType>(begin_tensor.tensor_type());

  // size
  aicpuops::Tensor size_tensor = node_def_.inputs(SizeToInt(kDim3));
  size_shape_ = GetShape(size_tensor.tensor_shape());
  auto size_type = static_cast<aicpuops::DataType>(size_tensor.tensor_type());
  if (size_type != begin_type_) {
    AICPU_LOGE("For 'SliceGrad', 'begin' and 'size' must have same data type.");
    return kAicpuKernelStateInvalid;
  }

  // output
  if (node_def_.outputs_size() != 1) {
    AICPU_LOGE("For 'SliceGrad', output tensor number must be 1, but got %d", node_def_.outputs_size());
    return kAicpuKernelStateInvalid;
  }
  aicpuops::Tensor output_tensor = node_def_.outputs(0);
  output_shape_ = GetShape(output_tensor.tensor_shape());

  return kAicpuKernelStateSucess;
}

uint32_t SliceGradKernel::DoCompute() {
  std::unordered_map<aicpuops::DataType, std::unordered_map<aicpuops::DataType, std::function<uint32_t()>>> func_list;
  // begin type int32
  func_list[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_FLOAT16] =
    std::bind(&SliceGradKernel::SliceGradTask<int32_t, Eigen::half>, this);
  func_list[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_FLOAT32] =
    std::bind(&SliceGradKernel::SliceGradTask<int32_t, float>, this);
  func_list[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_FLOAT64] =
    std::bind(&SliceGradKernel::SliceGradTask<int32_t, double>, this);
  func_list[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_UINT8] =
    std::bind(&SliceGradKernel::SliceGradTask<int32_t, uint8_t>, this);
  func_list[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_UINT16] =
    std::bind(&SliceGradKernel::SliceGradTask<int32_t, uint16_t>, this);
  func_list[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_UINT32] =
    std::bind(&SliceGradKernel::SliceGradTask<int32_t, uint32_t>, this);
  func_list[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_UINT64] =
    std::bind(&SliceGradKernel::SliceGradTask<int32_t, uint64_t>, this);
  func_list[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_INT8] =
    std::bind(&SliceGradKernel::SliceGradTask<int32_t, int8_t>, this);
  func_list[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_INT16] =
    std::bind(&SliceGradKernel::SliceGradTask<int32_t, int16_t>, this);
  func_list[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_INT32] =
    std::bind(&SliceGradKernel::SliceGradTask<int32_t, int32_t>, this);
  func_list[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_INT64] =
    std::bind(&SliceGradKernel::SliceGradTask<int32_t, int64_t>, this);
  // begin type int64
  func_list[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_FLOAT16] =
    std::bind(&SliceGradKernel::SliceGradTask<int64_t, Eigen::half>, this);
  func_list[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_FLOAT32] =
    std::bind(&SliceGradKernel::SliceGradTask<int64_t, float>, this);
  func_list[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_FLOAT64] =
    std::bind(&SliceGradKernel::SliceGradTask<int64_t, double>, this);
  func_list[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_UINT8] =
    std::bind(&SliceGradKernel::SliceGradTask<int64_t, uint8_t>, this);
  func_list[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_UINT16] =
    std::bind(&SliceGradKernel::SliceGradTask<int64_t, uint16_t>, this);
  func_list[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_UINT32] =
    std::bind(&SliceGradKernel::SliceGradTask<int64_t, uint32_t>, this);
  func_list[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_UINT64] =
    std::bind(&SliceGradKernel::SliceGradTask<int64_t, uint64_t>, this);
  func_list[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_INT8] =
    std::bind(&SliceGradKernel::SliceGradTask<int64_t, int8_t>, this);
  func_list[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_INT16] =
    std::bind(&SliceGradKernel::SliceGradTask<int64_t, int16_t>, this);
  func_list[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_INT32] =
    std::bind(&SliceGradKernel::SliceGradTask<int64_t, int32_t>, this);
  func_list[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_INT64] =
    std::bind(&SliceGradKernel::SliceGradTask<int64_t, int64_t>, this);

  if (func_list.find(begin_type_) == func_list.end()) {
    AICPU_LOGE("'SliceGrad' does not support current 'begin' type.");
    return kAicpuKernelStateFailed;
  }
  if (func_list[begin_type_].find(dy_type_) == func_list[begin_type_].end()) {
    AICPU_LOGE("'SliceGrad' does not support current 'dy' type.");
    return kAicpuKernelStateFailed;
  }
  return func_list[begin_type_][dy_type_]();
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t SliceGrad(void *param) {
  aicpu::SliceGradKernel slice_grad_kernel;
  return slice_grad_kernel.Compute(param);
}
}
