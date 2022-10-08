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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/range_kernel.h"
#include <cmath>
#include <thread>
#include <numeric>
#include <vector>
#include <functional>
#include "common/atomic_op.h"
#include "aicpu_sharder/aicpu_sharder.h"
#include "proto/aicpu_tensor.pb.h"

namespace aicpu {
namespace {
constexpr auto kAddressSize = 4;
constexpr auto kDim0 = 0;
constexpr auto kDim1 = 1;
constexpr auto kDim2 = 2;
constexpr auto kDim3 = 3;

template <typename T>
T Sign(T num) {
  if (num > static_cast<T>(0.0)) {
    return static_cast<T>(1.0);
  } else if (num == static_cast<T>(0.0)) {
    return static_cast<T>(0.0);
  } else {
    return static_cast<T>(-1.0);
  }
}
}  // namespace

template <typename T>
uint32_t RangeKernel::RangeTask() {
  if (io_addrs_.size() != kAddressSize) {
    AICPU_LOGE("RangeKernel's address is invalid");
    return kAicpuKernelStateFailed;
  }
  auto start = reinterpret_cast<T *>(io_addrs_[kDim0])[0];
  auto limit = reinterpret_cast<T *>(io_addrs_[kDim1])[0];
  auto delta = reinterpret_cast<T *>(io_addrs_[kDim2])[0];
  auto output = reinterpret_cast<T *>(io_addrs_[kDim3]);
  if (Sign(delta) * Sign(limit - start) >= 0) {
    size_t output_size = 0;
    if (std::is_integral<T>::value) {
      output_size = static_cast<size_t>((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta));
    } else {
      output_size = static_cast<size_t>(std::ceil((limit - start) / delta));
    }
    for (size_t index = 0; index < output_size; index++) {
      output[index] = delta * index + start;
    }
  } else {
    return kAicpuKernelStateInvalid;
  }
  return kAicpuKernelStateSucess;
}

uint32_t RangeKernel::ParseKernelParam() {
  aicpuops::Tensor input_tensor = node_def_.inputs(kDim0);
  index_type_ = static_cast<aicpuops::DataType>(input_tensor.tensor_type());
  return kAicpuKernelStateSucess;
}

uint32_t RangeKernel::DoCompute() {
  switch (index_type_) {
    case aicpuops::DataType::MS_INT32:
      return RangeTask<int>();
    case aicpuops::DataType::MS_INT64:
      return RangeTask<int64_t>();
    case aicpuops::DataType::MS_FLOAT32:
      return RangeTask<float>();
    case aicpuops::DataType::MS_FLOAT64:
      return RangeTask<double>();
    default:
      return kAicpuKernelStateInvalid;
  }
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t Range(void *param) {
  aicpu::RangeKernel range_kernel;
  return range_kernel.Compute(param);
}
}
