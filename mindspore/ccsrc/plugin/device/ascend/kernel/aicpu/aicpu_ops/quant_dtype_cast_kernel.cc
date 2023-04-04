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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/quant_dtype_cast_kernel.h"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <thread>
#include <functional>
#include "proto/aicpu_tensor.pb.h"
#include "aicpu_sharder/aicpu_sharder.h"
#include "mindspore/core/mindapi/base/type_id.h"

namespace aicpu {
namespace {
constexpr size_t C0NUM = 0;
constexpr size_t C1NUM = 1;
constexpr size_t C2NUM = 2;
constexpr size_t C3NUM = 3;
constexpr size_t C4NUM = 4;
constexpr size_t C5NUM = 5;
constexpr size_t C6NUM = 6;

std::vector<int64_t> GetShape(const ::aicpuops::TensorShape &shape) {
  std::vector<int64_t> res;
  for (int i = 0; i < shape.dim_size(); ++i) {
    res.push_back(shape.dim(i).size());
  }
  return res;
}
}  // namespace
bool QuantDTypeCastKernel::CheckParams() const { return true; }

void QuantDTypeCastKernel::FixedBitFloatDequantTask() {
  int8_t *input = reinterpret_cast<int8_t *>(io_addrs_[C0NUM]);
  float *scales = reinterpret_cast<float *>(io_addrs_[C1NUM]);
  int *zps = reinterpret_cast<int *>(io_addrs_[C2NUM]);
  float *mean_corrs = reinterpret_cast<float *>(io_addrs_[C3NUM]);
  float *var_corrs = reinterpret_cast<float *>(io_addrs_[C4NUM]);
  float *output = reinterpret_cast<float *>(io_addrs_[C5NUM]);

  // optimize in the pass.
  int element_cnt = std::accumulate(input_shapes_.begin(), input_shapes_.end(), 1, std::multiplies<int64_t>());
  if (quant_param_size_ == 1) {
    auto dequant = [&](size_t start, size_t end) {
      for (size_t pos = start; pos < end; pos++) {
        // formula: dequant = (x - zp) * scale
        output[pos] = (input[pos] - zps[0]) * scales[0] * var_corrs[0] + mean_corrs[0];
      }
    };
    const int64_t per_unit_size = element_cnt / std::thread::hardware_concurrency();
    ParallelFor(element_cnt, per_unit_size, dequant);
  } else {
    auto bucket_count = input_shapes_[axis_];
    size_t stride = 1;
    for (size_t i = axis_ + 1; i < input_shapes_.size(); i++) {
      stride *= input_shapes_[i];
    }
    auto dequant = [&](size_t start, size_t end) {
      for (size_t pos = start; pos < end; pos++) {
        size_t bucket_index = (pos / stride) % bucket_count;
        // formula: dequant = (x - zp) * scale
        output[pos] =
          (input[pos] - zps[bucket_index]) * scales[bucket_index] * var_corrs[bucket_index] + mean_corrs[bucket_index];
      }
    };
    const int64_t per_unit_size = element_cnt / std::thread::hardware_concurrency();
    ParallelFor(element_cnt, per_unit_size, dequant);
  }
}

void QuantDTypeCastKernel::FixedBitHalfDequantTask() {
  int8_t *input = reinterpret_cast<int8_t *>(io_addrs_[C0NUM]);
  float *scales = reinterpret_cast<float *>(io_addrs_[C1NUM]);
  int *zps = reinterpret_cast<int *>(io_addrs_[C2NUM]);
  float *mean_corrs = reinterpret_cast<float *>(io_addrs_[C3NUM]);
  float *var_corrs = reinterpret_cast<float *>(io_addrs_[C4NUM]);
  Eigen::half *output = reinterpret_cast<Eigen::half *>(io_addrs_[C5NUM]);

  // optimize in the pass.
  int element_cnt = std::accumulate(input_shapes_.begin(), input_shapes_.end(), 1, std::multiplies<int64_t>());
  if (quant_param_size_ == 1) {
    auto dequant = [&](size_t start, size_t end) {
      for (size_t pos = start; pos < end; pos++) {
        // formula: dequant = (x - zp) * scale
        output[pos] = Eigen::half((input[pos] - zps[0]) * scales[0] * var_corrs[0] + mean_corrs[0]);
      }
    };
    const int64_t per_unit_size = element_cnt / std::thread::hardware_concurrency();
    ParallelFor(element_cnt, per_unit_size, dequant);
  } else {
    auto bucket_count = input_shapes_[axis_];
    size_t stride = 1;
    for (size_t i = axis_ + 1; i < input_shapes_.size(); i++) {
      stride *= input_shapes_[i];
    }
    auto dequant = [&](size_t start, size_t end) {
      for (size_t pos = start; pos < end; pos++) {
        size_t bucket_index = (pos / stride) % bucket_count;
        // formula: dequant = (x - zp) * scale
        output[pos] = Eigen::half((input[pos] - zps[bucket_index]) * scales[bucket_index] * var_corrs[bucket_index] +
                                  mean_corrs[bucket_index]);
      }
    };
    const int64_t per_unit_size = element_cnt / std::thread::hardware_concurrency();
    ParallelFor(element_cnt, per_unit_size, dequant);
  }
}

uint32_t QuantDTypeCastKernel::QuantDTypeCastTask() {
  if (io_addrs_.empty() || io_addrs_.size() != C6NUM) {
    return kAicpuKernelStateFailed;
  }
  if (dst_type_ == mindspore::kNumberTypeFloat32) {
    FixedBitFloatDequantTask();
  } else if (dst_type_ == mindspore::kNumberTypeFloat16) {
    FixedBitHalfDequantTask();
  } else {
    return kAicpuKernelStateInvalid;
  }
  return kAicpuKernelStateSucess;
}

uint32_t QuantDTypeCastKernel::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  // get value of attr axis
  axis_ = attrs["axis"].i();
  dst_type_ = attrs["dst_t"].i();
  src_type_ = attrs["src_t"].i();

  // get input tensors shape
  if (node_def_.inputs_size() != C5NUM) {
    AICPU_LOGE("For 'QuantDTypeCast', input tensor number must be 1, but got %d", node_def_.inputs_size());
    return kAicpuKernelStateInvalid;
  }
  aicpuops::Tensor input_tensor = node_def_.inputs(0);
  input_shapes_ = GetShape(input_tensor.tensor_shape());
  quant_param_size_ = node_def_.inputs(1).tensor_shape().dim(0).size();
  // get output tensor shape
  if (node_def_.outputs_size() != 1) {
    AICPU_LOGE("For 'QuantDTypeCast', output tensor number must be 1, but got %d", node_def_.outputs_size());
    return kAicpuKernelStateInvalid;
  }

  return kAicpuKernelStateSucess;
}

uint32_t QuantDTypeCastKernel::DoCompute() { return QuantDTypeCastTask(); }
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t QuantDTypeCast(void *param) {
  aicpu::QuantDTypeCastKernel quant_dtype_cast_kernel;
  return quant_dtype_cast_kernel.Compute(param);
}
}
