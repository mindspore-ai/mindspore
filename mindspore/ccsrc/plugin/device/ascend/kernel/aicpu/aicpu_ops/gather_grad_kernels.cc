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
#include "./gather_grad_kernels.h"
#include <Eigen/Dense>
#include <map>
#include <thread>
#include <numeric>
#include <vector>
#include <functional>
#include "common/atomic_op.h"
#include "aicpu_sharder/aicpu_sharder.h"
#include "proto/aicpu_tensor.pb.h"

namespace aicpu {
namespace {
constexpr auto kDim = "dim";
constexpr auto kAddressSize = 4;
constexpr auto kDim0 = 0;
constexpr auto kDim1 = 1;
constexpr auto kDim2 = 2;
constexpr auto kDim3 = 3;

template <typename T, typename S>
static uint32_t GatherGrad(const T *index, const S *grad, S *output, int64_t dim_before_axis, int64_t dim_at_axis_index,
                           int64_t dim_at_axis_output, int64_t dim_after_axis) {
  if (dim_after_axis == 0) {
    AICPU_LOGE("dim_after_axis cannot be 0.");
    return kAicpuKernelStateFailed;
  }
  int64_t number = dim_before_axis * dim_at_axis_index * dim_after_axis;
  bool status = false;
  auto shard_gather_grad = [&](size_t start, size_t end) {
    int64_t dim_input = dim_at_axis_index * dim_after_axis;
    int64_t dim_output = dim_at_axis_output * dim_after_axis;
    for (size_t id = start; id < end; ++id) {
      T j_read = index[id];
      auto max_index = static_cast<T>(dim_at_axis_output);
      if (j_read >= max_index || j_read < -max_index) {
        AICPU_LOGE("The value of 'dim' should be in [%d %d), but got %d", -max_index, max_index, j_read);
        AtomicAdd<bool>(&status, true);
        return;
      }
      if (j_read < 0) {
        j_read += max_index;
      }

      int64_t signed_id = SizeToInt(id);
      int64_t signed_j_read = static_cast<int64_t>(j_read);
      int64_t read_id =
        signed_id / dim_input * dim_output + signed_j_read * dim_after_axis + signed_id % dim_after_axis;
      AtomicAdd<S>(output + read_id, grad[id]);
    }
  };

  const int64_t per_unit_size = number / std::thread::hardware_concurrency();
  ParallelFor(number, per_unit_size, shard_gather_grad);
  if (status) {
    return kAicpuKernelStateFailed;
  }
  return kAicpuKernelStateSucess;
}
}  // namespace

template <typename T, typename S>
uint32_t GatherDGradV2Kernel::GatherDGradV2Task() {
  if (io_addrs_.size() != kAddressSize) {
    AICPU_LOGE("GatherDGradV2Kernel's address is invalid");
    return kAicpuKernelStateFailed;
  }
  T *index = reinterpret_cast<T *>(io_addrs_[kDim1]);
  S *grad = reinterpret_cast<S *>(io_addrs_[kDim2]);
  S *output = reinterpret_cast<S *>(io_addrs_[kDim3]);

  int64_t output_rank = static_cast<int64_t>(output_shape_.size());
  if (dim_ >= output_rank || dim_ < -output_rank) {
    AICPU_LOGE("The value of 'dim' should be in [%d %d), but got %d", -output_rank, output_rank, dim_);
    return kAicpuKernelStateFailed;
  }
  if (dim_ < 0) {
    dim_ = dim_ + output_rank;
  }
  int64_t grad_rank = static_cast<int64_t>(grad_shape_.size());
  if (dim_ >= grad_rank) {
    AICPU_LOGE("The value of 'dim' should be in [%d %d), but got %d", -grad_rank, grad_rank, dim_);
    return kAicpuKernelStateFailed;
  }

  int64_t dim_before_axis =
    std::accumulate(output_shape_.begin(), output_shape_.begin() + dim_, 1, std::multiplies<int64_t>());
  int64_t dim_at_axis_grad = grad_shape_[LongToSize(dim_)];
  int64_t dim_at_axis_output = output_shape_[LongToSize(dim_)];
  int64_t dim_after_axis =
    std::accumulate(output_shape_.begin() + dim_ + 1, output_shape_.end(), 1, std::multiplies<int64_t>());
  int64_t output_size = dim_before_axis * dim_at_axis_output * dim_after_axis * sizeof(S);
  if (memset_s(output, output_size, 0x0, output_size) != EOK) {
    AICPU_LOGE("memset_s failed!");
    return kAicpuKernelStateFailed;
  }
  return GatherGrad(index, grad, output, dim_before_axis, dim_at_axis_grad, dim_at_axis_output, dim_after_axis);
}

uint32_t GatherDGradV2Kernel::ParseKernelParam() {
  // ori input
  aicpuops::Tensor input_tensor = node_def_.inputs(kDim0);
  const auto &input_shape = input_tensor.tensor_shape();
  for (int i = 0; i < input_shape.dim_size(); ++i) {
    (void)input_shape_.emplace_back(input_shape.dim(i).size());
  }

  // index input
  input_tensor = node_def_.inputs(kDim1);
  index_type_ = static_cast<aicpuops::DataType>(input_tensor.tensor_type());
  const auto &index_shape = input_tensor.tensor_shape();
  for (int i = 0; i < index_shape.dim_size(); ++i) {
    (void)index_shape_.emplace_back(index_shape.dim(i).size());
  }

  // grad input
  input_tensor = node_def_.inputs(kDim2);
  grad_type_ = static_cast<aicpuops::DataType>(input_tensor.tensor_type());
  const auto &grad_shape = input_tensor.tensor_shape();
  for (int i = 0; i < grad_shape.dim_size(); ++i) {
    (void)grad_shape_.emplace_back(grad_shape.dim(i).size());
  }
  if (index_shape_ != grad_shape_) {
    AICPU_LOGE("the shape of index and grad should be same!");
    return kAicpuKernelStateInvalid;
  }

  // output
  aicpuops::Tensor output_tensor = node_def_.outputs(kDim0);
  const auto &output_shape = output_tensor.tensor_shape();
  for (int i = 0; i < output_shape.dim_size(); ++i) {
    (void)output_shape_.emplace_back(output_shape.dim(i).size());
  }
  if (output_shape_ != input_shape_) {
    AICPU_LOGE("the shape of input and output should be same!");
    return kAicpuKernelStateInvalid;
  }

  auto node_def_attrs = node_def_.attrs();
  dim_ = node_def_attrs[kDim].i();

  return kAicpuKernelStateSucess;
}

uint32_t GatherDGradV2Kernel::DoCompute() {
  std::map<aicpuops::DataType, std::map<aicpuops::DataType, std::function<uint32_t()>>> calls;
  // index int32
  calls[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_INT8] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, int8_t>, this);
  calls[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_INT16] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, int16_t>, this);
  calls[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_INT32] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, int32_t>, this);
  calls[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_INT64] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, int64_t>, this);
  calls[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_FLOAT16] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, Eigen::half>, this);
  calls[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_FLOAT32] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, float>, this);
  calls[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_FLOAT64] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, double>, this);
  calls[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_UINT8] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, uint8_t>, this);
  calls[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_UINT16] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, uint16_t>, this);
  calls[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_UINT32] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, uint32_t>, this);
  calls[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_UINT64] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, uint64_t>, this);
  calls[aicpuops::DataType::MS_INT32][aicpuops::DataType::MS_BOOL] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, bool>, this);
  // index int64
  calls[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_INT8] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, int8_t>, this);
  calls[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_INT16] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, int16_t>, this);
  calls[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_INT32] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, int32_t>, this);
  calls[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_INT64] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, int64_t>, this);
  calls[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_FLOAT16] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, Eigen::half>, this);
  calls[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_FLOAT32] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, float>, this);
  calls[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_FLOAT64] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, double>, this);
  calls[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_UINT8] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, uint8_t>, this);
  calls[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_UINT16] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, uint16_t>, this);
  calls[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_UINT32] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, uint32_t>, this);
  calls[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_UINT64] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, uint64_t>, this);
  calls[aicpuops::DataType::MS_INT64][aicpuops::DataType::MS_BOOL] =
    std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, bool>, this);

  if (calls.find(index_type_) == calls.end()) {
    AICPU_LOGE("GatherDGradV2 op don't support index tensor types: %s", typeid(index_type_).name());
    return kAicpuKernelStateFailed;
  }
  return calls[index_type_][grad_type_]();
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t GatherDGradV2(void *param) {
  aicpu::GatherDGradV2Kernel gatherd_grad_v2_kernel;
  return gatherd_grad_v2_kernel.Compute(param);
}
}
