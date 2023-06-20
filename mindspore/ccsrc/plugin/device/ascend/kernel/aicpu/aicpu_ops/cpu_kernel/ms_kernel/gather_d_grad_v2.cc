/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "gather_d_grad_v2.h"
#include <algorithm>
#include <tuple>
#include <utility>
#include <functional>

#include "securec.h"
#include "tile.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"
#include "Eigen/Core"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/atomic_op.h"

namespace aicpu {
namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
const char *kGatherDGradV2 = "GatherDGradV2";
constexpr auto kDim = "dim";
constexpr auto kAddressSize = 4;
constexpr auto kDim0 = 0;
constexpr auto kDim1 = 1;
constexpr auto kDim2 = 2;
constexpr auto kDim3 = 3;

template <typename T, typename S>
static uint32_t GatherGrad(const T *index, const S *grad, S *output, int64_t dim_before_axis, int64_t dim_at_axis_index,
                           int64_t dim_at_axis_output, int64_t dim_after_axis, CpuKernelContext &ctx) {
  if (dim_after_axis == 0) {
    AICPU_LOGE("dim_after_axis cannot be 0.");
    return KERNEL_STATUS_INNER_ERROR;
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
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, number, per_unit_size, shard_gather_grad),
                      "GatherDGradV2 compute failed.");
  if (status) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}
}  // namespace

template <typename T, typename S>
uint32_t GatherDGradV2Kernel::GatherDGradV2Task(CpuKernelContext &ctx) {
  T *index = reinterpret_cast<T *>(ctx.Input(kDim1)->GetData());
  S *grad = reinterpret_cast<S *>(ctx.Input(kDim2)->GetData());
  S *output = reinterpret_cast<S *>(ctx.Output(kDim0)->GetData());

  int64_t output_rank = static_cast<int64_t>(output_shape_.size());
  if (dim_ >= output_rank || dim_ < -output_rank) {
    AICPU_LOGE("The value of 'dim' should be in [%d %d), but got %d", -output_rank, output_rank, dim_);
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (dim_ < 0) {
    dim_ = dim_ + output_rank;
  }
  int64_t grad_rank = static_cast<int64_t>(grad_shape_.size());
  if (dim_ >= grad_rank) {
    AICPU_LOGE("The value of 'dim' should be in [%d %d), but got %d", -grad_rank, grad_rank, dim_);
    return KERNEL_STATUS_INNER_ERROR;
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
    return KERNEL_STATUS_INNER_ERROR;
  }
  return GatherGrad(index, grad, output, dim_before_axis, dim_at_axis_grad, dim_at_axis_output, dim_after_axis, ctx);
}

uint32_t GatherDGradV2Kernel::ParseKernelParam(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "GatherDGradV2 check input and output number failed.");
  // ori input
  input_shape_ = ctx.Input(kDim0)->GetTensorShape()->GetDimSizes();

  // index input
  auto input_tensor = ctx.Input(kDim1);
  index_type_ = input_tensor->GetDataType();
  index_shape_ = input_tensor->GetTensorShape()->GetDimSizes();

  // grad input
  auto grad_tensor = ctx.Input(kDim2);
  grad_type_ = grad_tensor->GetDataType();
  grad_shape_ = grad_tensor->GetTensorShape()->GetDimSizes();
  if (index_shape_ != grad_shape_) {
    AICPU_LOGE("the shape of index and grad should be same!");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // output
  auto output_tensor = ctx.Output(kDim0);
  output_shape_ = output_tensor->GetTensorShape()->GetDimSizes();
  if (output_shape_ != input_shape_) {
    AICPU_LOGE("the shape of input and output should be same!");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  dim_ = ctx.GetAttr(kDim)->GetInt();
  return KERNEL_STATUS_OK;
}

using namespace std::placeholders;
uint32_t GatherDGradV2Kernel::Compute(CpuKernelContext &ctx) {
  ParseKernelParam(ctx);
  std::map<DataType, std::map<DataType, std::function<uint32_t(CpuKernelContext &)>>> calls;
  // index int32
  calls[DT_INT32][DT_INT8] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, int8_t>, this, _1);
  calls[DT_INT32][DT_INT16] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, int16_t>, this, _1);
  calls[DT_INT32][DT_INT32] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, int32_t>, this, _1);
  calls[DT_INT32][DT_INT64] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, int64_t>, this, _1);
  calls[DT_INT32][DT_FLOAT16] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, Eigen::half>, this, _1);
  calls[DT_INT32][DT_FLOAT] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, float>, this, _1);
  calls[DT_INT32][DT_DOUBLE] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, double>, this, _1);
  calls[DT_INT32][DT_UINT8] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, uint8_t>, this, _1);
  calls[DT_INT32][DT_UINT16] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, uint16_t>, this, _1);
  calls[DT_INT32][DT_UINT32] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, uint32_t>, this, _1);
  calls[DT_INT32][DT_UINT64] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, uint64_t>, this, _1);
  calls[DT_INT32][DT_BOOL] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, bool>, this, _1);
  // index int64
  calls[DT_INT64][DT_INT8] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, int8_t>, this, _1);
  calls[DT_INT64][DT_INT16] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, int16_t>, this, _1);
  calls[DT_INT64][DT_INT32] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, int32_t>, this, _1);
  calls[DT_INT64][DT_INT64] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, int64_t>, this, _1);
  calls[DT_INT64][DT_FLOAT16] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, Eigen::half>, this, _1);
  calls[DT_INT64][DT_FLOAT] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, float>, this, _1);
  calls[DT_INT64][DT_DOUBLE] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, double>, this, _1);
  calls[DT_INT64][DT_UINT8] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, uint8_t>, this, _1);
  calls[DT_INT64][DT_UINT16] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, uint16_t>, this, _1);
  calls[DT_INT64][DT_UINT32] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, uint32_t>, this, _1);
  calls[DT_INT64][DT_UINT64] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, uint64_t>, this, _1);
  calls[DT_INT64][DT_BOOL] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, bool>, this, _1);

  if (calls.find(index_type_) == calls.end()) {
    AICPU_LOGE("GatherDGradV2 op don't support index tensor types: %s", typeid(index_type_).name());
    return KERNEL_STATUS_INNER_ERROR;
  }
  return calls[index_type_][grad_type_](ctx);
}

REGISTER_CPU_KERNEL(kGatherDGradV2, GatherDGradV2Kernel);
}  // namespace aicpu
