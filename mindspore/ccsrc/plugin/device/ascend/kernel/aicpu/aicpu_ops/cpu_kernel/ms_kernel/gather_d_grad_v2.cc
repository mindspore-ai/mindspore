/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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
#include "context/inc/cpu_kernel_utils.h"
#include "Eigen/Core"
#include "base/bfloat16.h"
#include "utils/atomic_op.h"
#include "mindspore/core/utils/ms_utils_secure.h"

namespace aicpu {
namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
const char *kGatherDGradV2 = "GatherDGradV2";
constexpr auto kDim = "dim";
constexpr auto kDim0 = 0;
constexpr auto kDim1 = 1;
constexpr auto kDim2 = 2;

template <typename T, typename S>
static uint32_t GatherGrad(const T *index, const S *grad, S *output, int64_t dim,
                           const std::vector<int64_t> &index_shape, const std::vector<int64_t> &output_shape,
                           CpuKernelContext &ctx) {
  int64_t number = std::accumulate(index_shape.begin(), index_shape.end(), 1, std::multiplies<int64_t>());
  bool status = false;
  auto rank = index_shape.size();
  auto dim_size = static_cast<size_t>(dim);
  auto shard_gather_grad = [&](size_t start, size_t end) {
    for (size_t id = start; id < end; ++id) {
      T j_read = index[id];
      auto max_index = static_cast<T>(output_shape[dim_size]);
      if (j_read >= max_index || j_read < -max_index) {
        CUST_AICPU_LOGE(ctx, "The value of 'dim' should be in [%d %d), but got %d", -max_index, max_index, j_read);
        AtomicAdd<bool>(ctx, &status, true);
        return;
      }
      if (j_read < 0) {
        j_read += max_index;
      }

      int64_t signed_id = SizeToInt(ctx, id);
      int64_t signed_j_read = static_cast<int64_t>(j_read);
      int64_t accumulate_offset = 1;
      int64_t out_accumulate_offset = 1;
      int64_t offset = 0;
      for (size_t i = rank; i > 0; i--) {
        auto real_i = i - 1;
        auto tmp = real_i == dim_size ? signed_j_read : (signed_id / accumulate_offset) % index_shape[real_i];
        accumulate_offset *= index_shape[real_i];
        offset += tmp * out_accumulate_offset;
        out_accumulate_offset *= output_shape[real_i];
      }

      AtomicAdd<S>(ctx, output + offset, grad[id]);
    }
  };

  const int64_t per_unit_size = number / std::thread::hardware_concurrency();
  CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, number, per_unit_size, shard_gather_grad),
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
    CUST_AICPU_LOGE(ctx, "The value of 'dim' should be in [%d %d), but got %d", -output_rank, output_rank, dim_);
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (dim_ < 0) {
    dim_ = dim_ + output_rank;
  }
  int64_t grad_rank = static_cast<int64_t>(grad_shape_.size());
  if (dim_ >= grad_rank) {
    CUST_AICPU_LOGE(ctx, "The value of 'dim' should be in [%d %d), but got %d", -grad_rank, grad_rank, dim_);
    return KERNEL_STATUS_INNER_ERROR;
  }

  auto output_size =
    static_cast<size_t>(std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<int64_t>())) *
    sizeof(S);
  uint8_t *data = reinterpret_cast<uint8_t *>(output);
  if (data == nullptr) {
    CUST_AICPU_LOGE(ctx, "For '%s', the output is nullptr.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (mindspore::common::huge_memset(data, output_size, 0x0, output_size) != EOK) {
    CUST_AICPU_LOGE(ctx, "For '%s', failed to init output.", kGatherDGradV2);
    return KERNEL_STATUS_INNER_ERROR;
  }

  return GatherGrad(index, grad, output, dim_, index_shape_, output_shape_, ctx);
}

uint32_t GatherDGradV2Kernel::ParseKernelParam(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "GatherDGradV2 check input and output number failed.");
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

  if (index_shape_.empty()) {
    index_shape_ = std::vector<int64_t>({1});
  }
  if (grad_shape_.empty()) {
    grad_shape_ = std::vector<int64_t>({1});
  }

  if (index_shape_ != grad_shape_) {
    CUST_AICPU_LOGE(ctx, "the shape of index and grad should be same!");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // output
  auto output_tensor = ctx.Output(kDim0);
  output_shape_ = output_tensor->GetTensorShape()->GetDimSizes();
  if (output_shape_.empty()) {
    output_shape_ = std::vector<int64_t>({1});
  }
  if (output_shape_ != input_shape_) {
    CUST_AICPU_LOGE(ctx, "the shape of input and output should be same!");
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
  calls[DT_INT32][DT_BFLOAT16] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int32_t, bfloat16>, this, _1);
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
  calls[DT_INT64][DT_BFLOAT16] = std::bind(&GatherDGradV2Kernel::GatherDGradV2Task<int64_t, bfloat16>, this, _1);

  if (calls.find(index_type_) == calls.end()) {
    CUST_AICPU_LOGE(ctx, "GatherDGradV2 op don't support index tensor types: %s", typeid(index_type_).name());
    return KERNEL_STATUS_INNER_ERROR;
  }
  return calls[index_type_][grad_type_](ctx);
}

REGISTER_MS_CPU_KERNEL(kGatherDGradV2, GatherDGradV2Kernel);
}  // namespace aicpu
