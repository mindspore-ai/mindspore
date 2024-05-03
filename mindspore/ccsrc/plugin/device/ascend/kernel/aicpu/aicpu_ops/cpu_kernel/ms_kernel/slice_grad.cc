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
#include "slice_grad.h"
#include <functional>
#include "securec.h"
#include "base/bfloat16.h"
#include "context/inc/cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "context/common/status.h"
#include "utils/kernel_util.h"

namespace aicpu {
namespace {
const char *kSliceGrad = "SliceGrad";
constexpr size_t kInputNum = 4;
constexpr size_t kOutputNum = 1;
constexpr size_t kDim2 = 2;
constexpr size_t kDim3 = 3;
}  // namespace

bool SliceGradKernel::CheckParams(CpuKernelContext &ctx) const {
  // check dy shape, N-D
  auto n = dy_shape_.size();
  if (n == 0) {
    CUST_AICPU_LOGE(ctx, "For 'SliceGrad', 'dy' shape can not be empty.");
    return false;
  }

  // check begin shape, 1-D with shape [N]
  if (begin_shape_.size() != 1) {
    CUST_AICPU_LOGE(ctx, "For 'SliceGrad', 'begin' shape rank must be 1, but got %lu", begin_shape_.size());
    return false;
  }
  if (LongToSize(ctx, begin_shape_[0]) != n) {
    CUST_AICPU_LOGE(ctx, "For 'SliceGrad', 'begin' shape must be [%lu], but got [%ld]", n, begin_shape_[0]);
    return false;
  }

  // check size shape, 1-D with shape [N]
  if (size_shape_.size() != 1) {
    CUST_AICPU_LOGE(ctx, "For 'SliceGrad', 'size' shape rank must be 1, but got %lu", size_shape_.size());
    return false;
  }
  if (LongToSize(ctx, size_shape_[0]) != n) {
    CUST_AICPU_LOGE(ctx, "For 'SliceGrad', 'size' shape must be [%lu], but got [%ld]", n, size_shape_[0]);
    return false;
  }

  // check output shape, N-D
  if (output_shape_.size() != n) {
    CUST_AICPU_LOGE(ctx, "For 'SliceGrad', 'dy' shape and output tensor shape must have same rank, but got %lu vs %lu",
                    n, output_shape_.size());
    return false;
  }

  for (size_t i = 0; i < n; ++i) {
    if (dy_shape_[i] <= 0 || dy_shape_[i] > output_shape_[i]) {
      CUST_AICPU_LOGE(
        ctx, "For 'SliceGrad', it is required that 0 < 'dy' shape[%lu] <= output tensor shape[%lu], but got %ld vs %ld",
        i, i, dy_shape_[i], output_shape_[i]);
      return false;
    }
  }
  return true;
}

bool SliceGradKernel::CheckBeginSizeValue(CpuKernelContext &ctx) {
  for (size_t i = 0; i < begin_value_.size(); ++i) {
    if (begin_value_[i] < 0 || begin_value_[i] >= output_shape_[i]) {
      CUST_AICPU_LOGE(ctx, "For 'SliceGrad', 'begin' [%lu] must be in range [0, %ld), but got %ld", i, output_shape_[i],
                      begin_value_[i]);
      return false;
    }
    if (size_value_[i] < 0) {
      size_value_[i] = output_shape_[i] - begin_value_[i];
    }
    if (size_value_[i] != dy_shape_[i]) {
      CUST_AICPU_LOGE(ctx, "For 'SliceGrad', 'size' [%lu] must be equal to 'dy' shape[%lu], but got %ld vs %ld", i, i,
                      size_value_[i], dy_shape_[i]);
      return false;
    }
    if (begin_value_[i] + size_value_[i] > output_shape_[i]) {
      CUST_AICPU_LOGE(
        ctx, "For 'SliceGrad', 'begin' [%lu] + 'size' [%lu] must be <= output tensor shape[%lu], but got %ld, %ld, %ld",
        i, i, i, begin_value_[i], size_value_[i], output_shape_[i]);
      return false;
    }
  }
  return true;
}

template <typename T, typename S>
uint32_t SliceGradKernel::SliceGradTask(CpuKernelContext &ctx) {
  if (!CheckParams(ctx)) {
    return KERNEL_STATUS_INNER_ERROR;
  }

  S *dy_addr = reinterpret_cast<S *>(ctx.Input(0)->GetData());
  T *begin_addr = reinterpret_cast<T *>(ctx.Input(kDim2)->GetData());
  T *size_addr = reinterpret_cast<T *>(ctx.Input(kDim3)->GetData());
  S *out_addr = reinterpret_cast<S *>(ctx.Output(0)->GetData());

  for (size_t i = 0; i < dy_shape_.size(); ++i) {
    begin_value_.push_back(static_cast<int64_t>(begin_addr[i]));
    size_value_.push_back(static_cast<int64_t>(size_addr[i]));
  }
  if (!CheckBeginSizeValue(ctx)) {
    return KERNEL_STATUS_INNER_ERROR;
  }

  // Calc process
  // 1. fill output address with 0
  int64_t output_num = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<int64_t>());
  size_t output_byte = LongToSize(ctx, output_num) * sizeof(S);
  if (memset_s(out_addr, output_byte, 0, output_byte) != EOK) {
    CUST_AICPU_LOGE(ctx, "For 'SliceGrad', memset_s on output tensor address failed!");
    return KERNEL_STATUS_INNER_ERROR;
  }

  // 2. copy dy_addr to out_addr
  // case: 1D
  if (dy_shape_.size() == 1) {
    size_t block_byte = LongToSize(ctx, dy_shape_[0]) * sizeof(S);
    S *out_start_addr = out_addr + begin_value_[0];
    if (memcpy_s(out_start_addr, block_byte, dy_addr, block_byte) != EOK) {
      CUST_AICPU_LOGE(ctx, "For 'SliceGrad', memcpy_s failed!");
      return KERNEL_STATUS_INNER_ERROR;
    }
    return KERNEL_STATUS_OK;
  }
  // case: > 1D (0D already checked inside CheckParams), the last dim will be scheduled as a block
  std::vector<size_t> dy_block_shape;
  (void)std::transform(dy_shape_.begin(), dy_shape_.end() - 1, std::back_inserter(dy_block_shape),
                       [&ctx](int64_t x) { return LongToSize(ctx, x); });

  std::vector<size_t> out_block_shape_acc{1};
  size_t acc = 1;
  for (size_t i = output_shape_.size() - kDim2; i > 0; --i) {
    acc *= LongToSize(ctx, output_shape_[i]);
    (void)out_block_shape_acc.insert(out_block_shape_acc.begin(), acc);
  }

  auto block_task = [this, &dy_addr, &out_addr, &dy_block_shape, &out_block_shape_acc, &ctx](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      size_t k = 0;
      auto a = i;
      for (size_t j = 0; j < dy_block_shape.size(); ++j) {
        size_t m = dy_block_shape.size() - 1 - j;
        auto idx = a % dy_block_shape[m] + LongToSize(ctx, begin_value_[m]);
        a /= dy_block_shape[m];
        k += idx * out_block_shape_acc[m];
      }
      auto block_sz = LongToSize(ctx, dy_shape_.back());
      size_t block_byte = block_sz * sizeof(S);
      S *dy_start_addr = dy_addr + i * block_sz;
      S *out_start_addr = out_addr + k * LongToSize(ctx, output_shape_.back()) + begin_value_.back();
      if (memcpy_s(out_start_addr, block_byte, dy_start_addr, block_byte) != EOK) {
        CUST_AICPU_LOGE(ctx, "For 'SliceGrad', memcpy_s failed! Current block index is %lu", i);
        return;
      }
    }
  };

  int64_t block_num = 1;
  for (size_t i = 0; i < dy_shape_.size() - 1; ++i) {
    block_num *= dy_shape_[i];
  }
  const int64_t per_unit_size = block_num / static_cast<int64_t>(std::thread::hardware_concurrency());
  CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, block_num, per_unit_size, block_task),
                           "SliceGrad Compute failed.");

  return KERNEL_STATUS_OK;
}

uint32_t SliceGradKernel::ParseKernelParam(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "SliceGrad check input and output number failed.");
  // dy
  auto dy_tensor = ctx.Input(0);
  dy_shape_ = dy_tensor->GetTensorShape()->GetDimSizes();
  dy_type_ = dy_tensor->GetDataType();

  // begin
  auto begin_tensor = ctx.Input(SizeToInt(ctx, kDim2));
  begin_shape_ = begin_tensor->GetTensorShape()->GetDimSizes();
  begin_type_ = begin_tensor->GetDataType();

  // size
  auto size_tensor = ctx.Input(SizeToInt(ctx, kDim3));
  size_shape_ = size_tensor->GetTensorShape()->GetDimSizes();
  auto size_type = size_tensor->GetDataType();
  if (size_type != begin_type_) {
    CUST_AICPU_LOGE(ctx, "For 'SliceGrad', 'begin' and 'size' must have same data type.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto output_tensor = ctx.Output(0);
  output_shape_ = output_tensor->GetTensorShape()->GetDimSizes();
  return KERNEL_STATUS_OK;
}

using namespace std::placeholders;
uint32_t SliceGradKernel::Compute(CpuKernelContext &ctx) {
  ParseKernelParam(ctx);
  std::unordered_map<aicpu::DataType, std::unordered_map<aicpu::DataType, std::function<uint32_t(CpuKernelContext &)>>>
    func_list;
  // begin type int32
  func_list[DT_INT32][DT_FLOAT16] = std::bind(&SliceGradKernel::SliceGradTask<int32_t, Eigen::half>, this, _1);
  func_list[DT_INT32][DT_FLOAT] = std::bind(&SliceGradKernel::SliceGradTask<int32_t, float>, this, _1);
  func_list[DT_INT32][DT_DOUBLE] = std::bind(&SliceGradKernel::SliceGradTask<int32_t, double>, this, _1);
  func_list[DT_INT32][DT_UINT8] = std::bind(&SliceGradKernel::SliceGradTask<int32_t, uint8_t>, this, _1);
  func_list[DT_INT32][DT_UINT16] = std::bind(&SliceGradKernel::SliceGradTask<int32_t, uint16_t>, this, _1);
  func_list[DT_INT32][DT_UINT32] = std::bind(&SliceGradKernel::SliceGradTask<int32_t, uint32_t>, this, _1);
  func_list[DT_INT32][DT_UINT64] = std::bind(&SliceGradKernel::SliceGradTask<int32_t, uint64_t>, this, _1);
  func_list[DT_INT32][DT_INT8] = std::bind(&SliceGradKernel::SliceGradTask<int32_t, int8_t>, this, _1);
  func_list[DT_INT32][DT_INT16] = std::bind(&SliceGradKernel::SliceGradTask<int32_t, int16_t>, this, _1);
  func_list[DT_INT32][DT_INT32] = std::bind(&SliceGradKernel::SliceGradTask<int32_t, int32_t>, this, _1);
  func_list[DT_INT32][DT_INT64] = std::bind(&SliceGradKernel::SliceGradTask<int32_t, int64_t>, this, _1);
  func_list[DT_INT32][DT_BFLOAT16] = std::bind(&SliceGradKernel::SliceGradTask<int32_t, bfloat16>, this, _1);
  // begin type int64
  func_list[DT_INT64][DT_FLOAT16] = std::bind(&SliceGradKernel::SliceGradTask<int64_t, Eigen::half>, this, _1);
  func_list[DT_INT64][DT_FLOAT] = std::bind(&SliceGradKernel::SliceGradTask<int64_t, float>, this, _1);
  func_list[DT_INT64][DT_DOUBLE] = std::bind(&SliceGradKernel::SliceGradTask<int64_t, double>, this, _1);
  func_list[DT_INT64][DT_UINT8] = std::bind(&SliceGradKernel::SliceGradTask<int64_t, uint8_t>, this, _1);
  func_list[DT_INT64][DT_UINT16] = std::bind(&SliceGradKernel::SliceGradTask<int64_t, uint16_t>, this, _1);
  func_list[DT_INT64][DT_UINT32] = std::bind(&SliceGradKernel::SliceGradTask<int64_t, uint32_t>, this, _1);
  func_list[DT_INT64][DT_UINT64] = std::bind(&SliceGradKernel::SliceGradTask<int64_t, uint64_t>, this, _1);
  func_list[DT_INT64][DT_INT8] = std::bind(&SliceGradKernel::SliceGradTask<int64_t, int8_t>, this, _1);
  func_list[DT_INT64][DT_INT16] = std::bind(&SliceGradKernel::SliceGradTask<int64_t, int16_t>, this, _1);
  func_list[DT_INT64][DT_INT32] = std::bind(&SliceGradKernel::SliceGradTask<int64_t, int32_t>, this, _1);
  func_list[DT_INT64][DT_INT64] = std::bind(&SliceGradKernel::SliceGradTask<int64_t, int64_t>, this, _1);
  func_list[DT_INT64][DT_BFLOAT16] = std::bind(&SliceGradKernel::SliceGradTask<int64_t, bfloat16>, this, _1);

  if (func_list.find(begin_type_) == func_list.end()) {
    CUST_AICPU_LOGE(ctx, "'SliceGrad' does not support current 'begin' type.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (func_list[begin_type_].find(dy_type_) == func_list[begin_type_].end()) {
    CUST_AICPU_LOGE(ctx, "'SliceGrad' does not support current 'dy' type.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return func_list[begin_type_][dy_type_](ctx);
}
REGISTER_MS_CPU_KERNEL(kSliceGrad, SliceGradKernel);
}  // namespace aicpu