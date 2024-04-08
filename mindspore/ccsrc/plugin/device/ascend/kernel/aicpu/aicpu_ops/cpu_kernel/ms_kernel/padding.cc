/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#include "padding.h"
#include <Eigen/Core>
#include <securec.h>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *const kPadding = "Padding";
}  // namespace

namespace aicpu {
namespace {
template <typename T>
KernelStatus PaddingTask(CpuKernelContext &ctx, int64_t &pad_dim_size, int64_t &val_size, int64_t &out_size,
                         size_t &type_size) {
  T *input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  T *output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto ret = memset_s(output, out_size * type_size, 0x00, out_size * type_size);
  CUST_KERNEL_CHECK_FALSE(ctx, ret == EOK, KERNEL_STATUS_INNER_ERROR, "memset failed.");
  auto shardCopy = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      output[i * pad_dim_size] = input[i];
    }
  };
  (void)CpuKernelUtils::ParallelFor(ctx, val_size, val_size / 12, shardCopy);
  return KERNEL_STATUS_OK;
}
}  // namespace

uint32_t PaddingKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, ParseKernelParam(ctx), "%s check failed", ctx.GetOpType().c_str());
  std::map<DataType, std::function<uint32_t(CpuKernelContext &, int64_t &, int64_t &, int64_t &, size_t &)>> calls;

  calls[DT_INT8] = PaddingTask<int8_t>;
  calls[DT_INT16] = PaddingTask<int16_t>;
  calls[DT_INT32] = PaddingTask<int32_t>;
  calls[DT_INT64] = PaddingTask<int64_t>;
  calls[DT_FLOAT16] = PaddingTask<Eigen::half>;
  calls[DT_FLOAT] = PaddingTask<float>;
  calls[DT_DOUBLE] = PaddingTask<double>;
  calls[DT_UINT8] = PaddingTask<uint8_t>;
  calls[DT_UINT16] = PaddingTask<uint16_t>;
  calls[DT_UINT32] = PaddingTask<uint32_t>;
  calls[DT_UINT64] = PaddingTask<uint64_t>;
  calls[DT_BOOL] = PaddingTask<bool>;

  size_t type_size = GetSizeByDataType(x_type_);
  return calls[x_type_](ctx, pad_dim_size_, val_size_, out_size_, type_size);
}

uint32_t PaddingKernel::ParseKernelParam(CpuKernelContext &ctx) {
  auto pad_dim_size_attr = ctx.GetAttr("pad_dim_size");
  CUST_KERNEL_CHECK_NULLPTR(ctx, pad_dim_size_attr, KERNEL_STATUS_INNER_ERROR, "Failed to get attr 'pad_dim_size'.");
  pad_dim_size_ = pad_dim_size_attr->GetInt();
  auto x = ctx.Input(0);
  auto x_shape = x->GetTensorShape()->GetDimSizes();

  x_type_ = x->GetDataType();

  if (pad_dim_size_ < 1) {
    CUST_AICPU_LOGE(ctx, "attr pad_dim_size must be positive, got value %d", pad_dim_size_);
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (x_shape.size() < 2) {
    CUST_AICPU_LOGE(ctx, "The rank of the x tensor should be at least 2, got value %d", x_shape.size());
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (x_shape.back() != 1) {
    CUST_AICPU_LOGE(ctx, "The last dim of x tensor must be 1, got value %d.", x_shape.back());
    return KERNEL_STATUS_INNER_ERROR;
  }

  for (size_t i = 0; i < x_shape.size(); ++i) {
    val_size_ *= x_shape[i];
  }

  out_size_ *= val_size_ * pad_dim_size_;

  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kPadding, PaddingKernel);
}  // namespace aicpu
