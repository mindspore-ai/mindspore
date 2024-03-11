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
#include "cpu_kernel/ms_kernel/data_format_vec_permute.h"

#include <string>

#include "context/inc/cpu_kernel_utils.h"
#include "cpu_types.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kDataFormatVecPermute = "DataFormatVecPermute";

#define DATAFORMATVECPERMUTE_COMPUTE_CASE(DTYPE, TYPE, DIM, SRC_FORMAT_STR, DST_FORMAT_STR, X, Y, CTX)   \
  case (DTYPE): {                                                                                        \
    uint32_t result = DataFormatVecPermuteCompute<TYPE>(DIM, SRC_FORMAT_STR, DST_FORMAT_STR, X, Y, CTX); \
    if (result != KERNEL_STATUS_OK) {                                                                    \
      CUST_KERNEL_LOG_ERROR(ctx, "DataFormatVecPermute kernel compute failed.");                         \
      return result;                                                                                     \
    }                                                                                                    \
    break;                                                                                               \
  }

}  // namespace

namespace aicpu {
uint32_t DataFormatVecPermute::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "Check DataFormatVecPermute params failed.");
  AttrValue *src_format = ctx.GetAttr("src_format");
  std::string src_format_str = src_format->GetString();
  CUST_KERNEL_CHECK_FALSE(ctx, (src_format_str.size() == 4), KERNEL_STATUS_PARAM_INVALID,
                          "src_format must be of length 4, but the length of src_format = [%d].",
                          src_format_str.size());
  AttrValue *dst_format = ctx.GetAttr("dst_format");
  std::string dst_format_str = dst_format->GetString();
  CUST_KERNEL_CHECK_FALSE(ctx, (dst_format_str.size() == 4), KERNEL_STATUS_PARAM_INVALID,
                          "dst_format must be of length 4, but the length of dst_format = [%d].",
                          dst_format_str.size());
  Tensor *x = ctx.Input(0);
  auto x_shape = x->GetTensorShape();
  int32_t dim = x_shape->GetDims();
  CUST_KERNEL_CHECK_FALSE(ctx, (dim == 1 || dim == 2), KERNEL_STATUS_PARAM_INVALID,
                          "Input dimension must be 1 or 2, but got dimension = [%d].", dim);
  Tensor *y = ctx.Output(0);
  auto y_shape = y->GetTensorShape();
  if (dim == 1) {
    CUST_KERNEL_CHECK_FALSE(ctx, (x_shape->GetDimSize(0) == 4), KERNEL_STATUS_PARAM_INVALID,
                            "1D Input must be of size 4, but got size %lld.", x_shape->GetDimSize(0));
    CUST_KERNEL_CHECK_FALSE(ctx, (y_shape->GetDimSize(0) == 4), KERNEL_STATUS_PARAM_INVALID,
                            "1D Output must be of size 4, but got size %lld.", y_shape->GetDimSize(0));
  } else if (dim == 2) {
    CUST_KERNEL_CHECK_FALSE(ctx, (x_shape->GetDimSize(0) == 4), KERNEL_STATUS_PARAM_INVALID,
                            "First dimension of 2D Input must be of size 4, but got size %lld.",
                            x_shape->GetDimSize(0));
    CUST_KERNEL_CHECK_FALSE(ctx, (x_shape->GetDimSize(1) == 2), KERNEL_STATUS_PARAM_INVALID,
                            "Second dimension of 2D Input must be of size 2, but got size %lld.",
                            x_shape->GetDimSize(1));
    CUST_KERNEL_CHECK_FALSE(ctx, (y_shape->GetDimSize(0) == 4), KERNEL_STATUS_PARAM_INVALID,
                            "First dimension of 2D Output must be of size 4, but got size %lld.",
                            y_shape->GetDimSize(0));
    CUST_KERNEL_CHECK_FALSE(ctx, (y_shape->GetDimSize(1) == 2), KERNEL_STATUS_PARAM_INVALID,
                            "Second dimension of 2D Output must be of size 2, but got size %lld.",
                            y_shape->GetDimSize(1));
  }

  auto x_type = x->GetDataType();
  auto y_type = y->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (x_type == y_type), KERNEL_STATUS_PARAM_INVALID,
                          "Input[%s] and output[%s] must have the same DataType.", DTypeStr(x_type).c_str(),
                          DTypeStr(y_type).c_str());
  switch (x_type) {
    DATAFORMATVECPERMUTE_COMPUTE_CASE(DT_INT32, int32_t, dim, src_format_str, dst_format_str, x, y, ctx)
    DATAFORMATVECPERMUTE_COMPUTE_CASE(DT_INT64, int64_t, dim, src_format_str, dst_format_str, x, y, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] Data type of input is not support, input data type is [%s].",
                            ctx.GetOpType().c_str(), DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DataFormatVecPermute::DataFormatVecPermuteCompute(const int32_t dim, const std::string &src_format_str,
                                                           const std::string &dst_format_str, Tensor *x, Tensor *y,
                                                           CpuKernelContext &ctx) {
  T *x_addrs = reinterpret_cast<T *>(x->GetData());
  T *y_addrs = reinterpret_cast<T *>(y->GetData());

  if (dim == 1) {
    for (uint64_t i = 0; i < dst_format_str.size(); i++) {
      for (uint64_t j = 0; j < src_format_str.size(); j++) {
        if (dst_format_str[i] == src_format_str[j]) {
          y_addrs[i] = x_addrs[j];
          break;
        }
      }
    }
  } else if (dim == 2) {
    for (uint64_t i = 0; i < dst_format_str.size(); i++) {
      for (uint64_t j = 0; j < src_format_str.size(); j++) {
        if (dst_format_str[i] == src_format_str[j]) {
          y_addrs[i * 2] = x_addrs[j * 2];
          y_addrs[i * 2 + 1] = x_addrs[j * 2 + 1];
          break;
        }
      }
    }
  }

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kDataFormatVecPermute, DataFormatVecPermute);
}  // namespace aicpu
