/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "cpu_kernel/ms_kernel/irfft_double.h"
#include "context/inc/cpu_kernel_utils.h"
#include "mindspore/core/mindapi/base/types.h"
#include "utils/kernel_util.h"
#include "cpu_kernel/utils/fft_helper.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const uint32_t kIndex0 = 0;
const uint32_t kNIndex = 1;
const uint32_t kDimIndex = 2;
const char *kIRFFTDouble = "IRFFTDouble";

constexpr float kDoubleFactor = 2.0;
constexpr int kOnsideDivisor = 2;

#define IRFFTDOUBLE_COMPUTE_CASE(DTYPE, TYPE, CTX)                                                           \
  case (DTYPE): {                                                                                            \
    uint32_t result = IRFFTDoubleCompute<TYPE>(CTX);                                                         \
    if (result != KERNEL_STATUS_OK) {                                                                        \
      CUST_KERNEL_LOG_ERROR(ctx, "IRFFTDouble kernel data type [%s] not support.", DTypeStr(DTYPE).c_str()); \
      return result;                                                                                         \
    }                                                                                                        \
    break;                                                                                                   \
  }

template <typename T>
bool PartialDouble(T *input, T *output, const std::vector<int64_t> &input_shape, int64_t n, int64_t dim) {
  int64_t input_nums = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
  int64_t start_pos{1};
  int64_t end_pos = start_pos + n - (n / kOnsideDivisor + 1);

  // compute original offsets for each axes
  std::vector<int64_t> offsets(input_shape.size(), 0);
  for (size_t j = 0; j < input_shape.size(); j++) {
    int64_t pos = static_cast<int64_t>(j);
    offsets[j] = std::accumulate(input_shape.begin() + pos + 1, input_shape.end(), 1, std::multiplies<>());
  }

  for (int64_t i = 0; i < input_nums; ++i) {
    std::vector<int64_t> index(input_shape.size(), 0);
    int64_t flat_index = i;
    // compute original coordinates
    for (size_t dim_index = 0; dim_index < offsets.size(); ++dim_index) {
      index[dim_index] = flat_index / offsets[dim_index];
      flat_index %= offsets[dim_index];
    }
    T ele_val = input[i];
    T factor(kDoubleFactor, 0);
    if (index[dim] >= start_pos && index[dim] < end_pos) {
      ele_val = factor * ele_val;
    }
    output[i] = ele_val;
  }
  return true;
}
}  // namespace

namespace aicpu {
uint32_t IRFFTDoubleCpuKernel::Compute(CpuKernelContext &ctx) {
  op_name_ = GetOpName(ctx);
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "IRFFTDouble check input and output number failed.");
  auto x_type = ctx.Input(kIndex0)->GetDataType();
  switch (x_type) {
    IRFFTDOUBLE_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    IRFFTDOUBLE_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "IRFFTDouble kernel data type [%s] not support.", DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t IRFFTDoubleCpuKernel::IRFFTDoubleCompute(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T *>(ctx.Output(kIndex0)->GetData());

  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  // step1：Get or set attribute.
  int64_t dim = reinterpret_cast<int64_t *>(ctx.Input(kDimIndex)->GetData())[0];
  dim = dim < 0 ? x_rank + dim : dim;

  int64_t n;
  n = reinterpret_cast<int64_t *>(ctx.Input(kNIndex)->GetData())[0];

  auto output_nums = ctx.Output(kIndex0)->NumElements();
  auto ret = memset_s(output_ptr, output_nums * sizeof(T), 0, output_nums * sizeof(T));
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");

  PartialDouble<T>(input_ptr, output_ptr, tensor_shape, n, dim);
  return KERNEL_STATUS_OK;
};

REGISTER_MS_CPU_KERNEL(kIRFFTDouble, IRFFTDoubleCpuKernel);
}  // namespace aicpu
