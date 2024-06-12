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

#include <set>
#include <cmath>
#include "cpu_kernel/ms_kernel/fft_ortho.h"
#include "context/inc/cpu_kernel_utils.h"
#include "mindspore/core/mindapi/base/types.h"
#include "utils/kernel_util.h"
#include "cpu_kernel/utils/fft_helper.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const uint32_t kIndex0 = 0;
const uint32_t kAxesIndex = 1;
const char *kFFTOrtho = "FFTOrtho";

#define FFTORTHO_COMPUTE_CASE(DTYPE, TYPE, CTX)                                                           \
  case (DTYPE): {                                                                                         \
    uint32_t result = FFTOrthoCompute<TYPE>(CTX);                                                         \
    if (result != KERNEL_STATUS_OK) {                                                                     \
      CUST_KERNEL_LOG_ERROR(ctx, "FFTOrtho kernel data type [%s] not support.", DTypeStr(DTYPE).c_str()); \
      return result;                                                                                      \
    }                                                                                                     \
    break;                                                                                                \
  }

template <typename T>
bool Orthogonalize(T *input, T *output, const std::vector<int64_t> &input_shape, const std::vector<int64_t> dims,
                   bool forward) {
  int64_t input_nums = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
  std::set<int64_t> ortho_dims(dims.begin(), dims.end());
  T normal_factor{2};
  T head_factor{4};

  // compute original offsets for each axes
  std::vector<int64_t> offsets(input_shape.size(), 0);
  for (size_t j = 0; j < input_shape.size(); j++) {
    int64_t pos = static_cast<int64_t>(j);
    offsets[j] = std::accumulate(input_shape.begin() + pos + 1, input_shape.end(), 1, std::multiplies<>());
  }

  for (int64_t i = 0; i < input_nums; ++i) {
    std::vector<int64_t> index(input_shape.size(), 0);
    int64_t flat_index = i;
    T ele_factor{1};
    // compute original coordinates
    for (size_t dim_index = 0; dim_index < offsets.size(); ++dim_index) {
      index[dim_index] = flat_index / offsets[dim_index];
      flat_index %= offsets[dim_index];
      if (ortho_dims.find(static_cast<int64_t>(dim_index)) != ortho_dims.end()) {
        ele_factor = index[dim_index] == 0 ? ele_factor * head_factor : ele_factor * normal_factor;
        ele_factor *= static_cast<T>(input_shape[dim_index]);
      }
    }
    T ele_val = input[i];
    if (forward) {
      ele_val /= std::sqrt(ele_factor);
    } else {
      ele_val *= std::sqrt(ele_factor);
    }

    output[i] = ele_val;
  }
  return true;
}
}  // namespace

namespace aicpu {
uint32_t FFTOrthoCpuKernel::Compute(CpuKernelContext &ctx) {
  op_name_ = GetOpName(ctx);
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "FFTOrtho check input and output number failed.");
  auto x_type = ctx.Input(kIndex0)->GetDataType();
  switch (x_type) {
    FFTORTHO_COMPUTE_CASE(DT_FLOAT, float, ctx)
    FFTORTHO_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "FFTOrtho kernel data type [%s] not support.", DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FFTOrthoCpuKernel::FFTOrthoCompute(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T *>(ctx.Output(kIndex0)->GetData());

  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  // step1：Get or set attribute.
  auto dim_tensor = reinterpret_cast<int64_t *>(ctx.Input(kAxesIndex)->GetData());
  auto dim_tensor_size = ctx.Input(kAxesIndex)->NumElements();
  std::vector<int64_t> dims;
  for (int64_t i = 0; i < dim_tensor_size; i++) {
    int64_t tmp_pos = dim_tensor[i] < 0 ? x_rank + dim_tensor[i] : dim_tensor[i];
    (void)dims.emplace_back(tmp_pos);
  }

  AttrValue *attr = ctx.GetAttr("forward");
  CUST_KERNEL_CHECK_NULLPTR(ctx, attr, KERNEL_STATUS_PARAM_INVALID, "Get param[forward] failed.")
  bool forward = attr->GetBool();

  auto output_nums = ctx.Output(kIndex0)->NumElements();
  auto ret = memset_s(output_ptr, output_nums * sizeof(T), 0, output_nums * sizeof(T));
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");

  Orthogonalize<T>(input_ptr, output_ptr, tensor_shape, dims, forward);
  return KERNEL_STATUS_OK;
};

REGISTER_MS_CPU_KERNEL(kFFTOrtho, FFTOrthoCpuKernel);
}  // namespace aicpu
