/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#include "fill.h"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kFill = "Fill";
const char *kFillV2 = "FillV2";
const int64_t kParallelDataNumCriticalPoint1 = 128 * 1024;
const int64_t kParallelDataNumCriticalPoint2 = 2 * 1024 * 1024;

#define CALCULATE_DIMS_DTYPE_CASE(DTYPE, TYPE)                        \
  case (DTYPE): {                                                     \
    if (CalculateDims<TYPE>(dims_tensor, dims) != KERNEL_STATUS_OK) { \
      KERNEL_LOG_ERROR("Fill kernel calculate dims failed.");         \
      return KERNEL_STATUS_PARAM_INVALID;                             \
    }                                                                 \
    break;                                                            \
  }

#define FILL_GENERATE_DTYPE_CASE(DTYPE, TYPE)    \
  case (DTYPE): {                                \
    FillOutput<TYPE>(ctx, value_tensor, output); \
    break;                                       \
  }
}  // namespace

namespace aicpu {
uint32_t FillCpuKernel::Compute(CpuKernelContext &ctx) {
  // 校验输入个数和输出个数，以及输入和输入tensor的属性是否为空
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check input and output number failed.");

  std::vector<int64_t> dims;
  Tensor *dims_tensor = ctx.Input(0);
  auto dims_dtype = dims_tensor->GetDataType();
  switch (dims_dtype) {
    CALCULATE_DIMS_DTYPE_CASE(DT_INT32, int32_t)
    CALCULATE_DIMS_DTYPE_CASE(DT_INT64, int64_t)
    default:
      KERNEL_LOG_ERROR("Fill kernel dims data_type [%u] not support, support data_types: DT_INT32, DT_INT64.",
                       dims_dtype);
      return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *value_tensor = ctx.Input(1);
  if (value_tensor->NumElements() != 1) {
    KERNEL_LOG_ERROR("Fill kernel value input is not a scalar.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *output = ctx.Output(0);
  if (output->GetTensorShape()->GetDims() != static_cast<int64_t>(dims.size())) {
    KERNEL_LOG_ERROR("Fill kernel output shape not matched.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (output->GetTensorShape()->GetDimSizes() != dims) {
    output->GetTensorShape()->SetDimSizes(dims);
  }

  auto input_dtype = value_tensor->GetDataType();
  auto output_dtype = output->GetDataType();
  if (input_dtype != output_dtype) {
    KERNEL_LOG_ERROR(
      "Fill kernel data type not matched, value input dtype [%u], output dtype [%u], support data_types: "
      "DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16, "
      "DT_UINT32, DT_UINT64, DT_UINT8, DT_BOOL.",
      input_dtype, output_dtype);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (output_dtype) {
    FILL_GENERATE_DTYPE_CASE(DT_INT8, int8_t)
    FILL_GENERATE_DTYPE_CASE(DT_UINT8, uint8_t)
    FILL_GENERATE_DTYPE_CASE(DT_INT16, int16_t)
    FILL_GENERATE_DTYPE_CASE(DT_UINT16, uint16_t)
    FILL_GENERATE_DTYPE_CASE(DT_INT32, int32_t)
    FILL_GENERATE_DTYPE_CASE(DT_UINT32, uint32_t)
    FILL_GENERATE_DTYPE_CASE(DT_INT64, int64_t)
    FILL_GENERATE_DTYPE_CASE(DT_UINT64, uint64_t)
    FILL_GENERATE_DTYPE_CASE(DT_BOOL, bool)
    FILL_GENERATE_DTYPE_CASE(DT_FLOAT16, Eigen::half)
    FILL_GENERATE_DTYPE_CASE(DT_FLOAT, float)
    FILL_GENERATE_DTYPE_CASE(DT_DOUBLE, double)
    FILL_GENERATE_DTYPE_CASE(DT_COMPLEX64, std::complex<float>)
    FILL_GENERATE_DTYPE_CASE(DT_COMPLEX128, std::complex<double>)
    default:
      KERNEL_LOG_ERROR(
        "Fill kernel data type [%u] not support, not support data_types: DT_STRING, DT_DUAL_SUB_INT8, "
        "DT_DUAL_SUB_UINT8, DT_QUINT8, DT_QINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_RESOURCE, DT_STRING_REF, "
        "DT_DUAL, DT_UNDEFINED.",
        output_dtype);
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FillCpuKernel::CalculateDims(const Tensor *dims_tensor, std::vector<int64_t> &dims) {
  // 获取第一个输入tensor中的元素个数，第一个输入是一个一维的tensor(dims_tensor)
  uint64_t data_num = dims_tensor->GetDataSize() / sizeof(T);
  auto dims_data = reinterpret_cast<const T *>(dims_tensor->GetData());

  for (uint64_t i = 0; i < data_num; i++) {
    auto dim = *(dims_data + i);
    if (dim < 0) {
      KERNEL_LOG_ERROR("dims input dim [%llu] is negative, value=[%lld].", i, static_cast<int64_t>(dim));
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (dim == 0) {
      KERNEL_LOG_INFO("dims input dim [%llu] is zero.", i);
      dims.clear();
      break;
    }
    dims.emplace_back(dim);
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
void FillCpuKernel::FillOutput(CpuKernelContext &ctx, const Tensor *value_tensor, Tensor *output) {
  auto value = reinterpret_cast<T *>(value_tensor->GetData());
  auto output_data = reinterpret_cast<T *>(output->GetData());
  int64_t data_num = output->NumElements();

  if (data_num >= kParallelDataNumCriticalPoint1) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));

    if (data_num <= kParallelDataNumCriticalPoint2) {
      max_core_num = std::min(max_core_num, 4U);
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto shared_fill = [&](int64_t start, int64_t end) { SpecialFillOutput<T>(start, end, output_data, value); };

    CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_fill);
  } else {
    SpecialFillOutput<T>(0, data_num, output_data, value);
  }
}

template <typename T>
void FillCpuKernel::SpecialFillOutput(int64_t start, int64_t end, T *output_data, const T *value) {
  for (int64_t i = start; i < end; i++) {
    *(output_data + i) = *(value);
  }
}

REGISTER_CPU_KERNEL(kFill, FillCpuKernel);
REGISTER_CPU_KERNEL(kFillV2, FillCpuKernel);
}  // namespace aicpu
