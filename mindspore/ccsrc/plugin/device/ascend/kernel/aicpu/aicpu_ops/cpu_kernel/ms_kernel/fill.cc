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
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *const kFill = "Fill";
}

namespace aicpu {
template <typename T>
void FillGenerateCase(Tensor *&value_tensor, Tensor *&output) {
  auto value = *(reinterpret_cast<T *>(value_tensor->GetData()));
  if (AddrAlignedCheck(output->GetData())) {
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> eigen_output(static_cast<T *>(output->GetData()),
                                                                       output->GetTensorShape()->NumElements());
    eigen_output.setConstant(value);
  } else {
    Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Unaligned> eigen_output(static_cast<T *>(output->GetData()),
                                                                         output->GetTensorShape()->NumElements());
    eigen_output.setConstant(value);
  }
}

uint32_t FillCpuKernel::GetDimsByType(CpuKernelContext &ctx) {
  dims.clear();
  Tensor *dims_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(dims_tensor, KERNEL_STATUS_PARAM_INVALID, "Get dims input failed")
  uint32_t ret;
  auto dims_dtype = dims_tensor->GetDataType();
  switch (dims_dtype) {
    case (DT_INT32):
      ret = CalcDims<int32_t>(dims_tensor, dims);
      break;
    case (DT_INT64):
      ret = CalcDims<int64_t>(dims_tensor, dims);
      break;
    default:
      KERNEL_LOG_ERROR(
        "Fill kernel dims data_type [%u] not support, support data_types: "
        "DT_INT32, DT_INT64",
        dims_dtype);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ret != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Fill kernel calculate dims failed");
  }
  return ret;
}

uint32_t FillCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t check = GetDimsByType(ctx);
  if (check != KERNEL_STATUS_OK) {
    return check;
  }
  Tensor *value_tensor = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(value_tensor, KERNEL_STATUS_PARAM_INVALID, "Get value input failed")
  KERNEL_CHECK_NULLPTR(value_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get value input data failed")
  KERNEL_CHECK_NULLPTR(value_tensor->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get value input shape failed")
  if (!value_tensor->GetTensorShape()->GetDimSizes().empty()) {
    KERNEL_LOG_ERROR("Fill kernel value input is not a scalar.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *output = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Get output failed")
  KERNEL_CHECK_NULLPTR(output->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  KERNEL_CHECK_NULLPTR(output->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get output shape failed")
  if (output->GetTensorShape()->GetDimSizes() != dims) {
    KERNEL_LOG_ERROR("Fill kernel output shape not matched.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto input_dtype = value_tensor->GetDataType();
  auto output_dtype = output->GetDataType();
  if (input_dtype != output_dtype) {
    KERNEL_LOG_ERROR("Fill kernel data type not matched, value input dtype [%u], output dtype [%u].", input_dtype,
                     output_dtype);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  std::map<int, std::function<void(Tensor *&, Tensor *&)>> calls;
  calls[DT_INT8] = FillGenerateCase<int8_t>;
  calls[DT_UINT8] = FillGenerateCase<uint8_t>;
  calls[DT_INT16] = FillGenerateCase<int16_t>;
  calls[DT_UINT16] = FillGenerateCase<uint16_t>;
  calls[DT_INT32] = FillGenerateCase<int32_t>;
  calls[DT_UINT32] = FillGenerateCase<uint32_t>;
  calls[DT_INT64] = FillGenerateCase<int64_t>;
  calls[DT_UINT64] = FillGenerateCase<uint64_t>;
  calls[DT_BOOL] = FillGenerateCase<bool>;
  calls[DT_FLOAT16] = FillGenerateCase<Eigen::half>;
  calls[DT_FLOAT] = FillGenerateCase<float>;
  calls[DT_DOUBLE] = FillGenerateCase<double>;

  if (calls.find(output_dtype) == calls.end()) {
    KERNEL_LOG_ERROR("Fill kernel data type [%u] not support", output_dtype);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  calls[output_dtype](value_tensor, output);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FillCpuKernel::CalcDims(const Tensor *dims_tensor, std::vector<int64_t> &dim_vec) {
  uint64_t data_num = dims_tensor->GetDataSize() / sizeof(T);
  if (data_num == 0) {
    KERNEL_LOG_INFO("Fill kernel: dims is empty, fill scalar output.");
    return KERNEL_STATUS_OK;
  }

  KERNEL_CHECK_NULLPTR(dims_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get dims data failed")
  for (uint64_t i = 0; i < data_num; i++) {
    auto dim = *(reinterpret_cast<const T *>(dims_tensor->GetData()) + i);
    if (dim < 0) {
      KERNEL_LOG_ERROR("Fill kernel: input dim [%llu] is negative, value=[%lld]", i, static_cast<int64_t>(dim));
      return KERNEL_STATUS_PARAM_INVALID;
    }
    // zero dim is different from empty dim.
    if (dim == 0) {
      KERNEL_LOG_INFO("Fill kernel: input dim [%llu] is zero", i);
    }
    dim_vec.emplace_back(dim);
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kFill, FillCpuKernel);
}  // namespace aicpu
