/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "pack.h"
#include <securec.h>
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "Eigen/Core"

namespace {
const uint32_t kOutputNum{1u};
const uint32_t kInputNum{aicpu::kDynamicInput};
const char *kPack = "Pack";
// constexpr int64_t kParallelDataNums = 512 * 1024;

#define PACK_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                      \
    uint32_t result = PackCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                  \
      KERNEL_LOG_ERROR("Pack kernel compute failed."); \
      return result;                                   \
    }                                                  \
    break;                                             \
  }
}  // namespace

namespace aicpu {
uint32_t PackCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kPack);
  KERNEL_HANDLE_ERROR(PackCheck(ctx), "[%s] check params failed.", kPack);
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    PACK_COMPUTE_CASE(DT_BOOL, bool, ctx)
    PACK_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    PACK_COMPUTE_CASE(DT_FLOAT, float, ctx)
    PACK_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    PACK_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    PACK_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    PACK_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    PACK_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    PACK_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    PACK_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    PACK_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    PACK_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    PACK_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    PACK_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Pack kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t PackCpuKernel::PackCheck(CpuKernelContext &ctx) {
  auto *input = ctx.Input(0);
  AttrValue *n_attr = ctx.GetAttr("num");
  AttrValue *axis_attr = ctx.GetAttr("axis");
  int64_t axis = axis_attr->GetInt();
  auto expanded_num_dims = input->GetTensorShape()->GetDims() + 1;  // first_input.dims() + 1;
  if (axis < 0) axis += expanded_num_dims;

  if (axis < 0 || axis >= expanded_num_dims) {
    KERNEL_LOG_ERROR("Pack axis error.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int64_t input_num = n_attr->GetInt();
  auto x1_dims = input->GetTensorShape()->GetDims();
  for (int64_t i = 1; i < input_num; i++) {
    auto input_dims = ctx.Input(i)->GetTensorShape()->GetDims();
    if (x1_dims != input_dims) {
      KERNEL_LOG_ERROR("Pack input dims no equal.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PackCpuKernel::PackCompute(CpuKernelContext &ctx) {
  AttrValue *axis_attr = ctx.GetAttr("axis");
  int64_t axis = axis_attr->GetInt();

  AttrValue *n_attr = ctx.GetAttr("num");
  int64_t input_num = n_attr->GetInt();

  auto *input = ctx.Input(0);
  auto *output = ctx.Output(0);

  auto expanded_num_dims = input->GetTensorShape()->GetDims() + 1;
  if (axis < 0) axis += expanded_num_dims;

  std::vector<int64_t> temp_shape = input->GetTensorShape()->GetDimSizes();
  temp_shape.insert(temp_shape.begin() + axis, input_num);

  auto *y = reinterpret_cast<T *>(output->GetData());
  int64_t x_NumElements = input->GetTensorShape()->NumElements();

  if (axis == 0) {
    int64_t num = 0;
    for (int64_t j = 0; j < input_num; j++) {
      auto *input_x = reinterpret_cast<T *>(ctx.Input(j)->GetData());
      auto input_numelements = ctx.Input(j)->GetTensorShape()->NumElements();
      for (int64_t i = 0; i < input_numelements; i++) {
        *(y + num) = *(input_x + i);
        num++;
      }
    }
  } else {
    int64_t num = 0;
    for (int64_t j = 0; j < x_NumElements; j++) {
      for (int64_t i = 0; i < input_num; i++) {
        auto *input_x = reinterpret_cast<T *>(ctx.Input(i)->GetData());
        *(y + num) = *(input_x + j);
        num++;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kPack, PackCpuKernel);
}  // namespace aicpu
