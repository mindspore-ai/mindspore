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

#include "reversev2.h"
#include <securec.h>
#include "Eigen/Core"

#include "cpu_kernel_utils.h"
#include "iostream"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
using namespace std;
namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const char *kReverseV2 = "ReverseV2";
}  // namespace

namespace aicpu {
uint32_t ReverseV2CpuKernel::Compute(CpuKernelContext &ctx) {
  int x_max_dim = 8;
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "ReverseV2 check input or output is failed.");
  DataType axis_type = ctx.Input(1)->GetDataType();
  KERNEL_CHECK_FALSE((axis_type == DT_INT32 || axis_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of [axis] need be DT_INT32 or DT_INT64.")
  auto x_shape = ctx.Input(0)->GetTensorShape();
  auto axis_shape = ctx.Input(1)->GetTensorShape();
  DataType data_type = DataType(ctx.Input(0)->GetDataType());
  std::vector<int64_t> reverse_shape;
  for (int i = 0; i < x_shape->GetDims(); i++) {
    reverse_shape.push_back(false);
  }
  // dims check
  if (x_shape->GetDims() == 0 || axis_shape->GetDims() == 0) {
    uint32_t ret = ComputeDiffType(data_type, reverse_shape, ctx);
    if (ret != KERNEL_STATUS_OK) {
      return ret;
    }
    return KERNEL_STATUS_OK;
  }
  KERNEL_CHECK_FALSE((x_shape->GetDims() > 0 && x_shape->GetDims() <= x_max_dim), KERNEL_STATUS_PARAM_INVALID,
                     "Shapes of x is not support.")
  KERNEL_CHECK_FALSE((axis_shape->GetDims() == 1), KERNEL_STATUS_PARAM_INVALID, "Shapes of axis is not support.")

  auto input0_datasize = ctx.Input(0)->GetDataSize();
  auto output_datasize = ctx.Output(0)->GetDataSize();
  KERNEL_CHECK_FALSE((input0_datasize == output_datasize), KERNEL_STATUS_PARAM_INVALID,
                     "The data size of input0 [%d] need be same with "
                     "output0 [%d].",
                     input0_datasize, output_datasize)
  int64_t dim = x_shape->GetDims();
  auto input_axis = reinterpret_cast<int64_t *>(ctx.Input(1)->GetData());
  int64_t axis_element = axis_shape->NumElements();
  for (int j = 0; j < axis_element; j++) {
    int64_t realdim = *(input_axis + j) < 0 ? dim + *(input_axis + j) : *(input_axis + j);
    KERNEL_CHECK_FALSE((realdim >= 0 && realdim < dim), KERNEL_STATUS_PARAM_INVALID, "[%d] is invalid", realdim)
    KERNEL_CHECK_FALSE((!reverse_shape[realdim]), KERNEL_STATUS_PARAM_INVALID, "axis [%d], specified more than once.",
                       realdim)
    reverse_shape[realdim] = true;
  }
  uint32_t ret = ComputeDiffType(data_type, reverse_shape, ctx);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  return KERNEL_STATUS_OK;
}

uint32_t ReverseV2CpuKernel::ComputeDiffType(DataType data_type, std::vector<int64_t> reverse_shape,
                                             CpuKernelContext &ctx) {
  switch (data_type) {
    case DT_FLOAT16:
      return ComputeReverseV2<Eigen::half>(reverse_shape, ctx);
    case DT_FLOAT:
      return ComputeReverseV2<float>(reverse_shape, ctx);
    case DT_DOUBLE:
      return ComputeReverseV2<double>(reverse_shape, ctx);
    case DT_UINT8:
      return ComputeReverseV2<uint8_t>(reverse_shape, ctx);
    case DT_INT8:
      return ComputeReverseV2<int8_t>(reverse_shape, ctx);
    case DT_UINT16:
      return ComputeReverseV2<uint16_t>(reverse_shape, ctx);
    case DT_INT16:
      return ComputeReverseV2<int16_t>(reverse_shape, ctx);
    case DT_INT32:
      return ComputeReverseV2<int32_t>(reverse_shape, ctx);
    case DT_INT64:
      return ComputeReverseV2<int64_t>(reverse_shape, ctx);
    case DT_BOOL:
      return ComputeReverseV2<bool>(reverse_shape, ctx);
    case DT_COMPLEX64:
      return ComputeReverseV2<std::complex<float>>(reverse_shape, ctx);
    case DT_COMPLEX128:
      return ComputeReverseV2<std::complex<double>>(reverse_shape, ctx);
    case DT_STRING:
      return ComputeReverseV2<string>(reverse_shape, ctx);
    default:
      KERNEL_LOG_ERROR("ReverseV2 invalid input type[%s]", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ReverseV2CpuKernel::ComputeReverseV2(std::vector<int64_t> reverse_shape, CpuKernelContext &ctx) {
  auto x_shape = ctx.Input(0)->GetTensorShape();
  auto input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  if (x_shape->GetDims() == 0) {
    *(output_data) = *(input_data);
    return KERNEL_STATUS_OK;
  }
  auto axis_shape = ctx.Input(1)->GetTensorShape();
  if (axis_shape->GetDims() == 0) {
    for (int i = 0; i < x_shape->NumElements(); i++) {
      *(output_data + i) = *(input_data + i);
    }
    return KERNEL_STATUS_OK;
  }
  int64_t front = 1;
  int64_t shape_element = x_shape->NumElements();
  int64_t dim = x_shape->GetDims();
  std::vector<int64_t> dims = x_shape->GetDimSizes();
  bool redo = false;
  for (int j = 0; j < dim; j++) {
    front = front * dims[j];
    if (j != dim - 1 && reverse_shape[j] == true) {
      if (redo == true) {
        auto copy_size = shape_element * sizeof(T);
        auto ret_mem = memcpy_s(input_data, copy_size, output_data, copy_size);
        KERNEL_CHECK_FALSE(ret_mem == EOK, KERNEL_STATUS_INNER_ERROR, "Memcpy failed, size = [%zu].", copy_size);
      }
      int64_t row_size = shape_element / front;
      int64_t input_forward = (dims[j] - 1) * row_size;
      int64_t save = input_forward;
      int64_t output_forward = 0;
      int64_t behind = shape_element / (front / dims[j]);
      for (int k = 0; k < front / dims[j]; k++) {
        int64_t remain = dims[j];
        while (remain > 0) {
          auto copy_size = row_size * sizeof(T);
          auto cur_output = output_data + output_forward;
          auto cur_input = input_data + input_forward;
          auto ret_mem = memcpy_s(cur_output, copy_size, cur_input, copy_size);
          KERNEL_CHECK_FALSE(ret_mem == EOK, KERNEL_STATUS_INNER_ERROR, "Memcpy size[%zu] from input to output failed.",
                             copy_size);
          input_forward = input_forward - row_size;
          output_forward = output_forward + row_size;
          remain--;
        }
        save = save + behind;
        input_forward = save;
      }
      redo = true;
    } else if (j == dim - 1 && reverse_shape[j] == true) {
      if (redo == true) {
        auto copy_size = shape_element * sizeof(T);
        auto ret_mem = memcpy_s(input_data, copy_size, output_data, copy_size);
        KERNEL_CHECK_FALSE(ret_mem == EOK, KERNEL_STATUS_INNER_ERROR, "Memcpy failed, size = [%zu].", copy_size);
      }
      int64_t output_forward = 0;
      for (int k = 0; k < shape_element / dims[j]; k++) {
        for (int i = dims[j] - 1; i >= 0; i--) {
          *(output_data + output_forward) = *(input_data + i + k * dims[j]);
          output_forward++;
        }
      }
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kReverseV2, ReverseV2CpuKernel);
}  // namespace aicpu
