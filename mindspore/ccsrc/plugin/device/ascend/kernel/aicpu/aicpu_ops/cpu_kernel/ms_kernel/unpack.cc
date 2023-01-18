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
#include "unpack.h"
#include "utils/kernel_util.h"

namespace {
const char *kUnpack = "Unpack";
}

namespace aicpu {
uint32_t UnpackCpuKernel::CheckAndInitParams(CpuKernelContext &ctx) {
  Tensor *value_ptr = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(value_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input value failed.");
  value_data_ptr = value_ptr->GetData();
  KERNEL_CHECK_NULLPTR(value_data_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input value data failed.");
  auto value_shape_ptr = value_ptr->GetTensorShape();
  KERNEL_CHECK_NULLPTR(value_shape_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input value shape failed.");
  int64_t value_dim = value_shape_ptr->GetDims();

  AttrValue *unpack_axis_ptr = ctx.GetAttr("axis");
  int64_t real_unpack_axis = 0;
  KERNEL_CHECK_FALSE(unpack_axis_ptr, KERNEL_STATUS_PARAM_INVALID, "get axis failed!");
  unpack_axis = unpack_axis_ptr->GetInt();
  real_unpack_axis = unpack_axis >= 0 ? unpack_axis : unpack_axis + value_dim;
  KERNEL_CHECK_FALSE(value_dim > real_unpack_axis, KERNEL_STATUS_PARAM_INVALID,
                     "The axis value range should be [-value_dim, value_dim), "
                     "value dim is [%d], axis is [%d].",
                     value_dim, unpack_axis);
  unpack_axis = real_unpack_axis;

  AttrValue *unpack_num_ptr = ctx.GetAttr("num");
  KERNEL_CHECK_FALSE(unpack_num_ptr, KERNEL_STATUS_PARAM_INVALID, "get num failed!");
  int64_t axis_size = value_shape_ptr->GetDimSize(unpack_axis);
  unpack_num = unpack_num_ptr->GetInt();
  KERNEL_CHECK_FALSE(unpack_num == axis_size, KERNEL_STATUS_PARAM_INVALID,
                     "The num you want to unpack to should be equal to the "
                     "size of the specified dimension. "
                     "The num you want to unpack to is [%d], while the [%d] "
                     "dim's size is [%d].",
                     unpack_num, unpack_axis, axis_size);
  value_shape_vec = value_shape_ptr->GetDimSizes();
  data_type = value_ptr->GetDataType();
  value_num = value_ptr->NumElements();

  output_ptr_vec.resize(unpack_num);
  for (int64_t i = 0; i < unpack_num; i++) {
    Tensor *output_ptr = ctx.Output(i);
    KERNEL_CHECK_NULLPTR(output_ptr, KERNEL_STATUS_PARAM_INVALID, "Get output [%d] failed.", i);
    auto output_data_ptr = output_ptr->GetData();
    KERNEL_CHECK_NULLPTR(output_data_ptr, KERNEL_STATUS_PARAM_INVALID, "Get output data [%d] failed.", i);
    output_ptr_vec[i] = output_data_ptr;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t UnpackCpuKernel::UnpackWithOneOutput(T *input_data_ptr, std::vector<T *> output_data_vec) {
  int64_t copy_size = value_num * sizeof(T);
  auto mem_ret = memcpy_s(output_data_vec[0], copy_size, input_data_ptr, copy_size);
  KERNEL_CHECK_FALSE((mem_ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                     "Memcpy size[%zu] from input value to output[0] failed.", copy_size);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t UnpackCpuKernel::UnpackWithDimZero(T *input_data_ptr, std::vector<T *> output_data_vec) {
  if (value_shape_vec[0] == 0) {
    KERNEL_CHECK_FALSE(value_shape_vec[0] > 0, KERNEL_STATUS_PARAM_INVALID, "The shape of input tensor is invalid.");
  }
  int64_t copy_num = value_num / value_shape_vec[0];
  T *input_copy_ptr = input_data_ptr;
  for (int64_t i = 0; i < unpack_num; i++) {
    int64_t copy_size_per = copy_num;
    int64_t copy_size = copy_size_per * sizeof(T);
    auto mem_ret = memcpy_s(output_data_vec[i], copy_size, input_copy_ptr, copy_size);
    KERNEL_CHECK_FALSE((mem_ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                       "Memcpy size[%zu] from input value to output[%d] failed.", copy_size, i);
    input_copy_ptr += copy_size_per;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t UnpackCpuKernel::UnpackCompute(T *input_data_ptr, std::vector<T *> output_data_vec, CpuKernelContext &ctx) {
  int64_t prefix = 1;
  for (uint64_t i = 0; i < unpack_axis; i++) {
    if (value_shape_vec[i] == 0) {
      KERNEL_CHECK_FALSE(value_shape_vec[i] > 0, KERNEL_STATUS_PARAM_INVALID, "The shape of input tensor is invalid.");
    }
    prefix *= value_shape_vec[i];
  }
  if (unpack_axis >= value_shape_vec.size()) {
    KERNEL_CHECK_FALSE(unpack_axis < value_shape_vec.size(), KERNEL_STATUS_PARAM_INVALID,
                       "input attr axis is invalid.");
  }
  int64_t midfix = value_shape_vec[unpack_axis];
  int64_t subfix = 1;
  for (size_t i = unpack_axis + 1; i < value_shape_vec.size(); i++) {
    if (value_shape_vec[i] == 0) {
      KERNEL_CHECK_FALSE(value_shape_vec[i] > 0, KERNEL_STATUS_PARAM_INVALID, "The shape of input tensor is invalid.");
    }
    subfix *= value_shape_vec[i];
  }

  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  if (max_core_num > unpack_num) {
    max_core_num = unpack_num;
  }

  auto shard_unpack = [&](size_t start, size_t end) {
    int64_t offset = 0;
    for (uint64_t i = start; i < end; i++) {
      offset = i * subfix;
      T *output_data_ptr = output_data_vec[i];
      T *input_copy_ptr = input_data_ptr + offset;
      int64_t copy_size = subfix * sizeof(T);
      for (int64_t j = 0; j < prefix; j++) {
        auto mem_ret = memcpy_s(output_data_ptr, copy_size, input_copy_ptr, copy_size);
        KERNEL_CHECK_FALSE((mem_ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                           "Memcpy size[%zu] from input value to output[%d] failed.", copy_size, i);
        input_copy_ptr += (subfix * midfix);
        output_data_ptr += subfix;
      }
    }
    return KERNEL_STATUS_OK;
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, unpack_num, unpack_num / max_core_num, shard_unpack),
                      "Unpack Compute failed.");

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t UnpackCpuKernel::DoCompute(CpuKernelContext &ctx) {
  T *input_data_ptr = reinterpret_cast<T *>(value_data_ptr);
  std::vector<T *> output_data_vec;
  output_data_vec.resize(unpack_num);
  for (int64_t i = 0; i < unpack_num; i++) {
    output_data_vec[i] = reinterpret_cast<T *>(output_ptr_vec[i]);
  }
  if (unpack_num == 1) {
    KERNEL_CHECK_FALSE((UnpackWithOneOutput<T>(input_data_ptr, output_data_vec) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "UnpackWithOneOutput failed.");
    return KERNEL_STATUS_OK;
  }
  if (unpack_axis == 0) {
    KERNEL_CHECK_FALSE((UnpackWithDimZero<T>(input_data_ptr, output_data_vec) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "UnpackWithDimZero failed.");
    return KERNEL_STATUS_OK;
  }
  KERNEL_CHECK_FALSE((UnpackCompute<T>(input_data_ptr, output_data_vec, ctx) == KERNEL_STATUS_OK),
                     KERNEL_STATUS_PARAM_INVALID, "Unpack Compute failed.");
  return KERNEL_STATUS_OK;
}

uint32_t UnpackCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_CHECK_FALSE((CheckAndInitParams(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "CheckAndInitParams failed.");
  switch (data_type) {
    case DT_FLOAT16:
      return DoCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return DoCompute<float>(ctx);
    case DT_DOUBLE:
      return DoCompute<double>(ctx);
    case DT_BOOL:
      return DoCompute<bool>(ctx);
    case DT_INT8:
      return DoCompute<int8_t>(ctx);
    case DT_INT16:
      return DoCompute<int16_t>(ctx);
    case DT_INT32:
      return DoCompute<int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t>(ctx);
    case DT_UINT8:
      return DoCompute<uint8_t>(ctx);
    case DT_UINT16:
      return DoCompute<uint16_t>(ctx);
    case DT_UINT32:
      return DoCompute<uint32_t>(ctx);
    case DT_UINT64:
      return DoCompute<uint64_t>(ctx);
    case DT_COMPLEX64:
      return DoCompute<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return DoCompute<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported data type [%s]", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(kUnpack, UnpackCpuKernel);
}  // namespace aicpu