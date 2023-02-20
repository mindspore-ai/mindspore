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
#include "sparse_to_dense.h"
#include <securec.h>
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace aicpu {
const char *const SPARSETODENSE = "SparseToDense";
constexpr int64_t kParallelDateSize = 16 * 1024;
constexpr int64_t kCopyDataSize = 1024;
constexpr uint32_t kInput0 = 0;
constexpr uint32_t kInput1 = 1;
constexpr uint32_t kInput2 = 2;
constexpr uint32_t kInput3 = 3;
constexpr uint32_t kOutput0 = 0;
constexpr int32_t kRank = 2;
}  // namespace aicpu

namespace aicpu {
uint32_t SparseToDenseCpuKernel::SparseToDense(const CpuKernelContext &ctx, SparseTensor &st, const Tensor *indices,
                                               Tensor *output) {
  KERNEL_LOG_INFO("Start to execute SparseToDense");
  if (indices == nullptr || output == nullptr) {
    KERNEL_LOG_ERROR("Indices or output tensor is nullptr.");
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }

  DataType dt = static_cast<DataType>(output->GetDataType());
  switch (dt) {
    case DT_INT8:
      return EigenSparseToDense<int8_t>(ctx, st, indices, output);
    case DT_UINT8:
      return EigenSparseToDense<uint8_t>(ctx, st, indices, output);
    case DT_INT16:
      return EigenSparseToDense<int16_t>(ctx, st, indices, output);
    case DT_UINT16:
      return EigenSparseToDense<uint16_t>(ctx, st, indices, output);
    case DT_INT32:
      return EigenSparseToDense<int32_t>(ctx, st, indices, output);
    case DT_INT64:
      return EigenSparseToDense<int64_t>(ctx, st, indices, output);
    case DT_FLOAT16:
      return EigenSparseToDense<Eigen::half>(ctx, st, indices, output);
    case DT_FLOAT:
      return EigenSparseToDense<float>(ctx, st, indices, output);
    case DT_BOOL:
      return EigenSparseToDense<bool>(ctx, st, indices, output);
    case DT_DOUBLE:
      return EigenSparseToDense<double>(ctx, st, indices, output);
    default:
      KERNEL_LOG_ERROR("Sparse to dense can't support this data type [%d].", static_cast<int32_t>(dt));
      return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
}

KernelStatus SparseToDenseCpuKernel::ValidParam(const CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("Start to execute ValidParam");
  // valid input and output nullptr
  Tensor *indices_tensor = ctx.Input(0);
  Tensor *shape_tensor = ctx.Input(1);
  Tensor *sparse_values = ctx.Input(2);
  Tensor *default_value_tensor = ctx.Input(3);
  Tensor *output_tensor = ctx.Output(0);
  bool validNull = ((output_tensor == nullptr) || default_value_tensor == nullptr || (sparse_values == nullptr) ||
                    (indices_tensor == nullptr) || (shape_tensor == nullptr));
  if (validNull) {
    KERNEL_LOG_ERROR("Got input or output param is nullptr.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // valid shape nullptr
  auto output_shape = shape_tensor->GetTensorShape();
  auto values_shape = sparse_values->GetTensorShape();
  auto default_value_shape = default_value_tensor->GetTensorShape();
  auto indices_shape = indices_tensor->GetTensorShape();
  bool validShapeNull = ((default_value_shape == nullptr) || values_shape == nullptr || (output_shape == nullptr) ||
                         (indices_shape == nullptr));
  if (validShapeNull) {
    KERNEL_LOG_ERROR("Got input shape is nullptr.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // sparse_indices
  if (indices_shape->GetDims() > kRank) {
    KERNEL_LOG_ERROR(
      "Sparse_indices should be a scalar, vector, or matrix, got dim "
      "size [%d].",
      indices_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const int64_t elems_num = indices_shape->GetDims() > 0 ? indices_shape->GetDimSize(0) : 1;
  const int64_t dims_num = indices_shape->GetDims() > 1 ? indices_shape->GetDimSize(1) : 1;

  // output_shape
  if (output_shape->GetDims() != 1) {
    KERNEL_LOG_ERROR("Output_shape should be a vector, and got dim size [%d].", output_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape_tensor->NumElements() != dims_num) {
    KERNEL_LOG_ERROR("Output_shape has incorrect number of elements [%lld], should be [%lld]",
                     shape_tensor->NumElements(), dims_num);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // valid data type
  DataType IndiceType = indices_tensor->GetDataType();
  DataType outShapeType = shape_tensor->GetDataType();
  bool validIndiceType = ((IndiceType != DT_INT32) && (IndiceType != DT_INT64));
  bool validShapeType = ((outShapeType != DT_INT32) && (outShapeType != DT_INT64));
  if (validShapeType || validIndiceType) {
    KERNEL_LOG_ERROR(
      "Valid indice or output shape data type failed, indiceType [%d], "
      "shapeType [%d].",
      static_cast<int>(IndiceType), static_cast<int>(outShapeType));
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // sparse_values
  int32_t values_dims_size = values_shape->GetDims();
  if ((values_dims_size != 0) && (values_dims_size != 1)) {
    KERNEL_LOG_ERROR("Values_shape should be a scalar or a vector, got dim size [%d].", values_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if ((values_dims_size == 1) && (sparse_values->NumElements() != elems_num)) {
    KERNEL_LOG_ERROR("Values_shape has incorrect number of elements [%lld], should be [%lld]",
                     sparse_values->NumElements(), elems_num);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // default_value
  if (default_value_shape->GetDims() != 0 && default_value_tensor->NumElements() != 1) {
    KERNEL_LOG_ERROR("Default_value should be a scalar, and got dim size [%d].", default_value_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_LOG_INFO("Execute ValidParam end.");
  return KERNEL_STATUS_OK;
}

uint32_t SparseToDenseCpuKernel::ParallelSetDefaultValue(const CpuKernelContext &ctx,
                                                         const Tensor *default_value_tensor,
                                                         const Tensor *output_tensor, int64_t output_size) {
  auto type_size = GetSizeByDataType(static_cast<DataType>(output_tensor->GetDataType()));
  char *default_value_addr = reinterpret_cast<char *>(default_value_tensor->GetData());
  char *output_addr = reinterpret_cast<char *>(output_tensor->GetData());
  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
  auto default_value = [&](std::int64_t begin, std::int64_t end) {
    int64_t total = end - begin;
    int64_t remainder = total % kCopyDataSize;
    int64_t piece = total / kCopyDataSize;
    if (piece == 0) {
      for (int64_t index = begin; index < end; index++) {
        (void)memcpy_s(output_addr + (index * type_size), static_cast<size_t>(type_size), default_value_addr,
                       static_cast<size_t>(type_size));
      }
    } else {
      for (int64_t index = begin; index < begin + kCopyDataSize; index++) {
        (void)memcpy_s(output_addr + (index * type_size), static_cast<size_t>(type_size), default_value_addr,
                       static_cast<size_t>(type_size));
      }
      char *temp_addr = output_addr + (begin * type_size);
      size_t data_size = static_cast<size_t>(type_size * kCopyDataSize);
      for (int64_t loop = 1; loop < piece; loop++) {
        (void)memcpy_s(temp_addr + (loop * type_size * kCopyDataSize), data_size, temp_addr, data_size);
      }
      char *temp_addr1 = output_addr + (begin * type_size) + (piece * type_size * kCopyDataSize);
      for (int64_t loop1 = 0; loop1 < remainder; loop1++) {
        (void)memcpy_s(temp_addr1 + (loop1 * type_size), static_cast<size_t>(type_size), default_value_addr,
                       static_cast<size_t>(type_size));
      }
    }
  };
  return CpuKernelUtils::ParallelFor(ctx, output_size, output_size / max_core_num, default_value);
}
uint32_t SparseToDenseCpuKernel::SetDefaultValue(const CpuKernelContext &ctx, const Tensor *default_value_tensor,
                                                 const Tensor *output_tensor, int64_t output_size) {
  auto type_size = GetSizeByDataType(static_cast<DataType>(output_tensor->GetDataType()));
  if (type_size < 1) {
    KERNEL_LOG_ERROR("Don't support output tensor types");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  char *default_value_addr = reinterpret_cast<char *>(default_value_tensor->GetData());
  char *output_addr = reinterpret_cast<char *>(output_tensor->GetData());
  if (output_size < kParallelDateSize) {
    int64_t remainder = output_size % kCopyDataSize;
    int64_t piece = output_size / kCopyDataSize;
    if (piece == 0) {
      for (int index = 0; index < output_size; index++) {
        (void)memcpy_s(output_addr + (index * type_size), static_cast<size_t>(type_size), default_value_addr,
                       static_cast<size_t>(type_size));
      }
    } else {
      for (int index = 0; index < kCopyDataSize; index++) {
        (void)memcpy_s(output_addr + (index * type_size), static_cast<size_t>(type_size), default_value_addr,
                       static_cast<size_t>(type_size));
      }
      size_t data_size = static_cast<size_t>(type_size * kCopyDataSize);
      for (int loop = 1; loop < piece; loop++) {
        (void)memcpy_s(output_addr + (loop * type_size * kCopyDataSize), data_size, output_addr, data_size);
      }
      char *temp_addr = output_addr + (piece * type_size * kCopyDataSize);
      for (int loop1 = 0; loop1 < remainder; loop1++) {
        (void)memcpy_s(temp_addr + (loop1 * type_size), static_cast<size_t>(type_size), default_value_addr,
                       static_cast<size_t>(type_size));
      }
    }
    return KERNEL_STATUS_OK;
  } else {
    return ParallelSetDefaultValue(ctx, default_value_tensor, output_tensor, output_size);
  }
}
uint32_t SparseToDenseCpuKernel::Compute(CpuKernelContext &ctx) {
  if (ValidParam(ctx) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Valid sparse to dense param error.");
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  Tensor *indices_tensor = ctx.Input(kInput0);
  KERNEL_CHECK_NULLPTR(indices_tensor, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Indices_tensor is null")
  Tensor *shape_tensor = ctx.Input(kInput1);
  KERNEL_CHECK_NULLPTR(shape_tensor, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Shape_tensor is null")
  Tensor *sparse_values = ctx.Input(kInput2);
  KERNEL_CHECK_NULLPTR(sparse_values, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Sparse_values is null")
  Tensor *default_value_tensor = ctx.Input(kInput3);
  KERNEL_CHECK_NULLPTR(default_value_tensor, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID),
                       "Default_value_tensor is null")
  Tensor *output_tensor = ctx.Output(kOutput0);
  KERNEL_CHECK_NULLPTR(output_tensor, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Output_tensor is null")

  auto output_shape = shape_tensor->GetTensorShape();
  std::vector<int64_t> dense_shape;
  std::vector<int64_t> order;
  int64_t output_size = 1;
  size_t output_zero_dim_size = static_cast<size_t>(output_shape->GetDimSize(0));
  for (size_t index = 0; index < output_zero_dim_size; ++index) {
    if (shape_tensor->GetDataType() == DT_INT32) {
      int32_t *temp_dim = reinterpret_cast<int32_t *>(shape_tensor->GetData());
      dense_shape.emplace_back(static_cast<int64_t>(temp_dim[index]));
    } else {
      int64_t *temp_dim = reinterpret_cast<int64_t *>(shape_tensor->GetData());
      dense_shape.emplace_back(temp_dim[index]);
    }
    order.push_back(dense_shape[index]);
    output_size *= dense_shape[index];
  }

  std::iota(order.begin(), order.end(), 0);

  SparseTensor st;
  if (st.CreateSparseTensor(indices_tensor, sparse_values, dense_shape, order) !=
      static_cast<uint32_t>(KERNEL_STATUS_OK)) {
    KERNEL_LOG_ERROR("Create sparse tensor failed.");
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  AttrValue *validate_indices = ctx.GetAttr("validate_indices");
  if (validate_indices == nullptr) {
    KERNEL_LOG_ERROR("Get attr:validate_indices failed.");
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  if (validate_indices->GetBool()) {
    if (st.IndicesValid(ctx) != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
      KERNEL_LOG_ERROR("Indices is valid.");
      return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
  }

  if (SetDefaultValue(ctx, default_value_tensor, output_tensor, output_size) !=
      static_cast<uint32_t>(KERNEL_STATUS_OK)) {
    KERNEL_LOG_ERROR("Sparse_to_dense set default value failed.");
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }

  if (SparseToDense(ctx, st, indices_tensor, output_tensor) != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
    KERNEL_LOG_ERROR("Sparse_to_dense execute failed.");
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

REGISTER_CPU_KERNEL(SPARSETODENSE, SparseToDenseCpuKernel);
}  // namespace aicpu
