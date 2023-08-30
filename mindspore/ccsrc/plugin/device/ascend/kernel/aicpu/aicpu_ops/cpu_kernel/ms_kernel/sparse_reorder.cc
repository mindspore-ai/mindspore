/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "cpu_kernel/ms_kernel/sparse_reorder.h"

namespace {
constexpr uint32_t kSparseReorderInputNum = 3;
constexpr uint32_t kSparseReorderOutputNum = 2;
constexpr const char *kSparseReorder = "SparseReorder";
}  // namespace

namespace aicpu {
uint32_t SparseReorderCpuKernel::SparseReorder(const CpuKernelContext &ctx, SparseTensor &st, Tensor *y_indices,
                                               Tensor *y_values) {
  DataType dt = static_cast<DataType>(y_values->GetDataType());
  switch (dt) {
    case DT_INT8:
      return EigenSparseReorder<int8_t>(ctx, st, y_indices, y_values);
    case DT_UINT8:
      return EigenSparseReorder<uint8_t>(ctx, st, y_indices, y_values);
    case DT_INT16:
      return EigenSparseReorder<int16_t>(ctx, st, y_indices, y_values);
    case DT_UINT16:
      return EigenSparseReorder<uint16_t>(ctx, st, y_indices, y_values);
    case DT_INT32:
      return EigenSparseReorder<int32_t>(ctx, st, y_indices, y_values);
    case DT_INT64:
      return EigenSparseReorder<int64_t>(ctx, st, y_indices, y_values);
    case DT_FLOAT16:
      return EigenSparseReorder<Eigen::half>(ctx, st, y_indices, y_values);
    case DT_FLOAT:
      return EigenSparseReorder<float>(ctx, st, y_indices, y_values);
    case DT_BOOL:
      return EigenSparseReorder<bool>(ctx, st, y_indices, y_values);
    case DT_DOUBLE:
      return EigenSparseReorder<double>(ctx, st, y_indices, y_values);
    case DT_COMPLEX64:
      return EigenSparseReorder<std::complex<std::float_t>>(ctx, st, y_indices, y_values);
    case DT_COMPLEX128:
      return EigenSparseReorder<std::complex<std::double_t>>(ctx, st, y_indices, y_values);
    case DT_STRING:
      return EigenSparseReorder<std::string>(ctx, st, y_indices, y_values);
    default:
      KERNEL_LOG_ERROR("Sparse reorder can't support this data type [%d].", DTypeStr(dt).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t SparseReorderCpuKernel::ValidParam(const CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kSparseReorderInputNum, kSparseReorderOutputNum), "[%s] check params failed.",
                      kSparseReorder);
  // valid input and output nullptr
  Tensor *indices_tensor = ctx.Input(0);
  Tensor *values_tensor = ctx.Input(1);
  Tensor *shape_tensor = ctx.Input(2);
  Tensor *y_indices_tensor = ctx.Output(0);
  Tensor *y_values_tensor = ctx.Output(1);
  // valid shape nullptr
  auto indices_shape = indices_tensor->GetTensorShape();
  auto values_shape = values_tensor->GetTensorShape();
  auto input_shape = shape_tensor->GetTensorShape();
  auto y_indices_shape = y_indices_tensor->GetTensorShape();
  auto y_values_shape = y_values_tensor->GetTensorShape();
  // get dims
  const int64_t elems_num = indices_shape->GetDimSize(0);
  const int64_t dims_num = indices_shape->GetDimSize(1);
  const int64_t y_elems_num = y_indices_shape->GetDimSize(0);
  const int64_t y_dims_num = y_indices_shape->GetDimSize(1);
  const int64_t values_num = values_shape->GetDimSize(0);
  const int64_t y_values_num = y_values_shape->GetDimSize(0);
  // sparse_indices 2D
  if (indices_shape->GetDims() != 2) {
    KERNEL_LOG_ERROR(
      "Sparse_indices should be a 2D matrix, got dim."
      "size [%d].",
      indices_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (y_indices_shape->GetDims() != 2) {
    KERNEL_LOG_ERROR(
      "Sparse_y_indices should be a 2D matrix, got dim."
      "size [%d].",
      y_indices_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // values_tensor
  int32_t values_dims_size = values_shape->GetDims();
  if (values_dims_size != 1) {
    KERNEL_LOG_ERROR("Values_shape should be a vector, got dim size [%d].", values_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (values_tensor->NumElements() != elems_num) {
    KERNEL_LOG_ERROR(
      "Values_shape has incorrect number of elements [%lld], should be "
      "[%lld]",
      values_tensor->NumElements(), elems_num);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // input_shape
  if (input_shape->GetDims() != 1) {
    KERNEL_LOG_ERROR("Input_shape should be a vector, got dim size [%d].", input_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape_tensor->NumElements() != dims_num) {
    KERNEL_LOG_ERROR(
      "Input_shape has incorrect number of elements [%lld], should be "
      "[%lld] ",
      shape_tensor->NumElements(), dims_num);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // input size == output size
  if (elems_num != y_elems_num || dims_num != y_dims_num || values_num != y_values_num) {
    KERNEL_LOG_ERROR("The output size needs to be the same as the input size");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // get type
  int32_t indiceType = indices_tensor->GetDataType();
  int32_t inShapeType = shape_tensor->GetDataType();
  int32_t yIndiceType = y_indices_tensor->GetDataType();
  int32_t valuesType = values_tensor->GetDataType();
  int32_t yvaluesType = y_values_tensor->GetDataType();
  bool validIndiceType = (indiceType != DT_INT64);
  bool validShapeType = (inShapeType != DT_INT64);
  bool validYIndiceType = (yIndiceType != DT_INT64);
  bool sameIndicesType = (indiceType != yIndiceType);
  bool sameValuesType = (valuesType != yvaluesType);
  // valid data type
  if (validShapeType || validIndiceType || validYIndiceType) {
    KERNEL_LOG_ERROR("Dtype of indices and shape should be int64.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // same data type
  if (sameIndicesType || sameValuesType) {
    KERNEL_LOG_ERROR("The output dtype needs to be the same as the input dtype.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SparseReorderCpuKernel::Compute(CpuKernelContext &ctx) {
  if (ValidParam(ctx) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Valid sparse reorder param error.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *indices_tensor = ctx.Input(0);
  Tensor *values_tensor = ctx.Input(1);
  Tensor *shape_tensor = ctx.Input(2);
  Tensor *y_indices_tensor = ctx.Output(0);
  Tensor *y_values_tensor = ctx.Output(1);
  auto output_shape = shape_tensor->GetTensorShape();
  std::vector<int64_t> dense_shape;
  std::vector<int64_t> order;
  int64_t *temp_dim = reinterpret_cast<int64_t *>(shape_tensor->GetData());
  for (int32_t index = 0; index < output_shape->GetDimSize(0); ++index) {
    dense_shape.emplace_back(temp_dim[index]);
    order.push_back(dense_shape[index]);
  }
  std::iota(order.begin(), order.end(), 0);
  SparseTensor st;
  if (st.CreateSparseTensor(indices_tensor, values_tensor, dense_shape, order) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Create sparse tensor failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (SparseReorder(ctx, st, y_indices_tensor, y_values_tensor) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Sparse Reorder failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSparseReorder, SparseReorderCpuKernel);
}  // namespace aicpu
