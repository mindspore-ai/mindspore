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

#include "sparse_slice.h"

#include <unistd.h>
#include <complex>
#include <string>
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "utils/sparse_tensor.h"

namespace {
const uint32_t kOutputNum = 3;
const uint32_t kInputNum = 5;
const char *kSparseSlice = "SparseSlice";
}  // namespace

namespace aicpu {
template <typename T>
using ArraySlice = std::vector<T>;

template <typename T>
void Slice(Tensor *output_indices, Tensor *output_values, Tensor *output_dense_shape, SparseTensor *input_tensor,
           const aicpu::ArraySlice<int64_t> &start, const aicpu::ArraySlice<int64_t> &size) {
  auto output_shape = CpuKernelUtils::CreateTensorShape();
  output_shape->SetDimSizes(input_tensor->shape());
  auto output_shape_num_dims = CpuKernelUtils::CreateTensorShape();
  output_shape_num_dims->SetDimSizes(input_tensor->shape());
  const int dims = input_tensor->dims();
  std::vector<int64_t> dimsVec(dims, 0);

  for (int dim = 0; dim < dims; dim++) {
    // Determine the size of the result; if the selected slice goes beyond the
    // input boundary, the result will correspond to the size of the overlap
    // between the input and the selected slice.
    const auto input_size = output_shape->GetDimSize(dim);
    const int64_t start_index = start[dim];
    const int64_t slice_size = size[dim];

    if (start_index + slice_size < input_size) {
      dimsVec[dim] = slice_size;
    } else if (start_index < input_size) {
      dimsVec[dim] = input_size - start_index;
    } else {
      dimsVec[dim] = 0;
    }
  }

  output_shape->SetDimSizes(dimsVec);
  auto input_indices_t = input_tensor->indices().get()->matrix<int64_t>();
  auto input_values_t = input_tensor->values().get()->vec<T>();

  // Find the number of indices that fall inside start and size.
  int count = 0;
  int dim_size = input_tensor->indices()->GetTensor()->GetTensorShape()->GetDimSize(0);

  for (int i = 0; i < dim_size; i++) {
    // The following will check to see if an input is within the
    // range specified by start and size.
    // The for loop below iterates through all dimensions. In case
    // the index falls outside of the start and size at any dimension,
    // it will be considered as a "no hit" (hit = false). In this
    // case, it will not be counted as the index that fall inside
    // the range specified by start and size.

    bool hit = true;
    for (int dim = 0; dim < dims; dim++) {
      if (!(start[dim] <= input_indices_t(i, dim) && input_indices_t(i, dim) < start[dim] + size[dim])) {
        hit = false;
        break;
      }
    }
    if (!hit) {
      continue;
    }
    count++;
  }

  auto eigen_tensor_indices = EigenTensor(output_indices, output_indices->GetData());
  auto eigen_tensor_values = EigenTensor(output_values, output_values->GetData());
  auto eigen_tensor_shape = EigenTensor(output_dense_shape, output_dense_shape->GetData());
  auto output_values_t = eigen_tensor_values.vec<T>();
  auto output_indices_t = eigen_tensor_indices.matrix<int64_t>();
  auto output_shape_t = eigen_tensor_shape.vec<int64_t>();

  // Obtain the output indices that fall inside start and size.
  for (int dim = 0; dim < output_dense_shape->NumElements(); ++dim) {
    const auto input_size = output_shape->GetDimSize(dim);
    output_shape_t(dim) = input_size;
  }

  int index = 0;
  for (int i = 0; i < dim_size && index < count; i++) {
    // The logic here is similar as the above except that the above
    // only count the number of indices while here we actually generate
    // the output.

    bool hit = true;
    for (int dim = 0; dim < dims; dim++) {
      if (!(start[dim] <= input_indices_t(i, dim) && input_indices_t(i, dim) < start[dim] + size[dim])) {
        hit = false;
        break;
      }
    }
    if (!hit) {
      continue;
    }

    output_values_t(index) = input_values_t(i);

    for (int64_t dim = 0; dim < dims; dim++) {
      output_indices_t(index, dim) = input_indices_t(i, dim) - start[dim];
    }
    index++;
  }
  const int num_dims = dims;
  const int64_t y_nnz = index;

  std::vector<int64_t> indices_dims = {y_nnz, num_dims};
  auto output_indices_shape = output_indices->GetTensorShape();
  output_indices_shape->SetDimSizes(indices_dims);
  output_indices->SetTensorShape(output_indices_shape.get());

  std::vector<int64_t> values_dims = {y_nnz};
  auto output_values_shape = output_values->GetTensorShape();
  output_values_shape->SetDimSizes(values_dims);
  output_values->SetTensorShape(output_values_shape.get());
}

std::uint32_t SparseSliceCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *indices = ctx.Input(0);
  Tensor *values = ctx.Input(1);
  Tensor *shape = ctx.Input(2);
  Tensor *start = ctx.Input(3);
  Tensor *size = ctx.Input(4);

  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "sparseslice check input and output number failed.");
  KERNEL_HANDLE_ERROR(SparseSliceParamCheck(indices, values, shape, start, size), "sparseslice check params failed.");

  const int input_dims = shape->NumElements();
  auto shape_shape = shape->GetTensorShape();
  std::vector<int64_t> dense_shape;
  std::vector<int64_t> order;
  int64_t output_size = 1;
  for (int32_t index = 0; index < shape_shape->GetDimSize(0); ++index) {
    if (shape->GetDataType() == DT_INT32) {
      int32_t *temp_dim = static_cast<int32_t *>(shape->GetData());
      dense_shape.emplace_back(static_cast<int64_t>(temp_dim[index]));
    } else {
      int64_t *temp_dim = static_cast<int64_t *>(shape->GetData());
      dense_shape.emplace_back(temp_dim[index]);
    }
    order.push_back(dense_shape[index]);
    output_size *= dense_shape[index];
  }

  std::iota(order.begin(), order.end(), 0);

  SparseTensor st;
  if (st.CreateSparseTensor(indices, values, dense_shape, order) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Create sparse tensor failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  aicpu::ArraySlice<int64_t> slice_start(input_dims, 0);
  aicpu::ArraySlice<int64_t> slice_size(input_dims, 0);

  auto start_val = static_cast<int64_t *>(start->GetData());
  auto size_val = static_cast<int64_t *>(size->GetData());
  for (int64_t i = 0; i < input_dims; i++) {
    slice_start[i] = *(start_val + i);
  }

  for (int64_t i = 0; i < input_dims; i++) {
    slice_size[i] = *(size_val + i);
  }

  Tensor *output_indices = ctx.Output(0);
  Tensor *output_values = ctx.Output(1);
  Tensor *output_dense_shape = ctx.Output(2);

  DataType values_data_type = ctx.Input(1)->GetDataType();
  KERNEL_LOG_DEBUG("%s op input[a] data type is [%s].", kSparseSlice, DTypeStr(values_data_type).c_str());
  switch (values_data_type) {
    case DT_INT64:
      Slice<int64_t>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    case DT_INT32:
      Slice<int32_t>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    case DT_UINT16:
      Slice<uint16_t>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    case DT_INT16:
      Slice<int16_t>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    case DT_UINT8:
      Slice<uint8_t>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    case DT_INT8:
      Slice<int8_t>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    case DT_FLOAT16:
      Slice<Eigen::half>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    case DT_FLOAT:
      Slice<float>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    case DT_DOUBLE:
      Slice<double>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    case DT_COMPLEX64:
      Slice<std::complex<float>>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    case DT_COMPLEX128:
      Slice<std::complex<double>>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    case DT_BOOL:
      Slice<bool>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    case DT_STRING:
      Slice<std::string>(output_indices, output_values, output_dense_shape, &st, slice_start, slice_size);
      break;
    default:
      KERNEL_LOG_ERROR("SparseSlice kernel data type [%s] not support.", DTypeStr(values_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SparseSliceCpuKernel::SparseSliceParamCheck(Tensor *indices, Tensor *values, Tensor *shape, Tensor *start,
                                                     Tensor *size) {
  auto indices_shape = indices->GetTensorShape();
  KERNEL_CHECK_FALSE((IsMatrix(indices_shape->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                     "Input indeices must be a matrix.");

  auto values_shape = values->GetTensorShape();
  KERNEL_CHECK_FALSE((IsVector(values_shape->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                     "Input values must be a vector.");

  auto shape_shape = shape->GetTensorShape();
  std::vector<int64_t> shape_shape_vec;
  int64_t *shape_vec = static_cast<int64_t *>(shape->GetData());
  shape_shape_vec.push_back(*(shape_vec));
  shape_shape_vec.push_back(*(shape_vec + 1));
  KERNEL_CHECK_FALSE((IsVector(shape_shape->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                     " Input shape must be a vector.");

  auto start_shape = start->GetTensorShape();
  KERNEL_CHECK_FALSE((IsVector(start_shape->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                     "Input start must be a vector.");

  auto size_shape = size->GetTensorShape();
  KERNEL_CHECK_FALSE((IsVector(size_shape->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID, "Input size must be a vector");

  const int input_dims = shape->NumElements();
  KERNEL_CHECK_FALSE((input_dims == start->NumElements()), KERNEL_STATUS_PARAM_INVALID,
                     "Expected start to be a vector of length [%s]", input_dims, "but get length [%s]",
                     start->NumElements());

  KERNEL_CHECK_FALSE((input_dims == size->NumElements()), KERNEL_STATUS_PARAM_INVALID,
                     "Expected start to be a vector of length [%s]", input_dims, "but get length [%s]",
                     size->NumElements());
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSparseSlice, SparseSliceCpuKernel);
}  // namespace aicpu