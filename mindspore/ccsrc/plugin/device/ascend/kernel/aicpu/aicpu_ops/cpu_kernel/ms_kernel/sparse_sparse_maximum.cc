/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "sparse_sparse_maximum.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kSparseSparseMaximum = "SparseSparseMaximum";
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 6;
constexpr int64_t kIndex0 = 0;
constexpr int64_t kIndex1 = 1;
constexpr int64_t kIndex2 = 2;
constexpr int64_t kIndex3 = 3;
constexpr int64_t kIndex4 = 4;
constexpr int64_t kIndex5 = 5;
bool isMatrix(const std::shared_ptr<aicpu::TensorShape> shape) { return shape->GetDims() == 2; }
bool isVector(const std::shared_ptr<aicpu::TensorShape> shape) { return shape->GetDims() == 1; }
}  // namespace
// 定义命名空间aicpu
namespace aicpu {
uint32_t SparseMaximumCpuKernel::NullptrAndMatVecCheck(CpuKernelContext &ctx, DataBank &databank) {
  databank.a_indices_t = ctx.Input(kIndex0);
  databank.a_values_t = ctx.Input(kIndex1);
  databank.a_shape_t = ctx.Input(kIndex2);
  databank.b_indices_t = ctx.Input(kIndex3);
  databank.b_values_t = ctx.Input(kIndex4);
  databank.b_shape_t = ctx.Input(kIndex5);
  databank.output_indices_t = ctx.Output(kIndex0);
  databank.output_values_t = ctx.Output(kIndex1);
  KERNEL_CHECK_FALSE(
    isMatrix(databank.a_indices_t->GetTensorShape()) && isMatrix(databank.b_indices_t->GetTensorShape()),
    KERNEL_STATUS_PARAM_INVALID,
    "Inputs a_indices and b_indices should be "
    "matrices but received shapes: [%d], [%d]",
    databank.a_indices_t->GetTensorShape()->GetDims(), databank.b_indices_t->GetTensorShape()->GetDims());
  KERNEL_CHECK_FALSE(isVector(databank.a_values_t->GetTensorShape()) && isVector(databank.b_values_t->GetTensorShape()),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Inputs a_values and b_values should be vectors "
                     "but received shapes: [%d] and [%d]",
                     databank.a_values_t->GetTensorShape()->GetDims(),
                     databank.b_values_t->GetTensorShape()->GetDims());
  KERNEL_CHECK_FALSE(isVector(databank.a_shape_t->GetTensorShape()) && isVector(databank.b_shape_t->GetTensorShape()),
                     KERNEL_STATUS_PARAM_INVALID, "Input shapes should be a vector but received shapes [%d] and [%d]",
                     databank.a_shape_t->GetTensorShape()->GetDims(), databank.b_shape_t->GetTensorShape()->GetDims());

  return KERNEL_STATUS_OK;
}

inline static int64_t cmp(const TTypes<int64_t>::Matrix &a_idx, const TTypes<int64_t>::Matrix &b_idx,
                          const int64_t a_row, const int64_t b_row, const int64_t dims) {
  for (int d = 0; d < dims; ++d) {
    const int64_t a = a_idx(a_row, d);
    const int64_t b = b_idx(b_row, d);
    if (a < b) {
      return -1;
    } else if (a > b) {
      return 1;
    }
  }
  return 0;
}

template <typename T>
void SparseMaximumCpuKernel::UnionSparseIndicesAndValues(typename TTypes<int64_t>::Matrix a_indices_mat,
                                                         typename TTypes<T>::Flat a_values, int64_t a_nnz,
                                                         typename TTypes<int64_t>::Matrix b_indices_mat,
                                                         typename TTypes<T>::Flat b_values, int64_t b_nnz,
                                                         int64_t num_dims, std::vector<T> *a_augmented_values,
                                                         std::vector<T> *b_augmented_values,
                                                         std::vector<std::pair<bool, int64_t>> *entries_to_copy) {
  entries_to_copy->reserve(a_nnz + b_nnz);
  a_augmented_values->reserve(a_nnz);
  b_augmented_values->reserve(b_nnz);

  int64_t i = 0, j = 0;
  const T kZero = T(0);
  while (i < a_nnz && j < b_nnz) {
    switch (cmp(a_indices_mat, b_indices_mat, i, j, num_dims)) {
      case -1:
        entries_to_copy->emplace_back(true, i);
        a_augmented_values->push_back(a_values(i));
        b_augmented_values->push_back(kZero);
        ++i;
        break;
      case 0:
        entries_to_copy->emplace_back(true, i);
        a_augmented_values->push_back(a_values(i));
        b_augmented_values->push_back(b_values(j));
        ++i;
        ++j;
        break;
      case 1:
        entries_to_copy->emplace_back(false, j);
        a_augmented_values->push_back(kZero);
        b_augmented_values->push_back(b_values(j));
        ++j;
        break;
    }
  }
  // Handles leftovers; at most one loop runs.
  while (i < a_nnz) {
    entries_to_copy->emplace_back(true, i);
    a_augmented_values->push_back(a_values(i++));
    b_augmented_values->push_back(kZero);
  }
  while (j < b_nnz) {
    entries_to_copy->emplace_back(false, j);
    a_augmented_values->push_back(kZero);
    b_augmented_values->push_back(b_values(j++));
  }
}

template <typename T>
uint32_t SparseMaximumCpuKernel::EigenedSparseMax(DataBank &databank) {
  const int64_t a_nnz = databank.a_indices_t->GetTensorShape()->GetDimSize(0);
  const int64_t b_nnz = databank.b_indices_t->GetTensorShape()->GetDimSize(0);
  EigenTensor a_values_t(databank.a_values_t, databank.a_values_t->GetData());
  const auto a_values = a_values_t.vec<T>();
  EigenTensor b_values_t(databank.b_values_t, databank.b_values_t->GetData());
  const auto b_values = b_values_t.vec<T>();

  EigenTensor a_indices_t(databank.a_indices_t, databank.a_indices_t->GetData());
  const auto a_indices_mat = a_indices_t.matrix<int64_t>();
  EigenTensor b_indices_t(databank.b_indices_t, databank.b_indices_t->GetData());
  const auto b_indices_mat = b_indices_t.matrix<int64_t>();

  const int64_t num_dims = databank.a_indices_t->GetTensorShape()->GetDimSize(1);
  EigenTensor a_shape_t(databank.a_shape_t, databank.a_shape_t->GetData());
  const auto a_shape = a_shape_t.flat<int64_t>();
  EigenTensor b_shape_t(databank.b_shape_t, databank.b_shape_t->GetData());
  const auto b_shape = b_shape_t.flat<int64_t>();

  KERNEL_CHECK_FALSE(a_values.size() == a_nnz && b_values.size() == b_nnz, KERNEL_STATUS_PARAM_INVALID,
                     "Expected [%d] and [%d] non-empty input values, got [%d] and [%d]", a_nnz, b_nnz, a_values.size(),
                     b_values.size());
  KERNEL_CHECK_FALSE(databank.a_shape_t->GetTensorShape()->NumElements() == num_dims, KERNEL_STATUS_PARAM_INVALID,
                     "Second dimension of a_indices and length of "
                     "a_shape must match, got [%d] and [%d]",
                     databank.a_shape_t->GetTensorShape()->NumElements(), num_dims);
  KERNEL_CHECK_FALSE(num_dims > 0, KERNEL_STATUS_PARAM_INVALID, "Tensors must not be empty");
  KERNEL_CHECK_FALSE(
    databank.a_shape_t->GetTensorShape()->NumElements() == databank.b_shape_t->GetTensorShape()->NumElements(),
    KERNEL_STATUS_PARAM_INVALID, "Operands do not have the same ranks; got shapes: [%d] and [%d]",
    databank.a_shape_t->GetTensorShape()->NumElements(), databank.b_shape_t->GetTensorShape()->NumElements());

  for (int i = 0; i < num_dims; ++i) {
    KERNEL_CHECK_FALSE(a_shape(i) == b_shape(i), KERNEL_STATUS_PARAM_INVALID,
                       "Operands' shapes do not match: got [%d] and [%d] for dimension [%d]", a_shape(i), b_shape(i), i)
  }

  std::vector<T> a_augmented_values, b_augmented_values;
  std::vector<std::pair<bool, int64_t>> entries_to_copy;  // from_a?, idx
  UnionSparseIndicesAndValues(a_indices_mat, a_values, a_nnz, b_indices_mat, b_values, b_nnz, num_dims,
                              &a_augmented_values, &b_augmented_values, &entries_to_copy);

  const int64_t sum_nnz = a_augmented_values.size();
  EigenTensor output_values_t(databank.output_values_t, databank.output_values_t->GetData());
  EigenTensor output_indices_t(databank.output_indices_t, databank.output_indices_t->GetData());
  auto output_indices_mat = output_indices_t.matrix<int64_t>();
  for (int64_t i = 0; i < sum_nnz; ++i) {
    const bool from_a = entries_to_copy[i].first;
    const int64_t idx = entries_to_copy[i].second;
    output_indices_mat.chip<0>(i) = from_a ? a_indices_mat.chip<0>(idx) : b_indices_mat.chip<0>(idx);
  }

  using UnalignedTensorMap = Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>, Eigen::Unaligned>;
  auto a_augmented_values_t = UnalignedTensorMap(a_augmented_values.data(), sum_nnz);
  auto b_augmented_values_t = UnalignedTensorMap(b_augmented_values.data(), sum_nnz);

  output_values_t.flat<T>() =
    a_augmented_values_t.binaryExpr(b_augmented_values_t, Eigen::internal::scalar_max_op<T, T>());
  databank.output_indices_t->GetTensorShape()->SetDimSizes({sum_nnz, num_dims});
  databank.output_values_t->GetTensorShape()->SetDimSizes({sum_nnz});
  return KERNEL_STATUS_OK;
}

uint32_t SparseMaximumCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "SparseSparseMaximum check input and output number failed.");

  DataBank databank;
  KERNEL_HANDLE_ERROR(NullptrAndMatVecCheck(ctx, databank), "SparseSparseMaximum check params failed.");

  DataType dt = static_cast<DataType>(databank.output_values_t->GetDataType());
  uint32_t KERNEL_STATUS;
  switch (dt) {
    case DT_INT8:
      KERNEL_STATUS = EigenedSparseMax<int8_t>(databank);
      break;
    case DT_UINT8:
      KERNEL_STATUS = EigenedSparseMax<uint8_t>(databank);
      break;
    case DT_INT16:
      KERNEL_STATUS = EigenedSparseMax<int16_t>(databank);
      break;
    case DT_UINT16:
      KERNEL_STATUS = EigenedSparseMax<uint16_t>(databank);
      break;
    case DT_INT32:
      KERNEL_STATUS = EigenedSparseMax<int32_t>(databank);
      break;
    case DT_INT64:
      KERNEL_STATUS = EigenedSparseMax<int64_t>(databank);
      break;
    case DT_FLOAT16:
      KERNEL_STATUS = EigenedSparseMax<Eigen::half>(databank);
      break;
    case DT_FLOAT:
      KERNEL_STATUS = EigenedSparseMax<float>(databank);
      break;
    case DT_DOUBLE:
      KERNEL_STATUS = EigenedSparseMax<double>(databank);
      break;
    default:
      KERNEL_LOG_ERROR("SparseSparseMaximum can't support this data type [%d].", dt);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (KERNEL_STATUS != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("SparseSparseMaximum failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
// 注册该算子实现
REGISTER_CPU_KERNEL(kSparseSparseMaximum, SparseMaximumCpuKernel);
}  // namespace aicpu
