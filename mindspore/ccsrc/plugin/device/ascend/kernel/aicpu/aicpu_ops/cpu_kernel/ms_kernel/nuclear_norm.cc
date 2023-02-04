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
#include "nuclear_norm.h"
#include <string.h>
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils/kernel_util.h"
#define NoneN 1000
using namespace Eigen;
using namespace std;

namespace {
const char *kNuclearNorm = "NuclearNorm";
const size_t kNuclearNormInputNum = 1;
const size_t kNuclearNormOutputNum = 1;
constexpr int64_t kParallelDataNums = 1 * 1024;
const size_t DIM_SIZE1 = 1;
const size_t DIM_SIZE2 = 2;
const size_t DIM_SIZE3 = 3;
const size_t DIM_SIZE4 = 4;
const size_t DIM_SIZE5 = 5;
const size_t DIM_SIZE6 = 6;
const size_t DIM_SIZE7 = 7;
const size_t DIM_SIZE8 = 8;
const size_t DIM_INDEX0 = 0;
const size_t DIM_INDEX1 = 1;
const size_t DIM_INDEX2 = 2;
}  // namespace

namespace aicpu {
uint32_t NuclearNormCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kNuclearNormInputNum, kNuclearNormOutputNum),
                      "NuclearNorm Check input and output number failed.");
  KERNEL_HANDLE_ERROR(NuclearNormParamCheck(ctx), "NuclearNorm Check params failed.");

  auto data_type = ctx.Input(0)->GetDataType();
  uint32_t res = KERNEL_STATUS_OK;

  switch (data_type) {
    case (DT_FLOAT): {
      res = NuclearNormCompute<float>(ctx);
      break;
    }
    case (DT_DOUBLE): {
      res = NuclearNormCompute<double>(ctx);
      break;
    }
    default:
      KERNEL_LOG_ERROR("NuclearNorm kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t NuclearNormCpuKernel::NuclearNormParamCheck(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *output = ctx.Output(0);
  KERNEL_CHECK_FALSE((input->GetDataType() == output->GetDataType()), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of the input [%s] need be the same as the output [%s]",
                     DTypeStr(input->GetDataType()).c_str(), DTypeStr(output->GetDataType()).c_str());
  const size_t input_dimnum = input->GetTensorShape()->GetDims();
  KERNEL_CHECK_FALSE((input_dimnum >= DIM_SIZE2 && input_dimnum <= DIM_SIZE8), KERNEL_STATUS_PARAM_INVALID,
                     "The range of the dimension of the input tensor should be "
                     "[%lld,%lld], but got input's dimension=%lld",
                     DIM_SIZE2, DIM_SIZE8, input_dimnum);
  AttrValue *dim_ptr = ctx.GetAttr("dim");
  std::vector<int64_t> dim_temp = {0, 1};
  std::vector<int64_t> dim = (dim_ptr == nullptr) ? dim_temp : dim_ptr->GetListInt();
  if (dim_ptr == nullptr) {
    KERNEL_CHECK_FALSE((input_dimnum == DIM_SIZE2), KERNEL_STATUS_PARAM_INVALID,
                       "When Attr dim is none, NuclearNorm expected a tensor with 2 "
                       "dimensions, but got a tensor with [%lld] dimensions instead.",
                       input_dimnum);
  }
  if (dim.size() == 1 && dim[0] == NoneN) {
    dim.clear();
    dim.push_back(0);
    dim.push_back(1);
  }
  KERNEL_CHECK_FALSE((dim.size() == DIM_SIZE2), KERNEL_STATUS_PARAM_INVALID,
                     "Attr dim'size must equal to 2, but got dim's size : [%lld]", dim.size());
  int64_t lower_bound = 0 - input_dimnum;
  int64_t upper_bound = input_dimnum - 1;
  KERNEL_CHECK_FALSE((dim[0] >= lower_bound && dim[0] <= upper_bound), KERNEL_STATUS_PARAM_INVALID,
                     "The range of dim[0] should be [%lld,%lld], but got input dim[0]=%lld", lower_bound, upper_bound,
                     dim[0]);
  KERNEL_CHECK_FALSE((dim[1] >= lower_bound && dim[1] <= upper_bound), KERNEL_STATUS_PARAM_INVALID,
                     "The range of dim[1] should be [%lld,%lld], but got input dim[1]=%lld", lower_bound, upper_bound,
                     dim[1]);
  dim[0] = (dim[0] < 0) ? dim[0] + input_dimnum : dim[0];
  dim[1] = (dim[1] < 0) ? dim[1] + input_dimnum : dim[1];
  KERNEL_CHECK_FALSE((dim[0] != dim[1]), KERNEL_STATUS_PARAM_INVALID,
                     "The values in attr dim point to the same dimension.");
  KERNEL_LOG_DEBUG("NuclearNormCpuKernel[%s], input: size[%llu], output: size[%llu].", ctx.GetOpType().c_str(),
                   input->GetDataSize(), output->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t NuclearNormCpuKernel::NuclearNormCompute(CpuKernelContext &ctx) {
  Tensor *input_ptr = ctx.Input(0);
  auto input_shape = input_ptr->GetTensorShape();
  std::vector<int64_t> input_dims = input_shape->GetDimSizes();
  uint32_t res = KERNEL_STATUS_OK;
  switch (input_dims.size()) {
    case DIM_SIZE2:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE2>(ctx);
      break;
    case DIM_SIZE3:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE3>(ctx);
      break;
    case DIM_SIZE4:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE4>(ctx);
      break;
    case DIM_SIZE5:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE5>(ctx);
      break;
    case DIM_SIZE6:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE6>(ctx);
      break;
    case DIM_SIZE7:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE7>(ctx);
      break;
    case DIM_SIZE8:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE8>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR(
        "Only tensors with ranks between 2 and 8 are currently supported."
        "Tensor rank: [%d]",
        input_dims.size());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, int32_t RANK>
uint32_t NuclearNormCpuKernel::ComputeTensorNuclearNorm(const CpuKernelContext &ctx) {
  Tensor *input_ptr = ctx.Input(0);
  auto input_shape = input_ptr->GetTensorShape();
  void *data_ptr = input_ptr->GetData();
  int64_t value_num_ = input_ptr->NumElements();

  T *input_data_ptr = reinterpret_cast<T *>(data_ptr);
  int64_t total_copy_size = value_num_ * static_cast<int64_t>(sizeof(T));
  Eigen::Tensor<T, 1, Eigen::RowMajor> eigen_tensor(value_num_);
  int memcpy_ret = memcpy_s(&eigen_tensor(0), total_copy_size, input_data_ptr, total_copy_size);

  if (memcpy_ret != 0) {
    KERNEL_LOG_ERROR("memcpy_s error!");
  }
  std::vector<int64_t> input_dims = input_shape->GetDimSizes();
  std::array<Eigen::DenseIndex, RANK> dim_array;
  const int64_t input_dimnum = static_cast<int64_t>(input_shape->GetDims());
  for (int64_t i = 0; i < input_dimnum; i++) {
    dim_array.at(i) = input_dims[i];
  }
  Eigen::Tensor<T, RANK, Eigen::RowMajor> reshaped_tensor = eigen_tensor.reshape(dim_array);

  AttrValue *dim_ptr = ctx.GetAttr("dim");
  std::vector<int64_t> dim_temp = {0, 1};
  std::vector<int64_t> dim = (dim_ptr == nullptr) ? dim_temp : dim_ptr->GetListInt();
  if (dim.size() == 1 && dim[0] == NoneN) {
    dim.clear();
    dim.push_back(0);
    dim.push_back(1);
  }
  dim[0] = (dim[0] < 0) ? dim[0] + input_dimnum : dim[0];
  dim[1] = (dim[1] < 0) ? dim[1] + input_dimnum : dim[1];

  int64_t j = 0;
  for (int64_t i = 0; i < input_dimnum; i++) {
    if (i != dim[0] && i != dim[1]) {
      dim_array.at(j) = i;
      j++;
    }
  }
  dim_array.at(j) = dim[0];
  dim_array.at(j + 1) = dim[1];
  Eigen::Tensor<T, RANK, Eigen::RowMajor> shuffled_tensor = reshaped_tensor.shuffle(dim_array);

  int64_t dimsize0 = input_shape->GetDimSize(dim[0]);
  int64_t dimsize1 = input_shape->GetDimSize(dim[1]);
  int64_t iter_number = value_num_ / (dimsize0 * dimsize1);

  std::array<Eigen::DenseIndex, DIM_SIZE3> dim_array_last;
  dim_array_last.at(DIM_INDEX0) = iter_number;
  dim_array_last.at(DIM_INDEX1) = dimsize0;
  dim_array_last.at(DIM_INDEX2) = dimsize1;
  Eigen::Tensor<T, DIM_SIZE3, Eigen::RowMajor> permuted_tensor = shuffled_tensor.reshape(dim_array_last);

  auto output_data_ptr = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t copy_size = (dimsize0 * dimsize1) * static_cast<int64_t>(sizeof(T));
  if (iter_number <= kParallelDataNums) {
    for (int64_t i = 0; i < iter_number; i++) {
      T *mat = new T[dimsize0 * dimsize1];
      memcpy(mat, &permuted_tensor(i, 0, 0), copy_size);
      T nuclear_norm = matrix_nuclear_norm<T>(mat, dimsize0, dimsize1);
      *(output_data_ptr + i) = nuclear_norm;
    }
  } else {
    uint32_t min_core_num = 1;
    uint64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > static_cast<uint64_t>(iter_number)) {
      max_core_num = static_cast<uint64_t>(iter_number);
    }

    auto shared_nuclear_norm = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        T *mat = new T[dimsize0 * dimsize1];
        memcpy(mat, &permuted_tensor(i, 0, 0), copy_size);
        T nuclear_norm = matrix_nuclear_norm<T>(mat, dimsize0, dimsize1);
        *(output_data_ptr + i) = nuclear_norm;
      }
    };
    if (max_core_num != 0) {
      KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, static_cast<uint64_t>(iter_number),
                                    static_cast<uint64_t>(iter_number) / max_core_num, shared_nuclear_norm),
        "NuclearNorm Compute failed.");
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
std::vector<std::vector<T>> NuclearNormCpuKernel::matrix_multiply(std::vector<std::vector<T>> const arrL,
                                                                  std::vector<std::vector<T>> const arrR) {
  size_t rowL = arrL.size();
  size_t colL = arrL[0].size();
  size_t colR = arrR[0].size();

  std::vector<std::vector<T>> res(rowL);
  for (size_t i = 0; i < res.size(); i++) {
    res[i].resize(colR);
  }

  for (size_t i = 0; i < rowL; i++) {
    for (size_t j = 0; j < colR; j++) {
      for (size_t k = 0; k < colL; k++) {
        res[i][j] += arrL[i][k] * arrR[k][j];
      }
    }
  }

  return res;
}

template <typename T>
std::vector<std::vector<T>> NuclearNormCpuKernel::transpose(std::vector<std::vector<T>> const arr) {
  size_t row = arr.size();
  size_t col = arr[0].size();

  std::vector<std::vector<T>> trans(col);
  for (size_t i = 0; i < col; i++) {
    trans[i].resize(row);
  }

  for (size_t i = 0; i < col; i++) {
    for (size_t j = 0; j < row; j++) {
      trans[i][j] = arr[j][i];
    }
  }
  return trans;
}

template <typename T>
std::vector<size_t> NuclearNormCpuKernel::argsort(const std::vector<T> &array) {
  const size_t array_len(array.size());
  std::vector<size_t> array_index(array_len, 0);
  for (size_t i = 0; i < array_len; ++i) array_index[i] = i;

  sort(array_index.begin(), array_index.end(),
       [&array](size_t pos1, size_t pos2) { return (array[pos1] > array[pos2]); });

  return array_index;
}

template <typename T>
void NuclearNormCpuKernel::get_row_col(std::vector<std::vector<T>> arr, T *max, size_t *row, size_t *col) {
  size_t n = arr.size();
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      if (i != j && fabs(arr[i][j]) > *max) {
        *max = fabs(arr[i][j]);
        *row = i;
        *col = j;
      }
    }
  }
}

template <typename T>
void NuclearNormCpuKernel::svd(std::vector<std::vector<T>> arr, std::vector<std::vector<T>> &E, std::vector<T> &e) {
  size_t n = arr.size();
  size_t row = 0;
  size_t col = 0;
  size_t iter_max_num = 10000;
  size_t iter_num = 0;
  T eps = 1e-40;
  T max = eps;
  T dot5 = 0.5;

  E.resize(n);
  e.resize(n);
  for (size_t i = 0; i < n; i++) {
    E[i].resize(n, 0);
    E[i][i] = 1;
  }

  while (iter_num < iter_max_num && max >= eps) {
    max = fabs(arr[0][1]);
    row = 0;
    col = 1;

    get_row_col<T>(arr, &max, &row, &col);
    T theta = dot5 * atan2(-2 * arr[row][col], -(arr[row][row] - arr[col][col]));

    T aii = arr[row][row];
    T ajj = arr[col][col];
    T aij = arr[row][col];
    T sin_theta = sin(theta);
    T cos_theta = cos(theta);
    T sin_2theta = sin(2 * theta);
    T cos_2theta = cos(2 * theta);
    arr[row][row] = aii * cos_theta * cos_theta + ajj * sin_theta * sin_theta + aij * sin_2theta;
    arr[col][col] = aii * sin_theta * sin_theta + ajj * cos_theta * cos_theta - aij * sin_2theta;
    arr[row][col] = dot5 * (ajj - aii) * sin_2theta + aij * cos_2theta;
    arr[col][row] = arr[row][col];
    for (size_t k = 0; k < n; k++) {
      if (k != row && k != col) {
        T arowk = arr[row][k];
        T acolk = arr[col][k];
        arr[row][k] = arowk * cos_theta + acolk * sin_theta;
        arr[k][row] = arr[row][k];
        arr[col][k] = acolk * cos_theta - arowk * sin_theta;
        arr[k][col] = arr[col][k];
      }
    }

    T Eki;
    T Ekj;
    for (size_t k = 0; k < n; k++) {
      Eki = E[k][row];
      Ekj = E[k][col];
      E[k][row] = Eki * cos_theta + Ekj * sin_theta;
      E[k][col] = Ekj * cos_theta - Eki * sin_theta;
    }
    iter_num++;
  }

  for (size_t i = 0; i < n; i++) {
    e[i] = arr[i][i];
  }

  std::vector<size_t> sort_index;
  sort_index = argsort<T>(e);

  std::vector<std::vector<T>> E_sorted(n);
  for (size_t i = 0; i < n; i++) {
    E_sorted[i].resize(n);
  }
  std::vector<T> e_sorted(n);
  for (size_t i = 0; i < n; i++) {
    e_sorted[i] = e[sort_index[i]];
    for (size_t j = 0; j < n; j++) {
      E_sorted[i][j] = E[i][sort_index[j]];
    }
  }
  E = E_sorted;
  e = e_sorted;
}

template <typename T>
T NuclearNormCpuKernel::matrix_nuclear_norm(T *mat, size_t dim0, size_t dim1) {
  if (dim1 == DIM_SIZE1) {
    T nuclear_norm = 0.0;
    T temp = 0.0;
    for (size_t j = 0; j < dim0; j++) {
      temp = mat[j];
      temp = temp * temp;
      nuclear_norm += temp;
    }
    nuclear_norm = sqrt(nuclear_norm);
    return nuclear_norm;
  }
  std::vector<std::vector<double>> arr(dim0);
  size_t S_dim_size = dim0 < dim1 ? dim0 : dim1;
  for (size_t i = 0; i < arr.size(); i++) {
    arr[i].resize(dim1);
  }
  for (size_t i = 0; i < dim0; i++) {
    for (size_t j = 0; j < dim1; j++) {
      arr[i][j] = mat[i * dim1 + j];
    }
  }

  std::vector<std::vector<double>> ATA;
  std::vector<std::vector<double>> E;
  std::vector<double> e;

  ATA = matrix_multiply<double>(transpose(arr), arr);
  svd<double>(ATA, E, e);

  double nuclear_norm = 0.0;
  for (size_t i = DIM_INDEX0; i < S_dim_size; i++) {
    if (e[i] > 0) {
      nuclear_norm += sqrt(e[i]);
    }
  }

  return nuclear_norm;
}
REGISTER_CPU_KERNEL(kNuclearNorm, NuclearNormCpuKernel);
}  // namespace aicpu
