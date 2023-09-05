/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_EIGEN_UTIL_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_EIGEN_UTIL_H_

#include <Eigen/Dense>
#include <vector>
#include "ir/tensor.h"
#include "include/api/types.h"
#include "src/tensor.h"

namespace mindspore::lite::quant {
template <typename T>
int CalculateHessianMatrix(void *x, void *y, int64_t m, int64_t n, bool transpose = false) {
  CHECK_NULL_RETURN(x);
  CHECK_NULL_RETURN(y);
  auto input_x = reinterpret_cast<T *>(x);
  auto output_y = reinterpret_cast<T *>(y);
  using MatrixXd = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<MatrixXd> A(input_x, m, n);
  MatrixXd result;
  int32_t deep = m;
  // X * W
  if (transpose) {
    result = 2 * A.transpose() * A;
    deep = n;
  } else {
    result = 2 * A * A.transpose();
  }
  for (int32_t i = 0; i < deep; i++) {
    for (int32_t j = 0; j < deep; j++) {
      *(output_y + i * deep + j) = result(i, j);
    }
  }
  return RET_OK;
}

template <typename T>
int CalculateCholesky(const void *x, void *y, std::vector<int64_t> dims, bool upper = false) {
  CHECK_NULL_RETURN(x);
  CHECK_NULL_RETURN(y);
  auto input_x = reinterpret_cast<const T *>(x);
  auto output_y = reinterpret_cast<T *>(y);
  int64_t dims_num = dims.size();
  int64_t m = dims.at(dims_num - 2);
  int64_t count = 1;
  int64_t no_batch = 2;

  if (dims_num > no_batch) {
    for (int64_t i = 0; i < dims_num - no_batch; i++) {
      count *= dims[i];
    }
  }

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(m, m);

  for (int64_t k = 0; k < count; k++) {
    for (int64_t i = 0; i < m * m; i++) {
      A.data()[i] = input_x[k * m * m + i];
    }

    Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_llt(A);

    if (!A.isApprox(A.transpose()) || A_llt.info() == Eigen::NumericalIssue) {
      MS_LOG(ERROR) << "There exists non semi-positive definitie matrix!";
      return RET_ERROR;
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> L = A_llt.matrixL();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U = A_llt.matrixU();

    if (!upper) {
      for (int64_t i = 0; i < m * m; i++) {
        output_y[k * m * m + i] = L.data()[i];
      }
    } else {
      for (int64_t i = 0; i < m * m; i++) {
        output_y[k * m * m + i] = U.data()[i];
      }
    }
  }
  return RET_OK;
}

template <typename T>
int CalculateCholeskyInverse(void *x, void *y, int64_t n, bool upper = false) {
  CHECK_NULL_RETURN(x);
  CHECK_NULL_RETURN(y);
  auto input_x = reinterpret_cast<T *>(x);
  auto output_y = reinterpret_cast<T *>(y);
  using MatrixXd = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<MatrixXd> A(input_x, n, n);
  MatrixXd result;
  if (upper) {
    result = (A.transpose() * A).inverse();
  } else {
    result = (A * A.transpose()).inverse();
  }
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < n; j++) {
      *(output_y + i * n + j) = result(i, j);
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_EIGEN_UTIL_H_
