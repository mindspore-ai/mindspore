/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "./svd_cpu_kernel_function.h"
using f_complex = std::complex<float>;
using d_complex = std::complex<double>;

template <typename T>
Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> SVDComplex(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix, unsigned int Options) {
  Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> svd(matrix, Options);
  return svd;
}

template Eigen::BDCSVD<Eigen::Matrix<f_complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> SVDComplex<f_complex>(
  const Eigen::Matrix<f_complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix, unsigned int Options);

template Eigen::BDCSVD<Eigen::Matrix<d_complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> SVDComplex<d_complex>(
  const Eigen::Matrix<d_complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix, unsigned int Options);
