/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_EIGEN_COMMON_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_EIGEN_COMMON_UTILS_H_
#include <algorithm>
#include <functional>
#include <vector>
#include "kernel/common_utils.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"

#ifdef _WIN32
#undef ERROR
#endif

namespace mindspore {
namespace kernel {
using Eigen::ColMajor;
using Eigen::Dynamic;
using Eigen::Lower;
using Eigen::Map;
using Eigen::MatrixBase;
using Eigen::RowMajor;
using Eigen::UnitLower;
using Eigen::UnitUpper;
using Eigen::Upper;
template <typename T, int Major>
using Matrix = Eigen::Matrix<T, Dynamic, Dynamic, Major>;
template <typename T>
using MatrixSquare = Eigen::Matrix<T, Dynamic, Dynamic, RowMajor>;
template <typename T>
using ComplexMatrixSquare = Eigen::Matrix<std::complex<T>, Dynamic, Dynamic, RowMajor>;

template <typename T, int NDIMS = kDim1, typename IndexType = Eigen::DenseIndex>
struct TTypes {
  // Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, static_cast<Eigen::StorageOptions>(Eigen::RowMajor), IndexType>,
                           static_cast<Eigen::AlignmentType>(Eigen::Aligned)>
    Tensor;
  typedef Eigen::TensorMap<
    Eigen::Tensor<const T, NDIMS, static_cast<Eigen::StorageOptions>(Eigen::RowMajor), IndexType>,
    static_cast<Eigen::AlignmentType>(Eigen::Aligned)>
    ConstTensor;

  // Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, kDim1, static_cast<Eigen::StorageOptions>(Eigen::RowMajor), IndexType>,
                           static_cast<Eigen::AlignmentType>(Eigen::Aligned)>
    Flat;
  typedef Eigen::TensorMap<
    Eigen::Tensor<const T, kDim1, static_cast<Eigen::StorageOptions>(Eigen::RowMajor), IndexType>,
    static_cast<Eigen::AlignmentType>(Eigen::Aligned)>
    ConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, kDim1, static_cast<Eigen::StorageOptions>(Eigen::RowMajor), IndexType>,
                           static_cast<Eigen::AlignmentType>(Eigen::Aligned)>
    Vec;
  typedef Eigen::TensorMap<
    Eigen::Tensor<const T, kDim1, static_cast<Eigen::StorageOptions>(Eigen::RowMajor), IndexType>,
    static_cast<Eigen::AlignmentType>(Eigen::Aligned)>
    ConstVec;

  // Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, kDim2, static_cast<Eigen::StorageOptions>(Eigen::RowMajor), IndexType>,
                           static_cast<Eigen::AlignmentType>(Eigen::Aligned)>
    Matrix;
  typedef Eigen::TensorMap<
    Eigen::Tensor<const T, kDim2, static_cast<Eigen::StorageOptions>(Eigen::RowMajor), IndexType>,
    static_cast<Eigen::AlignmentType>(Eigen::Aligned)>
    ConstMatrix;
};

class EigenTensor {
 public:
  EigenTensor() = delete;
  EigenTensor(const ShapeVector &shape, void *data_ptr) : tensor_shape(shape), tensor_data_ptr(data_ptr) {}
  EigenTensor(std::vector<size_t> &shape, void *data_ptr) : tensor_data_ptr(data_ptr) {
    for (size_t dim : shape) {
      (void)tensor_shape.emplace_back(static_cast<int64_t>(dim));
    }
  }
  ~EigenTensor() = default;

  /*
   * Eigen vec
   * @return Eigen vec
   */
  template <typename T>
  typename TTypes<T>::Vec vec() {
    return tensor<T, 1>();
  }

  /*
   * Eigen matrix
   * @return Eigen matrix
   */
  template <typename T>
  typename TTypes<T>::Matrix matrix() {
    return tensor<T, kDim2>();
  }

  /*
   * Eigen ConstMatrix
   * @return Eigen ConstMatrix
   */
  template <typename T>
  typename TTypes<T>::ConstMatrix matrix() const {
    return tensor<T, kDim2>();
  }

  /*
   * Eigen tensor
   * @return Eigen tensor
   */
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor tensor() {
    return typename TTypes<T, NDIMS>::Tensor(reinterpret_cast<T *>(tensor_data_ptr), AsEigenDSizes<NDIMS>());
  }

  /*
   * Eigen ConstTensor
   * @return Eigen ConstTensor
   */
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor tensor() const {
    return typename TTypes<T, NDIMS>::ConstTensor(reinterpret_cast<const T *>(tensor_data_ptr), AsEigenDSizes<NDIMS>());
  }

  /*
   * Eigen Flat
   * @return Eigen Flat
   */
  template <typename T>
  typename TTypes<T>::Flat flat() {
    return typename TTypes<T>::Flat(
      reinterpret_cast<T *>(tensor_data_ptr),
      {std::accumulate(tensor_shape.begin(), tensor_shape.end(), 1, std::multiplies<int64_t>())});
  }

  /*
   * which case we pad the rest of the sizes with 1.
   * @return Eigen::DSizes: pad the rest of the sizes with 1
   */
  template <int NDIMS, typename IndexType>
  Eigen::DSizes<IndexType, NDIMS> AsEigenDSizesWithPadding() const {
    Eigen::DSizes<IndexType, NDIMS> dsizes;
    for (size_t d = 0; d < tensor_shape.size(); d++) {
      dsizes[d] = static_cast<IndexType>(tensor_shape[d]);
    }
    for (size_t d = tensor_shape.size(); d < NDIMS; d++) {
      dsizes[d] = 1;
    }
    return dsizes;
  }

  /*
   * Fill `*dsizes` from `*this`
   * @return Eigen::DSizes: pad the rest of the sizes with 1
   */
  template <int NDIMS, typename IndexType = Eigen::DenseIndex>
  Eigen::DSizes<IndexType, NDIMS> AsEigenDSizes() const {
    return AsEigenDSizesWithPadding<NDIMS, IndexType>();
  }

 private:
  ShapeVector tensor_shape;
  void *tensor_data_ptr;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_EIGEN_COMMON_UTILS_H_
