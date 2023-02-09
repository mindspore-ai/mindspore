/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#ifndef AICPU_EIGENTENSOR_H
#define AICPU_EIGENTENSOR_H

#include "cpu_tensor.h"
#include "kernel_log.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
// Helper to define Tensor types given that the scalar is of type T.
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct TTypes {
  // Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned> Tensor;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned> ConstTensor;

  // Unaligned Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType> > UnalignedTensor;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType> > UnalignedConstTensor;

  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, int>, Eigen::Aligned> Tensor32Bit;

  // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>, Eigen::Aligned>
    Scalar;
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>, Eigen::Aligned>
    ConstScalar;

  // Unaligned Scalar tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType> > UnalignedScalar;
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>, Eigen::RowMajor, IndexType> >
    UnalignedConstScalar;

  // Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned> Flat;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned> ConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned> Vec;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned> ConstVec;

  // Unaligned Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType> > UnalignedFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType> > UnalignedConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType> > UnalignedVec;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType> > UnalignedConstVec;

  // Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned> Matrix;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned> ConstMatrix;

  // Unaligned Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType> > UnalignedMatrix;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType> > UnalignedConstMatrix;
};
}  // namespace aicpu

namespace aicpu {

namespace {
using ShapeVector = std::vector<int64_t>;
}

class EigenTensor {
 public:
  EigenTensor() = delete;
  EigenTensor(Tensor *tensor, void *data) : tensor_(tensor), tensor_data_(data) {
    tensor_shape_ = tensor->GetTensorShape()->GetDimSizes();
  }
  EigenTensor(ShapeVector &shape, void *data_ptr) : tensor_shape_(shape), tensor_data_(data_ptr) {}
  ~EigenTensor() = default;

  /*
   * Get tensor
   * @return succ: tensor, error : nullptr
   */
  const Tensor *GetTensor() const;

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
    return tensor<T, 2>();
  }

  /*
   * Eigen ConstMatrix
   * @return Eigen ConstMatrix
   */
  template <typename T>
  typename TTypes<T>::ConstMatrix matrix() const {
    return tensor<T, 2>();
  }

  /*
   * Eigen tensor
   * @return Eigen tensor
   */
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor tensor() {
    return typename TTypes<T, NDIMS>::Tensor(reinterpret_cast<T *>(tensor_data_), AsEigenDSizes<NDIMS>());
  }

  /*
   * Eigen ConstTensor
   * @return Eigen ConstTensor
   */
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor tensor() const {
    return typename TTypes<T, NDIMS>::ConstTensor(reinterpret_cast<const T *>(tensor_data_), AsEigenDSizes<NDIMS>());
  }

  /*
   * Eigen Flat
   * @return Eigen Flat
   */
  template <typename T>
  typename TTypes<T>::Flat flat() {
    return typename TTypes<T>::Flat(
      reinterpret_cast<T *>(tensor_data_),
      {std::accumulate(tensor_shape_.begin(), tensor_shape_.end(), 1, std::multiplies<int64_t>())});
  }

  /*
   * which case we pad the rest of the sizes with 1.
   * @return Eigen::DSizes: pad the rest of the sizes with 1
   */
  template <int NDIMS, typename IndexType>
  Eigen::DSizes<IndexType, NDIMS> AsEigenDSizesWithPadding() const {
    Eigen::DSizes<IndexType, NDIMS> dsizes;
    for (size_t d = 0; d < tensor_shape_.size(); d++) {
      dsizes[d] = static_cast<IndexType>(tensor_shape_[d]);
    }
    for (size_t d = tensor_shape_.size(); d < NDIMS; d++) {
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
  Tensor *tensor_;
  ShapeVector tensor_shape_;
  void *tensor_data_;
};
}  // namespace aicpu

#endif  // AICPU_EIGENTENSOR_H
