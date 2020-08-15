/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_INCLUDE_MS_TENSOR_H_
#define MINDSPORE_INCLUDE_MS_TENSOR_H_

#include <utility>
#include <vector>
#include <memory>
#include "ir/dtype/type_id.h"

namespace mindspore {
#define MS_API __attribute__((visibility("default")))
namespace tensor {
/// \brief MSTensor defined tensor in MindSpore Lite.
class MS_API MSTensor {
 public:
  /// \brief Constructor of MindSpore Lite MSTensor.
  ///
  /// \return Instance of MindSpore Lite MSTensor.
  MSTensor() = default;

  /// \brief Static method to create a MSTensor pointer.
  ///
  /// \param[in] data_type Define data type of tensor to be created.
  /// \param[in] shape Define Shape of tensor to be created.
  ///
  /// \note TypeId is defined in mindspore/mindspore/core/ir/dtype/type_id.h. Only number types in TypeId enum are
  /// suitable for MSTensor.
  ///
  /// \return the pointer of MSTensor.
  static MSTensor *CreateTensor(TypeId data_type, const std::vector<int> &shape);

  /// \brief Destructor of MindSpore Lite Model.
  virtual ~MSTensor() = default;

  /// \brief Get data type of the MindSpore Lite MSTensor.
  ///
  /// \note TypeId is defined in mindspore/mindspore/core/ir/dtype/type_id.h. Only number types in TypeId enum are
  /// suitable for MSTensor.
  ///
  /// \return MindSpore Lite TypeId of the MindSpore Lite MSTensor.
  virtual TypeId data_type() const = 0;

  /// \brief Set data type for the MindSpore Lite MSTensor.
  ///
  /// \param[in] data_type Define MindSpore Lite TypeId to be set in the MindSpore Lite MSTensor.
  ///
  /// \return MindSpore Lite TypeId of the MindSpore Lite MSTensor after set.
  virtual TypeId set_data_type(TypeId data_type) = 0;

  /// \brief Get shape of the MindSpore Lite MSTensor.
  ///
  /// \return A vector of int as the shape of the MindSpore Lite MSTensor.
  virtual std::vector<int> shape() const = 0;

  /// \brief Set shape for the MindSpore Lite MSTensor.
  ///
  /// \param[in] shape Define a vector of int as shape to be set into the MindSpore Lite MSTensor.
  ///
  /// \return size of shape of the MindSpore Lite MSTensor after set.
  virtual size_t set_shape(const std::vector<int> &shape) = 0;

  /// \brief Get size of the dimension of the MindSpore Lite MSTensor index by the parameter index.
  ///
  /// \param[in] index Define index of dimension returned.
  ///
  /// \return Size of dimension of the MindSpore Lite MSTensor.
  virtual int DimensionSize(size_t index) const = 0;

  /// \brief Get number of element in MSTensor.
  ///
  /// \return Number of element in MSTensor.
  virtual int ElementsNum() const = 0;

  /// \brief Get hash of the MindSpore Lite MSTensor.
  ///
  /// \return Hash of the MindSpore Lite MSTensor.
  virtual std::size_t hash() const = 0;

  /// \brief Get byte size of data in MSTensor.
  ///
  /// \return Byte size of data in MSTensor.
  virtual size_t Size() const = 0;

  /// \brief Get the pointer of data in MSTensor.
  ///
  /// \note The data pointer can be used to both write and read data in MSTensor.
  ///
  /// \return the pointer points to data in MSTensor.
  virtual void *MutableData() const = 0;
};
}  // namespace tensor
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_MS_TENSOR_H_
