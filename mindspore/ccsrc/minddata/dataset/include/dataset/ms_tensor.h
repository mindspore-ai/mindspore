/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_MS_TENSOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_MS_TENSOR_H_

#include <string>
#include <vector>
#include <memory>

#include "mindspore/ccsrc/include/utils/data_type.h"
#include "utils/dual_abi_helper.h"

namespace mindspore {
/// \brief The MSTensor class defines a tensor in MindSpore.
class MSTensor {
 public:
  /// \brief Impl class of MSTensor.
  class Impl;

  /// \brief Constructor of MSTensor.
  MSTensor();

  /// \brief Constructor of MSTensor.
  explicit MSTensor(const std::shared_ptr<Impl> &impl);

  /// \brief Constructor of MSTensor.
  inline MSTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, const void *data,
                  size_t data_len);

  /// \brief Destructor of MSTensor.
  ~MSTensor();

  /// \brief Obtains the data type of the MSTensor.
  /// \return The data type of the MSTensor.
  enum DataType DataType() const;

  /// \brief Obtains the shape of the MSTensor.
  /// \return The shape of the MSTensor.
  const std::vector<int64_t> &Shape() const;

  /// \brief Obtains a shared pointer to the copy of data of the MSTensor. The data can be read on host.
  /// \return A shared pointer to the copy of data of the MSTensor.
  std::shared_ptr<const void> Data() const;

  /// \brief Obtains the pointer to the data of the MSTensor. If the MSTensor is a device tensor, the data cannot be
  ///     accessed directly on host.
  /// \return A pointer to the data of the MSTensor.
  void *MutableData();

  /// \brief Obtains the length of the data of the MSTensor, in bytes.
  /// \return The length of the data of the MSTensor, in bytes.
  size_t DataSize() const;

 private:
  MSTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape, const void *data,
           size_t data_len);

  std::shared_ptr<Impl> impl_;
};

MSTensor::MSTensor(const std::string &name, enum DataType type, const std::vector<int64_t> &shape, const void *data,
                   size_t data_len)
    : MSTensor(StringToChar(name), type, shape, data, data_len) {}

class Buffer {
 public:
  Buffer();
  Buffer(const void *data, size_t data_len);
  ~Buffer();
  const void *Data() const;
  void *MutableData();
  size_t DataSize() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

class MSTensor::Impl {
 public:
  virtual ~Impl() = default;
  virtual enum DataType DataType() const = 0;
  virtual const std::vector<int64_t> &Shape() const = 0;
  virtual std::shared_ptr<const void> Data() const = 0;
  virtual void *MutableData() = 0;
  virtual size_t DataSize() const = 0;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_MS_TENSOR_H_
