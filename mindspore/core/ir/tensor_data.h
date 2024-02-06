/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_TENSOR_DATA_H_
#define MINDSPORE_CORE_IR_TENSOR_DATA_H_

#include <memory>
#include <string>
#include "mindapi/base/macros.h"
#include "utils/os.h"

namespace mindspore::tensor {
// Tensor data interface.
class MS_CORE_API TensorData {
 public:
  /// \brief Virtual destructor is required for base classes.
  virtual ~TensorData() = default;

  /// \brief Get total number of elements.
  ///
  /// \return Total number of elements.
  virtual ssize_t size() const = 0;

  /// \brief Get byte size of a single element.
  ///
  /// \return Byte size of a single element.
  virtual ssize_t itemsize() const = 0;

  /// \brief Get total number of bytes.
  ///
  /// \return Total number of bytes.
  virtual ssize_t nbytes() const = 0;

  /// \brief Get number of dimensions.
  ///
  /// \return Number of dimensions.
  virtual ssize_t ndim() const = 0;

  /// \brief Get data pointer.
  ///
  /// \return Data pointer.
  virtual void *data() = 0;

  /// \brief Get const data pointer.
  ///
  /// \return Const data pointer.
  virtual const void *const_data() const = 0;

  /// \brief Get whether this tensor data is sub data.
  ///
  /// \return Whether this tensor data is sub data.
  virtual bool is_sub_data() const = 0;

  /// \brief Check whether this tensor data has sub data.
  ///
  /// \return True if this tensor data has sub data, otherwise false.
  virtual bool has_sub_data() const = 0;

  /// \brief Get whether this tensor data is from numpy.
  ///
  /// \return Whether this tensor data is from numpy.
  virtual bool is_from_numpy() const { return false; }

  /// \brief Get whether this tensor data have use persistent storage to save data.
  ///
  /// \return Whether this tensor data have use persistent storage to save data.
  virtual bool is_persistent_data() const { return false; }

  /// \brief Whether the data are equal.
  ///
  /// \param[in] other Another TensorData.
  /// \return Ture if the two data are equal, otherwise false.
  virtual bool equals(const TensorData &other) const {
    if (this == &other) {
      return true;
    }
    // By default, compare data byte by byte.
    auto this_data = static_cast<const uint8_t *>(const_data());
    auto other_data = static_cast<const uint8_t *>(other.const_data());
    if (this_data == nullptr || other_data == nullptr) {
      // null means data not initialized, compare uninitialized data always return false.
      return false;
    }
    return (this_data == other_data) || (ndim() == other.ndim() && nbytes() == other.nbytes() &&
                                         std::equal(this_data, this_data + nbytes(), other_data));
  }

  /// \brief Get display information about this TensorData.
  ///
  /// \param[in] type The type of tensor data.
  /// \param[in] shape The shape of tensor data.
  /// \param[in] use_comma Whether to use comma.
  /// \return The display information.
  virtual std::string ToString(TypeId type, const ShapeVector &shape, bool use_comma) const = 0;

  /// \brief Set data saved file path.
  ///
  /// \param[in] data file path.
  /// \return Void.
  virtual void set_file_path(const std::string &path) {
    MS_LOG(INFO) << "Call default set file path, and do nothing with " << path << ".";
  }

  /// \brief Get data saved file path.
  ///
  /// \return data file path.
  virtual const std::string file_path() const { return ""; }
};

using TensorDataPtr = std::shared_ptr<TensorData>;
}  // namespace mindspore::tensor
#endif  // MINDSPORE_CORE_IR_TENSOR_DATA_H_
