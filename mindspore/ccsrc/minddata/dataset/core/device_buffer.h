/**
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DEVICE_BUFFER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DEVICE_BUFFER_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace dataset {
class DeviceBuffer : public std::enable_shared_from_this<DeviceBuffer> {
 public:
  DeviceBuffer() = delete;

  explicit DeviceBuffer(const std::vector<size_t> &shape);

  DeviceBuffer(const DeviceBuffer &other) = delete;

  DeviceBuffer(DeviceBuffer &&other);

  DeviceBuffer &operator=(const DeviceBuffer &other) = delete;

  DeviceBuffer &operator=(DeviceBuffer &&other);

  ~DeviceBuffer();

  /// \brief Get the size of the DeviceBuffer in bytes.
  /// \return The size of the DeviceBuffer in bytes.
  size_t GetBufferSize() const;

  /// \brief Get the buffer pointer of the DeviceBuffer.
  /// \return The buffer pointer of the DeviceBuffer.
  void *GetBuffer();

  /// \brief Create a view of an existing DeviceBuffer.
  /// \param[in] other The DeviceBuffer to view.
  /// \param[in] offset The offset in the DeviceBuffer to start the view.
  DeviceBuffer(const std::shared_ptr<DeviceBuffer> &other, ptrdiff_t offset);

  /// \brief Get the shape of the DeviceBuffer.
  /// \return The shape of the DeviceBuffer.
  std::vector<size_t> GetShape();

  /// \brief Get the strides of the DeviceBuffer.
  /// \return The strides of the DeviceBuffer.
  std::vector<size_t> GetStrides();

 private:
  std::vector<size_t> shape_;
  void *ptr_;
  size_t size_;
  bool own_data_;
  std::vector<size_t> strides_;
  device::DeviceContext *device_context_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_DEVICE_BUFFER_H_
