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
#ifndef MINDSPORE_INCLUDE_API_TYPES_H
#define MINDSPORE_INCLUDE_API_TYPES_H

#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "include/api/data_type.h"

#ifdef _WIN32
#define MS_API __declspec(dllexport)
#else
#define MS_API __attribute__((visibility("default")))
#endif

namespace mindspore {
enum ModelType : uint32_t {
  kMindIR = 0,
  kAIR = 1,
  kOM = 2,
  kONNX = 3,
  // insert new data type here
  kUnknownType = 0xFFFFFFFF
};

class MS_API MSTensor {
 public:
  class Impl;

  static MSTensor CreateTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                               const void *data, size_t data_len) noexcept;
  static MSTensor CreateRefTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                                  const void *data, size_t data_len) noexcept;

  MSTensor();
  explicit MSTensor(const std::shared_ptr<Impl> &impl);
  MSTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, const void *data,
           size_t data_len);
  ~MSTensor();

  const std::string &Name() const;
  enum DataType DataType() const;
  const std::vector<int64_t> &Shape() const;
  int64_t ElementNum() const;

  std::shared_ptr<const void> Data() const;
  void *MutableData();
  size_t DataSize() const;

  bool IsDevice() const;

  MSTensor Clone() const;
  bool operator==(std::nullptr_t) const;

 private:
  friend class ModelImpl;
  explicit MSTensor(std::nullptr_t);
  std::shared_ptr<Impl> impl_;
};

class MS_API Buffer {
 public:
  Buffer();
  Buffer(const void *data, size_t data_len);
  ~Buffer();

  const void *Data() const;
  void *MutableData();
  size_t DataSize() const;

  bool ResizeData(size_t data_len);
  bool SetData(const void *data, size_t data_len);

  Buffer Clone() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_TYPES_H
