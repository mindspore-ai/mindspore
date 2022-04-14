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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_PARAM_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_PARAM_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

namespace mindspore {
namespace lite {
class Param {
 public:
  enum Mode { NORMAL, UNIFORM, ONES, ZEROS, NOT_SUPPORTED };
  int Fill(Mode type);
  static Mode String2Enum(std::string);
  std::vector<uint8_t> &data() { return data_; }
  size_t Load(std::string file_name, size_t offset = 0) { return data_.size() * sizeof(float); }
  size_t Load(std::ifstream &s, int offset = 0) { return data_.size() * sizeof(float); }
  void SetSize(size_t size) { size_ = size; }
  template <typename T>
  void Copy(const T *data, size_t size) {
    auto cast_data = reinterpret_cast<const uint8_t *>(data);
    data_ = decltype(data_)(cast_data, cast_data + size * sizeof(T) / sizeof(uint8_t));
  }
  template <typename T>
  void Copy(const std::vector<T> data) {
    Copy<T>(data.data(), data.size());
  }

  template <typename T>
  std::vector<T> Extract() {
    T *num = reinterpret_cast<T *>(data_.data());
    std::vector<T> res(num, num + data_.size() / sizeof(T));
    return res;
  }

 private:
  size_t size_;
  std::vector<uint8_t> data_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXPRESSION_PARAM_H_
