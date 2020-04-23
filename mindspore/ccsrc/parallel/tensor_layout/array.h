/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_ARRAY_H_
#define MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_ARRAY_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "parallel/status.h"

namespace mindspore {
namespace parallel {
class Array {
 public:
  Array() = default;
  virtual ~Array() = default;
  std::string ToString() const;
  virtual Status Init(const std::vector<int32_t>& array);
  bool IsvalidArray() const;
  std::vector<int32_t> array() const { return array_; }
  size_t GetDimSize() const { return array_.size(); }
  int32_t GetDimByIdx(uint32_t idx) const;
  int32_t GetDimByReverseIdx(uint32_t idx) const;
  bool operator==(const Array& a1) const;

 protected:
  std::vector<int32_t> array_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_ARRAY_H_
