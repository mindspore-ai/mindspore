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

#include "parallel/tensor_layout/array.h"
#include <utility>
#include "parallel/status.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {

std::string Array::ToString() const {
  std::ostringstream buffer;
  buffer << "[ ";
  for (auto& element : array_) {
    buffer << std::to_string(element) + " ";
  }
  buffer << "]";
  return buffer.str();
}

Status Array::Init(const std::vector<int32_t>& array) {
  array_ = array;
  return IsvalidArray() ? Status::SUCCESS : Status::FAILED;
}

bool Array::IsvalidArray() const { return true; }

int32_t Array::GetDimByIdx(uint32_t idx) const {
  size_t mod_idx = idx;
  if (idx >= GetDimSize()) {
    MS_LOG(EXCEPTION) << "idx is " << idx << ", but array size is " << GetDimSize();
  }
  return array_[mod_idx];
}

int32_t Array::GetDimByReverseIdx(uint32_t idx) const {
  size_t mod_idx = idx;
  if (idx >= GetDimSize()) {
    MS_LOG(EXCEPTION) << "idx is " << idx << " but array size is " << GetDimSize();
  }
  return array_[GetDimSize() - 1 - mod_idx];
}

bool Array::operator==(const Array& shape) const {
  if (GetDimSize() != shape.GetDimSize()) {
    return false;
  }
  for (uint32_t i = 0; i < GetDimSize(); i++) {
    if (GetDimByIdx(i) != shape.GetDimByIdx(i)) {
      return false;
    }
  }
  return true;
}
}  // namespace parallel
}  // namespace mindspore
