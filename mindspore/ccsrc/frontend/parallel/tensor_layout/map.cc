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

#include "frontend/parallel/tensor_layout/map.h"
#include <algorithm>
#include <utility>
#include "utils/ms_utils.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/shape_util.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status Map::Init(const Shape &array) {
  Status status = Array::Init(array);
  if (status != Status::SUCCESS) {
    return Status::FAILED;
  }
  if (!IsValidMap()) {
    MS_LOG(ERROR) << "invalid map " << this->ToString();
    return Status::FAILED;
  }
  return Status::SUCCESS;
}

bool Map::IsValidMap() {
  if (std::any_of(array_.begin(), array_.end(), [](int64_t value) { return ((value < 0) && (value != MAP_NONE)); })) {
    return false;
  }
  // check that all none -1 value in array_ is different
  Shape sorted_array = array_;
  std::sort(sorted_array.begin(), sorted_array.end());
  int64_t value = MAP_NONE;
  for (auto &element : sorted_array) {
    if (element == MAP_NONE) {
      continue;
    }
    if (element == value) {
      return false;
    }
    value = element;
  }
  return true;
}

int64_t Map::GetMaxItem() const {
  if (!array_.empty()) {
    return *std::max_element(array_.begin(), array_.end());
  } else {
    return MAP_NONE;
  }
}

int64_t Map::GetIndexByValue(int64_t value) const {
  auto iter = find(array_.begin(), array_.end(), value);
  if (iter != array_.end()) {
    return static_cast<int64_t>(std::distance(array_.begin(), iter));
  } else {
    return MAP_NONE;
  }
}

/*
 * expand.size() should be equal to array_.size()
 */
std::shared_ptr<Map> Map::ExpandMapByNone(const Arrangement &expand_num_list) const {
  if (expand_num_list.GetDimSize() != GetDimSize()) {
    return nullptr;
  }
  Shape new_shape;
  for (size_t i = 0; i != GetDimSize(); i++) {
    if (GetDimByIdx(i) == MAP_NONE) {
      for (int64_t j = 0; j < expand_num_list.GetDimByIdx(i); j++) {
        new_shape.push_back(MAP_NONE);
      }
    } else {
      new_shape.push_back(GetDimByIdx(i));
      int64_t j = 1;
      while (j < expand_num_list.GetDimByIdx(i)) {
        new_shape.push_back(MAP_NONE);
        j++;
      }
    }
  }
  auto map_new = std::make_shared<Map>();
  (void)map_new->Init(new_shape);
  return map_new;
}

/*
 * expand.size() should be equal to array_.size()
 */
std::shared_ptr<Map> Map::ExpandMapByDecreaseNumber(const Arrangement &expand_num_list) const {
  if (GetMaxItem() >= static_cast<int64_t>(expand_num_list.GetDimSize())) {
    return nullptr;
  }
  Shape new_shape;
  for (size_t i = 0; i < GetDimSize(); i++) {
    if (GetDimByIdx(i) == MAP_NONE) {
      new_shape.push_back(MAP_NONE);
    } else {
      int64_t start_map =
        expand_num_list.ComputeReverseAccumulateSumInReverseOrder()[static_cast<size_t>(GetDimByIdx(i))];
      for (int64_t k = expand_num_list.GetDimByReverseIdx(static_cast<size_t>(GetDimByIdx(i))) - 1; k >= 0; k--) {
        new_shape.push_back(k + start_map);
      }
    }
  }
  auto map_new = std::make_shared<Map>();
  (void)map_new->Init(new_shape);
  return map_new;
}

std::shared_ptr<std::vector<Arrangement>> Map::ReMapVector(const std::vector<Arrangement> &input_vector) const {
  if (GetMaxItem() >= static_cast<int64_t>(input_vector.size())) {
    return nullptr;
  }
  std::vector<Arrangement> out;
  Arrangement empty_arrangement;
  for (size_t i = 0; i < GetDimSize(); i++) {
    if (GetDimByIdx(i) == MAP_NONE) {
      out.push_back(empty_arrangement);
    } else {
      out.push_back(input_vector[input_vector.size() - 1 - LongToSize(GetDimByIdx(i))]);
    }
  }
  return std::make_shared<std::vector<Arrangement>>(out);
}

bool Map::CheckNoneByIdxList(std::vector<size_t> idx_list) const {
  for (auto &value : idx_list) {
    if (GetDimByIdx(value) != MAP_NONE) {
      return false;
    }
  }
  return true;
}

Map Map::SqueezeMapByIdxList(std::vector<size_t> idx_list) const {
  Shape out_shape;
  for (size_t i = 0; i < GetDimSize(); i++) {
    auto it = std::find(idx_list.begin(), idx_list.end(), i);
    if (it == idx_list.end()) {
      out_shape.push_back(GetDimByIdx(i));
    }
  }
  if (out_shape.empty()) {
    MS_LOG(ERROR) << "out_shape size is 0, this may not happen under current situation";
    out_shape.push_back(MAP_NONE);
  }
  Map out;
  (void)out.Init(out_shape);
  return out;
}
}  // namespace parallel
}  // namespace mindspore
