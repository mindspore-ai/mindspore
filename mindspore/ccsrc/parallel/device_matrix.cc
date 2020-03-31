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

#include "parallel/device_matrix.h"

#include <cstdint>
#include <algorithm>
#include <utility>
#include <numeric>
#include <functional>
#include <vector>

#include "parallel/status.h"
#include "parallel/ops_info/operator_info.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {

DeviceMatrix::DeviceMatrix(int32_t rank, RankList dev_list, Shape dev_shape)
    : rank_(rank), dev_list_(std::move(dev_list)), dev_shape_(std::move(dev_shape)) {
  if (!std::any_of(dev_list_.begin(), dev_list_.end(), [rank](int32_t a) { return a == rank; })) {
    MS_LOG(EXCEPTION) << "Rank " << rank << " is not in the current stage!";
  }
  int32_t total = std::accumulate(dev_shape_.begin(), dev_shape_.end(), 1, std::multiplies<int>());
  if (IntToSize(total) != dev_list_.size()) {
    MS_LOG(EXCEPTION) << "Device shape does not match the size of the device list!";
  }
}

Status DeviceMatrix::CreateGroupList() {
  size_t size = dev_shape_.size();
  RankList group;
  for (size_t i = 0; i < size; i++) {
    Status status = GetDevicesAlongDim(SizeToUint(i), &group);
    group_list_.push_back(group);
    if (status == Status::FAILED) {
      return Status::FAILED;
    }
  }
  return Status::SUCCESS;
}

Status DeviceMatrix::GetDevicesAlongDim(const uint32_t& dim, RankList* devices) {
  if (dim >= dev_shape_.size()) {
    MS_LOG(EXCEPTION) << "The dimension " << dim << " is out of the size of the device shape!";
  }
  if (dev_shape_[dim] == 1) {
    *devices = {rank_};
    return Status::SUCCESS;
  }

  RankList group;
  std::vector<RankList> local_group_list;

  // lower than dim
  int32_t step = 1;
  for (uint32_t i = dim + 1; i < dev_shape_.size(); i++) {
    step = step * dev_shape_[i];
  }
  int32_t num = *dev_list_.begin();
  for (int32_t i = 0; i < dev_shape_[dim]; i++) {
    group.push_back(num);
    num += step;
  }

  for (int32_t i = 0; i < step; i++) {
    local_group_list.push_back(group);
    (void)std::for_each(group.begin(), group.end(), [](int32_t& a) { a++; });
  }

  // higher than dim
  step = step * dev_shape_[dim];
  int32_t len = SizeToInt(dev_list_.size()) / step;

  // search rank
  int32_t target = rank_;
  for (int32_t i = 0; i < len; i++) {
    for (RankList& temp : local_group_list) {
      if (std::any_of(temp.begin(), temp.end(), [target](int32_t a) { return a == target; })) {
        *devices = temp;
        return Status::SUCCESS;
      }
      (void)std::for_each(temp.begin(), temp.end(), [step](int32_t& a) { a = a + step; });
    }
  }
  MS_LOG(ERROR) << "Can't find groups for rank" << rank_ << " in device list!";
  return Status::FAILED;
}

Shape ConvertRankToCoordinate(int32_t rank, const Shape& dev_shape) {
  Shape dev_coordinate;
  for (size_t i = 0; i < dev_shape.size(); ++i) {
    int32_t size = dev_shape[dev_shape.size() - i - 1];
    if (size == 0) {
      MS_LOG(EXCEPTION) << "Invalid dev shape: " << ShapeToString(dev_shape);
    } else {
      int32_t index = rank % size;
      (void)dev_coordinate.insert(dev_coordinate.begin(), index);
      rank = rank / size;
    }
  }
  return dev_coordinate;
}

Status DeviceMatrix::GetDevicesByTensorMap(const Shape& tensor_map, RankList* rank_list) {
  for (auto& element : tensor_map) {
    // -1 means the corresponding dimension is not split.
    if (element == MAP_NONE) {
      continue;
    } else if ((element < 0) || (IntToSize(element) >= dev_shape_.size())) {
      MS_LOG(ERROR) << "create group by tensor map: the tensor map is invalid";
      return FAILED;
    }
  }

  Shape current_rank_coordinate = ConvertRankToCoordinate(rank_, dev_shape_);
  for (auto& tmp_rank : dev_list_) {
    Shape tmp_rank_coordinate = ConvertRankToCoordinate(tmp_rank, dev_shape_);
    bool matched = true;
    for (auto& map : tensor_map) {
      if (map == MAP_NONE) {
        continue;
      }
      size_t index = dev_shape_.size() - IntToSize(map) - 1;
      if (current_rank_coordinate[index] != tmp_rank_coordinate[index]) {
        matched = false;
        break;
      }
    }
    if (matched) {
      rank_list->push_back(tmp_rank);
    }
  }

  return SUCCESS;
}

std::string ShapeToString(const Shape& shape) {
  std::string str = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    str += std::to_string(shape[i]);
    if (i < shape.size() - 1) {
      str += ", ";
    }
  }
  return str + "]";
}

std::string ListToString(const std::vector<int32_t>& list) {
  std::string str = "[";
  for (auto& element : list) {
    str += std::to_string(element) + ", ";
  }
  return str + "]";
}
}  // namespace parallel
}  // namespace mindspore
