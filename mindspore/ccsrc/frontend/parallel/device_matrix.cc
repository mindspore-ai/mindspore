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

#include "frontend/parallel/device_matrix.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "frontend/parallel/status.h"
#include "utils/log_adapter.h"
#include "frontend/parallel/tensor_layout/map.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace parallel {
DeviceMatrix::DeviceMatrix(int64_t rank, RankList dev_list, Shape dev_shape)
    : rank_(rank), dev_list_(std::move(dev_list)), dev_shape_(std::move(dev_shape)) {
  if (!std::any_of(dev_list_.begin(), dev_list_.end(), [rank](int64_t a) { return a == rank; })) {
    MS_LOG(EXCEPTION) << "Rank " << rank << " is not in the current stage!";
  }
  int64_t total = std::accumulate(dev_shape_.begin(), dev_shape_.end(), 1, std::multiplies<int64_t>());
  if (LongToSize(total) != dev_list_.size()) {
    MS_LOG(EXCEPTION) << "Device shape does not match the size of the device list!";
  }
}

Status DeviceMatrix::CreateGroupList() {
  size_t size = dev_shape_.size();
  RankList group;
  for (size_t i = 0; i < size; i++) {
    Status status = GetDevicesAlongDim(SizeToUlong(i), &group);
    group_list_.push_back(group);
    if (status == Status::FAILED) {
      return Status::FAILED;
    }
  }
  return Status::SUCCESS;
}

Status DeviceMatrix::GetDevicesAlongDim(const uint64_t &dim, RankList *devices) {
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
  int64_t step = 1;
  for (uint64_t i = dim + 1; i < dev_shape_.size(); i++) {
    step = step * dev_shape_[i];
  }
  int64_t num = *dev_list_.begin();
  for (int64_t i = 0; i < dev_shape_[dim]; i++) {
    group.push_back(num);
    num += step;
  }

  for (int64_t i = 0; i < step; i++) {
    local_group_list.push_back(group);
    (void)std::for_each(group.begin(), group.end(), [](int64_t &a) { a++; });
  }

  // higher than dim
  step = step * dev_shape_[dim];
  int64_t len = SizeToLong(dev_list_.size()) / step;

  // search rank
  int64_t target = rank_;
  for (int64_t i = 0; i < len; i++) {
    for (RankList &temp : local_group_list) {
      if (std::any_of(temp.begin(), temp.end(), [target](int64_t a) { return a == target; })) {
        *devices = temp;
        return Status::SUCCESS;
      }
      (void)std::for_each(temp.begin(), temp.end(), [step](int64_t &a) { a = a + step; });
    }
  }
  MS_LOG(ERROR) << "Can't find groups for rank" << rank_ << " in device list!";
  return Status::FAILED;
}

Shape ConvertRankToCoordinate(int64_t rank, const Shape &dev_shape) {
  Shape dev_coordinate;
  for (size_t i = 0; i < dev_shape.size(); ++i) {
    int64_t size = dev_shape[dev_shape.size() - i - 1];
    if (size == 0) {
      MS_LOG(EXCEPTION) << "Invalid dev shape: " << ShapeToString(dev_shape);
    } else {
      int64_t index = rank % size;
      (void)dev_coordinate.insert(dev_coordinate.cbegin(), index);
      rank = rank / size;
    }
  }
  return dev_coordinate;
}

Status DeviceMatrix::GetDevicesByTensorMap(const Shape &tensor_map, RankList *rank_list) {
  for (auto &element : tensor_map) {
    // -1 means the corresponding dimension is not split.
    if (element == MAP_NONE) {
      continue;
    } else if ((element < 0) || (LongToSize(element) >= dev_shape_.size())) {
      MS_LOG(ERROR) << "create group by tensor map: the tensor map is invalid";
      return FAILED;
    }
  }

  // Convert the global rank to the local rank(The index of the array) to compute the coordinate
  uint32_t local_rank = 0;
  for (auto &tmp_rank : dev_list_) {
    if (tmp_rank == rank_) {
      break;
    }
    ++local_rank;
  }
  if (local_rank == dev_list_.size()) {
    MS_LOG(ERROR) << "Rank id: " << local_rank << "is not in the device list.";
    return FAILED;
  }

  Shape current_rank_coordinate = ConvertRankToCoordinate(static_cast<int32_t>(local_rank), dev_shape_);
  for (uint32_t loop_local_rank = 0; loop_local_rank < dev_list_.size(); ++loop_local_rank) {
    Shape tmp_rank_coordinate = ConvertRankToCoordinate(loop_local_rank, dev_shape_);
    bool matched = true;
    for (auto &map : tensor_map) {
      if (map == MAP_NONE) {
        continue;
      }
      size_t index = dev_shape_.size() - LongToSize(map) - 1;
      if (current_rank_coordinate[index] != tmp_rank_coordinate[index]) {
        matched = false;
        break;
      }
    }
    if (matched) {
      rank_list->push_back(dev_list_[loop_local_rank]);
    }
  }

  return SUCCESS;
}

std::string ShapeToString(const Shape &shape) {
  std::string str = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    str += std::to_string(shape[i]);
    if (i < shape.size() - 1) {
      str += ", ";
    }
  }
  return str + "]";
}

std::string ListToString(const RankList &list) {
  std::string str = "[";
  for (auto &element : list) {
    str += std::to_string(element) + ", ";
  }
  return str + "]";
}
}  // namespace parallel
}  // namespace mindspore
