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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_DEVICE_MATRIX_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_DEVICE_MATRIX_H_

#include <cstdint>
#include <string>
#include <vector>

#include "frontend/parallel/status.h"
#include "include/common/utils/convert_utils.h"

namespace mindspore {
namespace parallel {
using RankList = std::vector<int64_t>;
using Shape = std::vector<int64_t>;

class DeviceMatrix {
 public:
  DeviceMatrix(int64_t rank, RankList dev_list, Shape dev_shape);
  DeviceMatrix() = default;
  ~DeviceMatrix() = default;
  std::vector<RankList> group_list() const { return group_list_; }
  Status CreateGroupList();
  Status GetDevicesByTensorMap(const Shape &tensor_map, RankList *rank_list);
  Status GetDevicesAlongDim(const uint64_t &dim, RankList *devices);
  Status GetDevicesAlongMultiDim(const std::vector<int64_t> &dims, RankList *devices);

 private:
  int64_t rank_ = -1;
  RankList dev_list_;
  // From low dim to high dim. eg: [D0 D1 D2 D3]
  Shape dev_shape_;
  std::vector<RankList> group_list_;
};

std::string ShapeToString(const Shape &shape);
std::string ListToString(const RankList &list);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_DEVICE_MATRIX_H_
