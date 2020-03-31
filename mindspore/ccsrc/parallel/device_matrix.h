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

#ifndef MINDSPORE_CCSRC_PARALLEL_DEVICE_MATRIX_H_
#define MINDSPORE_CCSRC_PARALLEL_DEVICE_MATRIX_H_

#include <cstdint>
#include <string>
#include <vector>

#include "parallel/status.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace parallel {

using RankList = std::vector<int32_t>;
using Shape = std::vector<int32_t>;

class DeviceMatrix {
 public:
  DeviceMatrix(int32_t rank, RankList devices, Shape dev_shape);
  DeviceMatrix() = default;
  ~DeviceMatrix() = default;
  std::vector<RankList> group_list() const { return group_list_; }
  Status CreateGroupList();
  Status GetDevicesByTensorMap(const Shape& tensor_map, RankList* rank_list);
  Status GetDevicesAlongDim(const uint32_t& dim, RankList* devices);

 private:
  int32_t rank_ = -1;
  RankList dev_list_;
  // From low dim to high dim. eg: [D0 D1 D2 D3]
  Shape dev_shape_;
  std::vector<RankList> group_list_;
};

std::string ShapeToString(const Shape& shape);
std::string ListToString(const std::vector<int32_t>& list);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_DEVICE_MATRIX_H_
