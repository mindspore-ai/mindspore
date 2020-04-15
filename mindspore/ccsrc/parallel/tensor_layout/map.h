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

#ifndef MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_MAP_H_
#define MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_MAP_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "parallel/status.h"
#include "parallel/tensor_layout/arrangement.h"
#include "parallel/tensor_layout/array.h"

namespace mindspore {
namespace parallel {
constexpr int32_t MAP_NONE = -1;

class Map : public Array {
 public:
  Map() = default;
  ~Map() override = default;
  Status Init(const std::vector<int32_t>& array) override;
  int32_t GetMaxItem() const;
  int32_t GetIndexByValue(int32_t value) const;
  std::shared_ptr<Map> ExpandMapByNone(const Arrangement& expand_num_list) const;
  std::shared_ptr<Map> ExpandMapByDecreaseNumber(const Arrangement& expand_num_list) const;
  std::shared_ptr<std::vector<Arrangement>> ReMapVector(const std::vector<Arrangement>& input_vector) const;
  bool CheckNoneByIdxList(std::vector<size_t> idx_list) const;
  Map SqueezeMapByIdxList(std::vector<size_t> idx_list) const;

 private:
  bool IsValidMap();
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_MAP_H_
