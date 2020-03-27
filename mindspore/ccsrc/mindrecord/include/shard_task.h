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

#ifndef MINDRECORD_INCLUDE_SHARD_TASK_H_
#define MINDRECORD_INCLUDE_SHARD_TASK_H_

#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include "mindrecord/include/common/shard_utils.h"

namespace mindspore {
namespace mindrecord {
class ShardTask {
 public:
  void MakePerm();

  void InsertTask(int shard_id, int group_id, const std::vector<uint64_t> &offset, const json &label);

  void InsertTask(std::tuple<std::tuple<int, int>, std::vector<uint64_t>, json> task);

  void PopBack();

  uint32_t Size() const;

  uint32_t SizeOfRows() const;

  std::tuple<std::tuple<int, int>, std::vector<uint64_t>, json> &get_task_by_id(size_t id);

  static ShardTask Combine(std::vector<ShardTask> &category_tasks);

  uint32_t categories = 1;

  std::vector<std::tuple<std::tuple<int, int>, std::vector<uint64_t>, json>> task_list_;
  std::vector<int> permutation_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDRECORD_INCLUDE_SHARD_TASK_H_
