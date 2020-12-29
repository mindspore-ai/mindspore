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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_TASK_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_TASK_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/common/shard_utils.h"

namespace mindspore {
namespace mindrecord {
class __attribute__((visibility("default"))) ShardTask {
 public:
  ShardTask();

  ShardTask(const ShardTask &task);  // copy construction

  ShardTask &operator=(const ShardTask &task);  // assignment operator

  ~ShardTask() = default;

  void MakePerm();

  inline void InsertTask(TaskType task_type, int shard_id, int group_id, const std::vector<uint64_t> &offset,
                         const json &label);

  inline void InsertTask(const uint32_t &i, TaskType task_type, int shard_id, int group_id,
                         const std::vector<uint64_t> &offset, const json &label);

  inline void InsertTask(std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> task);

  inline void InsertTask(const uint32_t &i,
                         std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> task);

  void PopBack();

  uint32_t Size() const;

  uint32_t SizeOfRows() const;

  std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> &GetTaskByID(size_t id);

  std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> &GetRandomTask();

  static ShardTask Combine(std::vector<ShardTask> &category_tasks, bool replacement, int64_t num_elements,
                           int64_t num_samples);

  inline void ResizeTask(const uint32_t &size);

  uint32_t categories;

  // The total sample ids which used to shuffle operation. The ids like: [0, 1, 2, 3, ..., n-1, n]
  std::vector<int> permutation_;

  // The data struct is as below:
  // 1. TaskType: kCommonTask / kPaddedTask
  // 2. std::tuple<int, int> : shard_id, group_id(fast load) / sample_id(lazy load)
  // 3. std::vector<uint64_t>, json>> : [blob_start, blob_end], scalar_variable_fields
  std::vector<std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json>> task_list_;
};

inline void ShardTask::InsertTask(TaskType task_type, int shard_id, int group_id, const std::vector<uint64_t> &offset,
                                  const json &label) {
  MS_LOG(DEBUG) << "Into insert task, shard_id: " << shard_id << ", group_id: " << group_id
                << ", label: " << label.dump() << ", size of task_list_: " << task_list_.size() << ".";
  task_list_.emplace_back(task_type, std::make_tuple(shard_id, group_id), offset, label);
}

inline void ShardTask::InsertTask(const uint32_t &i, TaskType task_type, int shard_id, int group_id,
                                  const std::vector<uint64_t> &offset, const json &label) {
  task_list_[i] = {task_type, std::make_tuple(shard_id, group_id), offset, label};
}

inline void ShardTask::InsertTask(std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> task) {
  MS_LOG(DEBUG) << "Into insert task, shard_id: " << std::get<0>(std::get<1>(task))
                << ", group_id: " << std::get<1>(std::get<1>(task)) << ", label: " << std::get<3>(task).dump()
                << ", size of task_list_: " << task_list_.size() << ".";

  task_list_.push_back(std::move(task));
}

inline void ShardTask::InsertTask(const uint32_t &i,
                                  std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> task) {
  task_list_[i] = std::move(task);
}

inline void ShardTask::ResizeTask(const uint32_t &size) { task_list_.resize(size); }
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_TASK_H_
